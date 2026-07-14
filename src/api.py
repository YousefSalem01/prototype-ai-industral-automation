"""FastAPI service exposing the quality model to the desktop application.

Endpoints (all JSON in / JSON out, language-agnostic over local HTTP):
    GET  /health    -> liveness + which features the model expects
    POST /predict   -> predicted quality for a supplied process state
    POST /optimize  -> recommended controllable setpoints for fixed conditions
    POST /explain   -> SHAP drivers for a supplied process state

The request schema is built DYNAMICALLY from process_config.yaml, so when the
real factory CSV (and config) is swapped in, the API surface follows the new
columns automatically -- no edits here.

--------------------------------------------------------------------------------
EXAMPLE CALL FROM THE DESKTOP APP (any language -- shown as curl)
--------------------------------------------------------------------------------
The desktop app reads the operator's current fixed conditions and setpoints,
POSTs them, and renders the recommendation + explanation.

    curl -s -X POST http://127.0.0.1:8000/optimize \
      -H "Content-Type: application/json" \
      -d '{
            "state": {
              "feed_rate": 120.0,
              "material_moisture": 6.5,
              "ambient_temp": 18.0,
              "gas_temperature": 1180.0,
              "oxygen_pct": 4.0,
              "flow_rate": 1600.0,
              "furnace_speed": 3.2
            }
          }'

Response (abridged):
    {
      "recommended_setpoints": {"gas_temperature": 1201.3, "oxygen_pct": 3.8, ...},
      "predicted_quality": 92.4,
      "current_quality": 85.1,
      "delta": 7.3,
      "within_limits": true
    }

Equivalent from C# (desktop app):
    var payload = new { state = currentState };            // Dictionary<string,double>
    var resp = await http.PostAsJsonAsync(
        "http://127.0.0.1:8000/optimize", payload);
    var rec = await resp.Content.ReadFromJsonAsync<OptimizeResponse>();
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .explain import QualityExplainer
from .model import QualityModel, load_model
from .optimizer import optimize_setpoints
from .preprocess import clean_dataframe, split_xy
from .schema import ProcessConfig, load_config

app = FastAPI(
    title="Industrial Quality Optimizer",
    description="AI setpoint recommendations for furnace/kiln process quality.",
    version="1.0.0",
)

# Allow the local React dev server (and packaged desktop webview) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local-only service; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- lazy singletons: model + explainer loaded once, on first use ------------
class _State:
    config: ProcessConfig | None = None
    model: QualityModel | None = None
    explainer: QualityExplainer | None = None


_state = _State()


def _get_config() -> ProcessConfig:
    if _state.config is None:
        _state.config = load_config()
    return _state.config


def _get_model() -> QualityModel:
    if _state.model is None:
        try:
            _state.model = load_model(_get_config())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _state.model


def _get_explainer() -> QualityExplainer:
    if _state.explainer is None:
        _state.explainer = QualityExplainer(_get_model(), _get_config())
    return _state.explainer


# Cache for the (relatively expensive) global-importance computation.
_importance_cache: list[dict[str, Any]] | None = None


# --- request/response models -------------------------------------------------
class StateRequest(BaseModel):
    """A process state. Keys are feature names from the config.

    Partial states are allowed for /predict and /explain (missing features are
    imputed). For /optimize, all fixed-condition columns are required.
    """

    state: dict[str, float] = Field(
        ...,
        description="Mapping of feature name -> value.",
        json_schema_extra={"example": {
            "feed_rate": 120.0, "material_moisture": 6.5, "ambient_temp": 18.0,
            "gas_temperature": 1180.0, "oxygen_pct": 4.0,
            "flow_rate": 1600.0, "furnace_speed": 3.2,
        }},
    )


class OptimizeRequest(StateRequest):
    """Optimize request; optional override for the number of trials."""

    n_trials: int | None = Field(
        default=None, description="Override Optuna trial count.")


# --- endpoints ---------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check plus the contract the model currently expects."""
    cfg = _get_config()
    return {
        "status": "ok",
        "target": cfg.target,
        "direction": cfg.target_direction,
        "controllable": cfg.controllable,
        "fixed": cfg.fixed,
        "model_loaded": _state.model is not None,
    }


@app.get("/meta")
def meta() -> dict[str, Any]:
    """Full column contract, so the UI can build its inputs dynamically.

    The frontend reads this once and renders a slider/field per column with the
    right label, unit, and min/max -- so swapping the config reshapes the UI
    automatically, exactly like the backend.
    """
    cfg = _get_config()
    columns = []
    for name in cfg.features:
        spec = cfg.columns[name]
        lo, hi = cfg.bounds(name)
        columns.append({
            "name": name,
            "role": spec.role,
            "unit": spec.unit,
            "description": spec.description,
            "min": lo,
            "max": hi,
            "default": round(0.5 * (lo + hi), 3),
        })
    tgt = cfg.columns[cfg.target]
    return {
        "target": {
            "name": cfg.target, "unit": tgt.unit,
            "min": tgt.min, "max": tgt.max, "direction": cfg.target_direction,
        },
        "controllable": cfg.controllable,
        "fixed": cfg.fixed,
        "columns": columns,
    }


@app.get("/importance")
def importance() -> dict[str, Any]:
    """Dataset-wide SHAP feature importance (computed once, then cached)."""
    global _importance_cache
    if _importance_cache is None:
        cfg = _get_config()
        try:
            df = pd.read_csv(cfg.data_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503,
                                detail=f"Dataset not found: {exc}") from exc
        df_clean, _ = clean_dataframe(df, cfg)
        X, _ = split_xy(df_clean, cfg)
        _importance_cache = _get_explainer().global_importance(X)
    return {"importance": _importance_cache}


@app.post("/predict")
def predict(req: StateRequest) -> dict[str, Any]:
    """Predict quality for a supplied process state."""
    cfg = _get_config()
    model = _get_model()
    quality = model.predict(req.state)
    return {"predicted_quality": round(quality, 3), "target": cfg.target}


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    """Recommend controllable setpoints for the given fixed conditions."""
    cfg = _get_config()
    model = _get_model()
    try:
        result = optimize_setpoints(model, req.state, cfg, n_trials=req.n_trials)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.as_dict()


@app.post("/explain")
def explain(req: StateRequest) -> dict[str, Any]:
    """Return SHAP drivers explaining the prediction for a state."""
    explainer = _get_explainer()
    return explainer.explain_state(req.state)


def main() -> None:
    """Run the API with uvicorn (``python -m src.api``)."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
