"""XGBoost quality-surrogate model: train, evaluate, persist, load, predict.

The model learns the mapping (controllable + fixed features) -> quality. It is
the surrogate the optimizer searches over and that SHAP explains. Training also
persists the imputation fill-values so that inference reproduces the exact
training-time preprocessing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor

from .preprocess import (aggregate_dataframe, clean_dataframe, prepare_state,
                         train_test_split_xy)
from .schema import ProcessConfig, load_raw_dataframe, validate_dataframe


@dataclass
class Metrics:
    """Held-out evaluation metrics."""

    r2: float
    rmse: float
    mae: float
    n_test: int

    def __str__(self) -> str:
        return (f"R2={self.r2:.4f}  RMSE={self.rmse:.3f}  "
                f"MAE={self.mae:.3f}  (n_test={self.n_test})")


class QualityModel:
    """Wrapper around an XGBoost regressor bound to a :class:`ProcessConfig`.

    Holds the trained booster, the feature order, and the imputation values
    needed to preprocess inference requests identically to training.
    """

    def __init__(self, config: ProcessConfig,
                 model: XGBRegressor | None = None,
                 impute_values: dict[str, float] | None = None) -> None:
        self.config = config
        self.features = config.features
        self.model = model
        self.impute_values = impute_values or {}

    # ---- training -----------------------------------------------------------

    def train(self, df_clean: pd.DataFrame) -> Metrics:
        """Fit the model on a CLEAN DataFrame and return held-out metrics.

        Args:
            df_clean: Output of :func:`~src.preprocess.clean_dataframe`.

        Returns:
            :class:`Metrics` computed on the held-out test split.
        """
        X_train, X_test, y_train, y_test = train_test_split_xy(df_clean, self.config)
        params = dict(self.config.section("model").get("params", {}))
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("random_state",
                          int(self.config.section("dataset").get("random_seed", 42)))
        params.setdefault("n_jobs", -1)

        self.model = XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        return self._evaluate(X_test, y_test)

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Metrics:
        preds = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        return Metrics(
            r2=float(r2_score(y_test, preds)),
            rmse=rmse,
            mae=float(mean_absolute_error(y_test, preds)),
            n_test=len(y_test),
        )

    # ---- inference ----------------------------------------------------------

    def predict(self, state: dict[str, Any]) -> float:
        """Predict quality for a single process state.

        Args:
            state: Mapping of feature name -> value. May be partial; missing
                features are imputed using the training fill-values.

        Returns:
            Predicted target value as a float.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded.")
        X = prepare_state(state, self.config, impute_values=self.impute_values)
        return float(self.model.predict(X)[0])

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on a pre-built feature frame (columns in feature order)."""
        if self.model is None:
            raise RuntimeError("Model is not trained/loaded.")
        return self.model.predict(X[self.features])

    # ---- persistence --------------------------------------------------------

    def save(self) -> None:
        """Persist the booster and sidecar metadata to the configured paths."""
        if self.model is None:
            raise RuntimeError("Nothing to save: model is not trained.")
        artifact = self.config.artifact_path
        meta = self.config.metadata_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(artifact)
        meta_payload: dict[str, Any] = {
            "features": self.features,
            "target": self.config.target,
            "impute_values": self.impute_values,
        }
        with meta.open("w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh, indent=2)

    @classmethod
    def load(cls, config: ProcessConfig) -> "QualityModel":
        """Load a persisted model + metadata for a given config.

        Raises:
            FileNotFoundError: If the artifact or metadata is missing.
        """
        artifact = config.artifact_path
        meta = config.metadata_path
        if not artifact.exists() or not meta.exists():
            raise FileNotFoundError(
                f"Model artifacts not found ({artifact}, {meta}). "
                f"Train the model first (run_pipeline.py)."
            )
        booster = XGBRegressor()
        booster.load_model(artifact)
        with meta.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(config=config, model=booster,
                   impute_values=payload.get("impute_values", {}))


def load_model(config: ProcessConfig) -> QualityModel:
    """Convenience wrapper matching the requested `load_model` interface."""
    return QualityModel.load(config)


def train_from_csv(config: ProcessConfig,
                   csv_path: str | Path | None = None) -> tuple[QualityModel, Metrics]:
    """End-to-end training helper used by the pipeline and tests.

    Loads the CSV, validates it against the schema, cleans it, trains the model,
    and returns the trained model plus held-out metrics. Does NOT persist; the
    caller decides (so tests can train without touching artifacts/).

    Args:
        config: The process configuration.
        csv_path: Optional override of the dataset path.

    Returns:
        ``(model, metrics)``.
    """
    df = load_raw_dataframe(config, path=csv_path)
    # Validate presence/dtype before cleaning; skip range check (cleaning clips).
    validate_dataframe(df, config, require_target=True, check_ranges=False)
    df = aggregate_dataframe(df, config)
    df_clean, report = clean_dataframe(df, config)
    print(f"[preprocess] {report.summary()}")
    model = QualityModel(config=config, impute_values=report.impute_values)
    metrics = model.train(df_clean)
    return model, metrics
