# Industrial AI MVP — Process Quality Optimizer

An AI layer for a furnace/kiln process (cement domain). It learns how quality
responds to the process state, **recommends optimal controllable setpoints** for
the current operating conditions, and **explains why** — so engineers stop
tuning gas temperature, oxygen, flow, and kiln speed by hand.

This is an MVP built to be *swapped onto real factory data* with a config edit
and nothing else.

---

## The core design principle: the CSV is a contract

Every column name, its role (controllable vs fixed condition vs target), its
unit, and its operating limits live in **`config/process_config.yaml`**. That
file is the single source of truth. **No column name is hard-coded anywhere in
`src/`.** Roles are resolved through helpers on the config object
(`config.controllable`, `config.fixed`, `config.target`, `config.bounds(...)`).

Consequence: swapping the synthetic CSV for the real factory CSV requires
editing only that YAML and dropping in the new file.

---

## Quick start

```bash
pip install -r requirements.txt      # Python 3.11+ (tested on 3.13)
python run_pipeline.py               # generate → validate → train → optimize → explain
python -m src.api                    # serve the model at http://127.0.0.1:8000
pytest -q                            # run the test suite
```

`run_pipeline.py` is the one-command proof the whole system works. On the
synthetic data it reports a held-out **R² ≈ 0.86**, prints one concrete setpoint
recommendation with its improvement delta, and lists the SHAP drivers behind it.

---

## Architecture

```
config/process_config.yaml   # THE CONTRACT: columns, roles, units, limits
src/
  schema.py         # loads + validates config and any CSV against it
  generate_data.py  # physics-based synthetic kiln simulator (see below)
  preprocess.py     # impute + clip outliers; ONE code path for synth & real
  model.py          # XGBoost surrogate: train / evaluate / save / load / predict
  optimizer.py      # Optuna search over controllables within operating limits
  explain.py        # SHAP global importance + local per-recommendation drivers
  api.py            # FastAPI: /health /predict /optimize /explain
run_pipeline.py     # end-to-end orchestration
tests/              # schema rejection, end-to-end, optimizer-respects-limits
artifacts/          # persisted model.json + model_meta.json
data/               # process_data.csv
```

Data flow: **schema** validates the CSV against the contract → **preprocess**
cleans it (identically for synthetic and real) → **model** trains an XGBoost
surrogate of quality → **optimizer** searches only the controllable setpoints,
within their config limits, holding the fixed conditions constant → **explain**
attributes the result to individual variables → **api** exposes all of it as
JSON over local HTTP.

### The synthetic generator is physics-based, not linear

`generate_data.py` does **not** use `noise + linear formula`. Quality is the
product of four physical response factors — thermal adequacy, combustion
completeness, residence time, aeration — each with a **condition-dependent
optimum**, **asymmetric penalties** (too cold hurts more than too hot), and a
**genuine interaction** (ideal excess-O₂ depends on flame temperature). The
sampler simulates a realistic plant historian: mostly reasonable operation near
the optima with operator scatter, plus a ~20% minority of off-spec excursions so
the model sees the full response surface. It also injects ~1% missing values and
~0.5% physically-impossible sensor faults so preprocessing has real work to do.
Full assumptions are documented at the top of that file.

---

## Swapping in the REAL factory CSV

No code changes. Only `config/process_config.yaml` and the data file.

1. **Drop in the data.** Put the real export at `data/process_data.csv`, or
   point `dataset.path` at wherever it lives.

2. **Edit the column contract.** Under `columns:`, make one entry per real CSV
   header (the key must match the header *exactly*). For each, set:
   - `role`: `controllable` (a setpoint engineers can dial), `fixed` (a measured
     condition they can't set), or `target` (the quality KPI). Exactly one
     `target` is required.
   - `unit`: engineering unit (documentation + API output).
   - `min` / `max`: the physical/operating limits. These drive **both** outlier
     clipping **and** the optimizer's search bounds — set them to the real safe
     operating envelope.
   - On the target only: `direction: maximize` or `minimize`.

3. **(Optional) tune settings.** `preprocess.impute_strategy`,
   `preprocess.outlier_clip_tolerance`, `model.params`, `optimizer.n_trials`.

4. **Validate + retrain.** Run `python run_pipeline.py`. `schema.py` checks
   column presence, numeric dtype, and ranges, and **fails loudly** naming the
   offending column if the real data doesn't conform to the contract. Fix the
   YAML (or the export) until it passes; the model retrains on the real data and
   the API serves it.

Because roles are read from the config, adding/removing a controllable or fixed
variable, or renaming any column, is a YAML edit — the optimizer, SHAP, and API
surface all follow automatically.

---

## Desktop application integration model

The AI layer runs as a **local HTTP service** (`python -m src.api`, default
`127.0.0.1:8000`). The existing engineering desktop app integrates by POSTing
JSON — **language-agnostic**, so it works whether the desktop app is C#, C++,
Java, or anything else. No shared runtime, no Python embedding.

Typical loop inside the desktop app:

1. Read the operator's current fixed conditions (and current setpoints).
2. `POST /optimize` with that state → receive recommended setpoints, predicted
   quality, and the improvement delta vs. current.
3. `POST /explain` → receive the SHAP drivers to show *why*.
4. Display the recommendation; the engineer accepts or overrides.

### Endpoints

| Method | Path        | Purpose                                              |
|--------|-------------|------------------------------------------------------|
| GET    | `/health`   | Liveness + the controllable/fixed/target contract    |
| POST   | `/predict`  | Predicted quality for a supplied state               |
| POST   | `/optimize` | Recommended setpoints for the current fixed conditions |
| POST   | `/explain`  | SHAP drivers for a supplied state                    |

### Example call (from the desktop app)

```bash
curl -s -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"state": {
        "feed_rate": 120.0, "material_moisture": 6.5, "ambient_temp": 18.0,
        "gas_temperature": 1150.0, "oxygen_pct": 3.0,
        "flow_rate": 1400.0, "furnace_speed": 2.5
      }}'
```

```jsonc
{
  "recommended_setpoints": {"gas_temperature": 1289.2, "oxygen_pct": 4.03,
                            "flow_rate": 1692.0, "furnace_speed": 3.51},
  "current_quality": 59.23,
  "predicted_quality": 82.29,
  "delta": 23.06,
  "within_limits": true
}
```

`/optimize` requires all **fixed** conditions in the state (it returns HTTP 422
naming any missing ones); current controllable setpoints are optional and used
only to compute the improvement delta. Recommended setpoints are guaranteed to
lie within the config operating limits.

A C# sketch is included in the header comment of [`src/api.py`](src/api.py).

---

## Notes for production hardening (out of MVP scope)

- Persist a preprocessing/scaler artifact alongside the model (already done for
  imputation fill-values in `model_meta.json`).
- Add model-version + training-data hash to `/health` for auditability.
- Constrain the optimizer with rate-of-change / safety interlocks before any
  closed-loop use; today it recommends, the engineer decides.
- Monitor live prediction error vs. realized quality to trigger retraining.
