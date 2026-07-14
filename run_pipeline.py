"""End-to-end pipeline: prove the whole system works in one command.

    generate data -> validate schema -> preprocess -> train -> evaluate
                  -> example optimization -> SHAP global + local drivers

Run:
    python run_pipeline.py

This is the smoke test an engineer runs after swapping in the real factory CSV:
if the data conforms to the config and the model trains, it prints metrics, one
concrete setpoint recommendation, and the drivers behind it.
"""

from __future__ import annotations

import pandas as pd

from src.explain import QualityExplainer
from src.fetch_real_data import main as fetch_real_data_main
from src.generate_data import main as generate_data_main
from src.model import QualityModel, train_from_csv
from src.optimizer import optimize_setpoints
from src.preprocess import clean_dataframe, split_xy
from src.schema import load_config, validate_dataframe


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    config = load_config()

    # 1. Obtain data. The `source` flag in the config decides whether we
    #    regenerate the physics-synthetic dataset or use/fetch the real one.
    source = config.section("dataset").get("source", "synthetic")
    if source == "synthetic":
        _rule("STEP 1/6  Generate physics-based synthetic data")
        generate_data_main()
    else:
        _rule("STEP 1/6  Obtain REAL dataset")
        if config.data_path.exists():
            print(f"[data] using existing real dataset at {config.data_path}")
        else:
            print("[data] real dataset missing -- fetching...")
            fetch_real_data_main()

    # 2. Load + validate against the config contract.
    _rule("STEP 2/6  Validate CSV against config contract")
    df = pd.read_csv(config.data_path)
    validate_dataframe(df, config, require_target=True, check_ranges=False)
    print(f"[schema] OK: {len(df.columns)} columns, {len(df):,} rows conform "
          f"to the contract.")
    print(f"[schema] controllable={config.controllable}")
    print(f"[schema] fixed={config.fixed}")
    print(f"[schema] target='{config.target}' (direction={config.target_direction})")

    # 3. Preprocess (clip outliers, impute) -- and 4. train + evaluate.
    _rule("STEP 3/6  Preprocess (clip outliers + impute)")
    df_clean, report = clean_dataframe(df, config)
    print(f"[preprocess] {report.summary()}")

    _rule("STEP 4/6  Train XGBoost + evaluate on held-out test set")
    model = QualityModel(config=config, impute_values=report.impute_values)
    metrics = model.train(df_clean)
    model.save()
    print(f"[model] {metrics}")
    print(f"[model] persisted to {config.artifact_path}")

    # 5. One concrete optimization example.
    _rule("STEP 5/6  Example optimization (recommend setpoints)")
    example_state = {
        "feed_rate": 120.0,
        "material_moisture": 6.5,
        "ambient_temp": 18.0,
        # current operator setpoints (deliberately sub-optimal):
        "gas_temperature": 1150.0,
        "oxygen_pct": 3.0,
        "flow_rate": 1400.0,
        "furnace_speed": 2.5,
    }
    result = optimize_setpoints(model, example_state, config)
    print("[optimize] fixed conditions:")
    for k, v in result.fixed_conditions.items():
        print(f"           {k:20s} = {v:8.2f} {config.columns[k].unit}")
    print("[optimize] current -> recommended setpoints:")
    for k in config.controllable:
        lo, hi = config.bounds(k)
        print(f"           {k:20s} {result.current_setpoints[k]:8.2f} -> "
              f"{result.recommended_setpoints[k]:8.2f} {config.columns[k].unit:5s} "
              f"[limits {lo:g}..{hi:g}]")
    print(f"[optimize] current quality   = {result.current_quality:6.2f}")
    print(f"[optimize] predicted quality = {result.predicted_quality:6.2f}")
    print(f"[optimize] improvement delta = {result.delta:+6.2f}")
    print(f"[optimize] within limits     = {result.within_limits}")

    # 6. SHAP: global importance + local drivers for the recommendation.
    _rule("STEP 6/6  SHAP explanations")
    explainer = QualityExplainer(model, config)
    X, _ = split_xy(df_clean, config)
    print("[shap] GLOBAL feature importance (mean |SHAP|):")
    for row in explainer.global_importance(X):
        print(f"       {row['feature']:20s} {row['importance_pct']:5.1f}%  "
              f"({row['role']})")

    recommended_state = {**result.fixed_conditions, **result.recommended_setpoints}
    print("\n[shap] LOCAL drivers for THIS recommendation "
          "(why this quality was predicted):")
    local = explainer.explain_state(recommended_state)
    print(f"       base value = {local['base_value']}, "
          f"predicted = {local['predicted_quality']}")
    for c in local["contributions"]:
        print(f"       {c['feature']:20s} value={c['value']:8.2f} "
              f"shap={c['shap']:+7.3f}  {c['direction']} quality")

    _rule("PIPELINE COMPLETE")
    print("All stages passed. Start the API with:  python -m src.api")


if __name__ == "__main__":
    main()
