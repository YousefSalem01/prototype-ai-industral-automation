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

from src.explain import QualityExplainer
from src.generate_data import main as generate_data_main
from src.model import QualityModel
from src.optimizer import optimize_setpoints
from src.preprocess import aggregate_dataframe, clean_dataframe, split_xy
from src.schema import load_config, load_raw_dataframe, validate_dataframe


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    config = load_config()

    # 1. Data source. For synthetic datasets we fabricate the CSV; for a real
    #    factory dataset (dataset.source: real) we use the dropped-in file as-is.
    if config.source == "real":
        _rule("STEP 1/6  Real dataset (source: real) -- using CSV as-is")
        print(f"[data] {config.data_path}")
    else:
        _rule("STEP 1/6  Generate physics-based synthetic data")
        generate_data_main()

    # 2. Load + validate against the config contract.
    _rule("STEP 2/6  Validate CSV against config contract")
    df = load_raw_dataframe(config)
    validate_dataframe(df, config, require_target=True, check_ranges=False)
    print(f"[schema] OK: {len(df.columns)} columns, {len(df):,} rows conform "
          f"to the contract.")
    print(f"[schema] controllable={config.controllable}")
    print(f"[schema] fixed={config.fixed}")
    print(f"[schema] target='{config.target}' (direction={config.target_direction})")

    # 3. Preprocess: optional aggregation (e.g. 20s sensors -> hourly), then
    #    clip outliers + impute.
    _rule("STEP 3/6  Preprocess (aggregate + clip outliers + impute)")
    df = aggregate_dataframe(df, config)
    agg_by = config.section("preprocess").get("aggregate_by")
    if agg_by:
        print(f"[preprocess] aggregated by '{agg_by}' -> {len(df):,} rows")
    df_clean, report = clean_dataframe(df, config)
    print(f"[preprocess] {report.summary()}")

    _rule("STEP 4/6  Train XGBoost + evaluate on held-out test set")
    model = QualityModel(config=config, impute_values=report.impute_values)
    metrics = model.train(df_clean)
    model.save()
    print(f"[model] {metrics}")
    print(f"[model] persisted to {config.artifact_path}")

    # 5. One concrete optimization example. Build the "current state" from a
    #    real row (a median-quality operating point) so the demo is dataset-
    #    agnostic: it works for the synthetic furnace AND the real plant.
    _rule("STEP 5/6  Example optimization (recommend setpoints)")
    ref_row = df_clean.sort_values(config.target).iloc[len(df_clean) // 2]
    example_state = {c: float(ref_row[c]) for c in config.features}
    result = optimize_setpoints(model, example_state, config)
    w = max(len(c) for c in config.columns)
    print("[optimize] fixed conditions:")
    for k, v in result.fixed_conditions.items():
        print(f"           {k:{w}s} = {v:10.2f} {config.columns[k].unit}")
    print("[optimize] current -> recommended setpoints:")
    for k in config.controllable:
        lo, hi = config.bounds(k)
        print(f"           {k:{w}s} {result.current_setpoints[k]:10.2f} -> "
              f"{result.recommended_setpoints[k]:10.2f} {config.columns[k].unit:6s} "
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
        print(f"       {row['feature']:{w}s} {row['importance_pct']:5.1f}%  "
              f"({row['role']})")

    recommended_state = {**result.fixed_conditions, **result.recommended_setpoints}
    print("\n[shap] LOCAL drivers for THIS recommendation "
          "(why this quality was predicted):")
    local = explainer.explain_state(recommended_state)
    print(f"       base value = {local['base_value']}, "
          f"predicted = {local['predicted_quality']}")
    for c in local["contributions"]:
        print(f"       {c['feature']:{w}s} value={c['value']:10.2f} "
              f"shap={c['shap']:+7.3f}  {c['direction']} quality")

    _rule("PIPELINE COMPLETE")
    print("All stages passed. Start the API with:  python -m src.api")


if __name__ == "__main__":
    main()
