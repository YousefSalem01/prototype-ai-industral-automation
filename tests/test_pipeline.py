"""Test suite for the industrial AI MVP.

Covers the three required guarantees:
  1. schema rejects a malformed CSV;
  2. the pipeline runs end-to-end on a small sample;
  3. the optimizer respects operating limits.

Plus a preprocessing test proving outliers are clipped and NaNs imputed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_dataset
from src.model import QualityModel, load_model, train_from_csv
from src.optimizer import optimize_setpoints
from src.preprocess import clean_dataframe
from src.schema import SchemaError, load_config, validate_dataframe


@pytest.fixture(scope="module")
def config():
    """Load the real project config once per module."""
    return load_config()


@pytest.fixture(scope="module")
def small_config(config, tmp_path_factory):
    """A copy of the config that generates only a small dataset, for speed."""
    # Reuse the same ProcessConfig but shrink the synthetic row count.
    config.raw["dataset"]["n_synthetic_rows"] = 800
    config.raw["optimizer"]["n_trials"] = 40
    return config


# --- 1. schema rejects malformed CSV -----------------------------------------
def test_schema_rejects_missing_column(config):
    """A CSV missing a declared column must raise SchemaError."""
    df = pd.DataFrame({config.controllable[0]: [1.0, 2.0]})  # only one column
    with pytest.raises(SchemaError, match="missing required column"):
        validate_dataframe(df, config)


def test_schema_rejects_non_numeric(config):
    """A declared column full of non-numeric junk must raise SchemaError."""
    row = {c: [1.0] for c in config.columns}
    row[config.fixed[0]] = ["not_a_number"]
    df = pd.DataFrame(row)
    with pytest.raises(SchemaError, match="non-numeric"):
        validate_dataframe(df, config, check_ranges=False)


def test_schema_rejects_out_of_range(config):
    """Values grossly outside operating limits fail the range check."""
    row = {c: [0.5 * (config.bounds(c)[0] + config.bounds(c)[1])]
           for c in config.columns}
    bad = config.controllable[0]
    row[bad] = [config.bounds(bad)[1] * 1000]  # absurdly high
    df = pd.DataFrame(row)
    with pytest.raises(SchemaError, match="Out-of-range"):
        validate_dataframe(df, config, check_ranges=True)


# --- preprocessing does real work --------------------------------------------
def test_preprocess_clips_and_imputes(config):
    """Impossible outliers are clipped to limits and NaNs are imputed."""
    df = generate_dataset(config).head(500)
    # Force a known impossible value and a known missing value.
    bad_col = config.controllable[0]
    df.iloc[0, df.columns.get_loc(bad_col)] = 1e6
    df.iloc[1, df.columns.get_loc(bad_col)] = np.nan

    clean, report = clean_dataframe(df, config)

    lo, hi = config.bounds(bad_col)
    assert clean[bad_col].max() <= hi + 1e-6
    assert clean[bad_col].min() >= lo - 1e-6
    assert not clean[config.features].isna().any().any()
    assert report.n_outliers_clipped.get(bad_col, 0) >= 1


# --- 2. end-to-end pipeline on a small sample --------------------------------
def test_pipeline_end_to_end(small_config, tmp_path):
    """Generate -> validate -> clean -> train -> predict on a small sample."""
    df = generate_dataset(small_config)
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)

    validate_dataframe(pd.read_csv(csv), small_config, check_ranges=False)
    model, metrics = train_from_csv(small_config, csv_path=csv)

    # Model learned *something* on this nonlinear signal.
    assert metrics.n_test > 0
    assert metrics.r2 > 0.3
    assert metrics.rmse > 0

    # predict() accepts a partial state and returns a bounded score.
    state = {c: 0.5 * (small_config.bounds(c)[0] + small_config.bounds(c)[1])
             for c in small_config.features}
    pred = model.predict(state)
    lo, hi = small_config.bounds(small_config.target)
    assert lo - 5 <= pred <= hi + 5


def test_persistence_roundtrip(small_config, tmp_path, monkeypatch):
    """A saved model loads back and predicts identically."""
    df = generate_dataset(small_config)
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    model, _ = train_from_csv(small_config, csv_path=csv)

    # Redirect artifact paths into tmp so we don't clobber real artifacts.
    monkeypatch.setitem(small_config.raw["model"], "artifact_path",
                        str(tmp_path / "m.json"))
    monkeypatch.setitem(small_config.raw["model"], "metadata_path",
                        str(tmp_path / "m_meta.json"))
    model.save()
    reloaded = load_model(small_config)

    state = {c: 0.5 * (small_config.bounds(c)[0] + small_config.bounds(c)[1])
             for c in small_config.features}
    assert abs(model.predict(state) - reloaded.predict(state)) < 1e-6


# --- 3. optimizer respects operating limits ----------------------------------
def test_optimizer_respects_limits(small_config, tmp_path):
    """Every recommended setpoint must lie within its config operating limits."""
    df = generate_dataset(small_config)
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    model, _ = train_from_csv(small_config, csv_path=csv)

    state = {
        "feed_rate": 130.0, "material_moisture": 8.0, "ambient_temp": 15.0,
        "gas_temperature": 1100.0, "oxygen_pct": 3.0,
        "flow_rate": 1300.0, "furnace_speed": 2.0,
    }
    result = optimize_setpoints(model, state, small_config)

    assert result.within_limits
    for c in small_config.controllable:
        lo, hi = small_config.bounds(c)
        val = result.recommended_setpoints[c]
        assert lo <= val <= hi, f"{c}={val} outside [{lo}, {hi}]"


def test_optimizer_requires_fixed_conditions(small_config, tmp_path):
    """Optimizing without the fixed conditions raises a clear error."""
    df = generate_dataset(small_config)
    csv = tmp_path / "sample.csv"
    df.to_csv(csv, index=False)
    model, _ = train_from_csv(small_config, csv_path=csv)

    # Missing all fixed conditions.
    with pytest.raises(ValueError, match="fixed conditions not provided"):
        optimize_setpoints(model, {"gas_temperature": 1200.0}, small_config)
