"""Preprocessing: the SAME code path for synthetic and real factory data.

Responsibilities:
  * Coerce all declared columns to numeric.
  * Flag and winsorise (clip) physically-impossible values using config limits.
  * Impute missing values (median/mean per config).
  * Produce a reproducible train/test split.

Everything is driven by :class:`~src.schema.ProcessConfig`; no column names or
limits are hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import ProcessConfig


@dataclass
class PreprocessReport:
    """Diagnostics from a preprocessing run (for logging / audit trails)."""

    n_rows_in: int
    n_missing_imputed: dict[str, int] = field(default_factory=dict)
    n_outliers_clipped: dict[str, int] = field(default_factory=dict)
    impute_values: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        total_missing = sum(self.n_missing_imputed.values())
        total_outliers = sum(self.n_outliers_clipped.values())
        return (
            f"rows={self.n_rows_in:,} | imputed={total_missing} cells "
            f"| clipped={total_outliers} out-of-range values"
        )


def _clip_outliers(df: pd.DataFrame, config: ProcessConfig,
                   report: PreprocessReport) -> pd.DataFrame:
    """Winsorise values outside the tolerant operating band to the limits.

    We never DROP rows: a single faulty sensor should not throw away the other
    good readings in that record. Out-of-range values are pulled back to the
    nearest operating limit and counted in the report.
    """
    tol = float(config.section("preprocess").get("outlier_clip_tolerance", 0.0))
    for col in config.columns:
        if col not in df.columns:
            continue
        lo, hi = config.bounds(col)
        band = tol * (hi - lo)
        low_lim, high_lim = lo - band, hi + band
        col_vals = df[col]
        out_mask = ((col_vals < low_lim) | (col_vals > high_lim)) & col_vals.notna()
        n_out = int(out_mask.sum())
        if n_out:
            report.n_outliers_clipped[col] = n_out
            df[col] = col_vals.clip(lower=lo, upper=hi)
    return df


def _impute(df: pd.DataFrame, config: ProcessConfig,
            report: PreprocessReport,
            impute_values: dict[str, float] | None = None) -> pd.DataFrame:
    """Fill missing values per the configured strategy.

    Args:
        impute_values: If provided (e.g. from training), reuse these fill values
            instead of recomputing -- important so inference matches training.
    """
    strategy = config.section("preprocess").get("impute_strategy", "median")
    for col in config.columns:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if impute_values is not None and col in impute_values:
            fill = impute_values[col]
        elif strategy == "mean":
            fill = float(df[col].mean())
        else:
            fill = float(df[col].median())
        if n_missing:
            report.n_missing_imputed[col] = n_missing
        report.impute_values[col] = fill
        df[col] = df[col].fillna(fill)
    return df


def clean_dataframe(df: pd.DataFrame, config: ProcessConfig,
                    *, impute_values: dict[str, float] | None = None
                    ) -> tuple[pd.DataFrame, PreprocessReport]:
    """Run the full cleaning pipeline on a DataFrame.

    Order matters: coerce -> clip outliers -> impute. Outliers are clipped
    *before* imputation so that impossible values do not skew the median/mean.

    Args:
        df: Raw DataFrame (already schema-validated for column presence).
        config: The process configuration.
        impute_values: Optional pre-computed fill values to reuse at inference.

    Returns:
        A tuple ``(clean_df, report)``.
    """
    report = PreprocessReport(n_rows_in=len(df))
    df = df.copy()

    # Coerce declared columns to numeric (turns stray strings into NaN).
    for col in config.columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _clip_outliers(df, config, report)
    df = _impute(df, config, report, impute_values=impute_values)
    return df, report


def aggregate_dataframe(df: pd.DataFrame, config: ProcessConfig) -> pd.DataFrame:
    """Optionally resample rows by a grouping column, taking column means.

    Driven by ``preprocess.aggregate_by``. Needed when a fast sensor stream is
    paired with a slow lab measurement: e.g. this plant logs sensors every 20s
    but measures quality hourly, so ~180 rows share one target. Averaging each
    group to a single row yields genuinely independent samples (no repeated
    label leaking across a train/test split) and denoises the sensors.

    Args:
        df: Raw DataFrame (already schema-validated).
        config: The process configuration.

    Returns:
        The aggregated DataFrame, or ``df`` unchanged if no ``aggregate_by`` set.
    """
    agg_col = config.section("preprocess").get("aggregate_by")
    if not agg_col or agg_col not in df.columns:
        return df
    declared = [c for c in config.columns if c in df.columns]
    return df.groupby(agg_col, sort=True, as_index=False)[declared].mean()


def split_xy(df: pd.DataFrame, config: ProcessConfig
             ) -> tuple[pd.DataFrame, pd.Series]:
    """Split a clean DataFrame into feature matrix X and target vector y."""
    X = df[config.features].copy()
    y = df[config.target].copy()
    return X, y


def train_test_split_xy(df: pd.DataFrame, config: ProcessConfig
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Produce a reproducible train/test split of features and target.

    Two strategies, chosen by ``preprocess.split``:
      * ``random`` (default): shuffled split -- fine for i.i.d. rows.
      * ``time``: sort by ``preprocess.time_column`` and hold out the most
        RECENT fraction as the test set. Use this when consecutive rows share a
        label (e.g. hourly lab value repeated across 20s sensor rows): a random
        split would place the same period in both train and test and inflate R².
        A time split also mirrors production -- train on the past, score the
        future.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    pp = config.section("preprocess")
    test_size = float(pp.get("test_size", 0.2))
    seed = int(config.section("dataset").get("random_seed", 42))
    strategy = pp.get("split", "random")
    time_col = config.time_column

    if strategy == "time" and time_col and time_col in df.columns:
        ordered = df.sort_values(time_col, kind="stable")
        X, y = split_xy(ordered, config)
        n_test = int(round(len(ordered) * test_size))
        split_at = len(ordered) - n_test
        return (X.iloc[:split_at], X.iloc[split_at:],
                y.iloc[:split_at], y.iloc[split_at:])

    X, y = split_xy(df, config)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def prepare_state(state: dict[str, Any], config: ProcessConfig,
                  impute_values: dict[str, float] | None = None) -> pd.DataFrame:
    """Turn a single-state dict (API payload) into a one-row feature frame.

    Applies the identical clip+impute path so a live request is treated exactly
    like a training row. Missing features are imputed from ``impute_values``.

    Args:
        state: Mapping of feature name -> value (may be partial).
        config: The process configuration.
        impute_values: Fill values learned at training time.

    Returns:
        A single-row DataFrame with columns in ``config.features`` order.
    """
    row = {c: state.get(c, np.nan) for c in config.features}
    df = pd.DataFrame([row])
    clean, _ = clean_dataframe(df, config, impute_values=impute_values)
    return clean[config.features]
