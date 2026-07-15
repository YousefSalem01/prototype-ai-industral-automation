"""Schema and configuration layer.

This module is the ONLY place that reads ``process_config.yaml``. Every other
module obtains column names, roles, units, and operating limits through the
:class:`ProcessConfig` object returned here -- never by hard-coding a string.

If the real factory CSV does not conform to the contract in the YAML, the
:func:`validate_dataframe` function fails loudly with an actionable message
naming the offending column(s), so an engineer can fix the YAML or the export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Valid roles a column may declare in the config.
_VALID_ROLES = {"controllable", "fixed", "target"}


@dataclass(frozen=True)
class ColumnSpec:
    """Declared contract for a single CSV column.

    Attributes:
        name: Column header, matching the CSV exactly.
        role: One of ``controllable``, ``fixed`` or ``target``.
        unit: Human-readable engineering unit (e.g. ``"degC"``).
        min: Lower physical/operating limit.
        max: Upper physical/operating limit.
        description: Free-text description for humans and API docs.
        direction: For the target only: ``maximize`` or ``minimize``.
    """

    name: str
    role: str
    unit: str
    min: float
    max: float
    description: str = ""
    direction: str | None = None


class SchemaError(ValueError):
    """Raised when a CSV does not conform to the config contract."""


class ProcessConfig:
    """Parsed, validated view over ``process_config.yaml``.

    This object is the single source of truth for the rest of the pipeline.
    """

    def __init__(self, raw: dict[str, Any], config_dir: Path) -> None:
        self._raw = raw
        self._config_dir = config_dir
        self.columns: dict[str, ColumnSpec] = self._parse_columns(raw)
        self._validate_config()

    # ---- construction -------------------------------------------------------

    @staticmethod
    def _parse_columns(raw: dict[str, Any]) -> dict[str, ColumnSpec]:
        cols: dict[str, ColumnSpec] = {}
        for name, spec in raw.get("columns", {}).items():
            cols[name] = ColumnSpec(
                name=name,
                role=spec["role"],
                unit=spec.get("unit", ""),
                min=float(spec["min"]),
                max=float(spec["max"]),
                description=spec.get("description", ""),
                direction=spec.get("direction"),
            )
        return cols

    def _validate_config(self) -> None:
        """Sanity-check the config itself (independent of any CSV)."""
        if not self.columns:
            raise SchemaError("Config declares no columns under 'columns:'.")

        for spec in self.columns.values():
            if spec.role not in _VALID_ROLES:
                raise SchemaError(
                    f"Column '{spec.name}' has invalid role '{spec.role}'. "
                    f"Expected one of {sorted(_VALID_ROLES)}."
                )
            if spec.min >= spec.max:
                raise SchemaError(
                    f"Column '{spec.name}' has min ({spec.min}) >= max "
                    f"({spec.max}); operating limits must be a valid range."
                )

        targets = [s for s in self.columns.values() if s.role == "target"]
        if len(targets) != 1:
            raise SchemaError(
                f"Config must declare exactly one 'target' column, "
                f"found {len(targets)}: {[t.name for t in targets]}."
            )
        if targets[0].direction not in {"maximize", "minimize"}:
            raise SchemaError(
                f"Target '{targets[0].name}' must set direction to "
                f"'maximize' or 'minimize', got '{targets[0].direction}'."
            )
        if not self.controllable:
            raise SchemaError("Config declares no 'controllable' columns.")

    # ---- role helpers (used everywhere instead of literal names) -----------

    @property
    def controllable(self) -> list[str]:
        """Names of the controllable setpoint columns (optimizer variables)."""
        return [n for n, s in self.columns.items() if s.role == "controllable"]

    @property
    def fixed(self) -> list[str]:
        """Names of the fixed operating-condition columns."""
        return [n for n, s in self.columns.items() if s.role == "fixed"]

    @property
    def target(self) -> str:
        """Name of the single target column."""
        return next(n for n, s in self.columns.items() if s.role == "target")

    @property
    def features(self) -> list[str]:
        """Model input features = controllable + fixed, in config order."""
        return [n for n, s in self.columns.items() if s.role != "target"]

    @property
    def target_direction(self) -> str:
        """``maximize`` or ``minimize`` for the target."""
        return self.columns[self.target].direction  # type: ignore[return-value]

    def bounds(self, name: str) -> tuple[float, float]:
        """Return ``(min, max)`` operating limits for a column."""
        spec = self.columns[name]
        return spec.min, spec.max

    # ---- paths --------------------------------------------------------------

    @property
    def data_path(self) -> Path:
        """Absolute path to the dataset CSV (resolved relative to project root)."""
        return self._project_root / self._raw["dataset"]["path"]

    @property
    def artifact_path(self) -> Path:
        """Absolute path to the persisted model file."""
        return self._project_root / self._raw["model"]["artifact_path"]

    @property
    def metadata_path(self) -> Path:
        """Absolute path to the persisted model metadata file."""
        return self._project_root / self._raw["model"]["metadata_path"]

    @property
    def _project_root(self) -> Path:
        # config lives in <root>/config, so root is its parent.
        return self._config_dir.parent

    # ---- generic access -----------------------------------------------------

    @property
    def raw(self) -> dict[str, Any]:
        """The raw parsed YAML (for section-specific settings)."""
        return self._raw

    def section(self, key: str) -> dict[str, Any]:
        """Return a top-level config section (e.g. ``model``, ``optimizer``)."""
        return self._raw.get(key, {})

    @property
    def source(self) -> str:
        """Dataset source: ``real`` (use CSV as-is) or ``synthetic`` (generate)."""
        return self.section("dataset").get("source", "synthetic")

    @property
    def csv_read_kwargs(self) -> dict[str, Any]:
        """Extra pandas.read_csv kwargs for this dataset (decimal, sep, ...)."""
        return dict(self.section("dataset").get("csv_read_kwargs", {}))

    @property
    def time_column(self) -> str | None:
        """Optional timestamp column used for a time-ordered train/test split."""
        return self.section("preprocess").get("time_column")


def load_raw_dataframe(config: ProcessConfig,
                       path: str | Path | None = None) -> pd.DataFrame:
    """Read the dataset CSV honouring config-declared read options.

    Real factory exports vary (European decimal commas, alternate delimiters,
    encodings). Those quirks are declared in ``dataset.csv_read_kwargs`` so the
    swap stays config-only -- no code edits per dataset.

    Args:
        config: The process configuration.
        path: Optional override of the dataset path.

    Returns:
        The raw DataFrame (all columns, unmodified).
    """
    csv_path = Path(path) if path else config.data_path
    return pd.read_csv(csv_path, **config.csv_read_kwargs)


def _default_config_path() -> Path:
    """Locate ``config/process_config.yaml`` relative to this file."""
    return Path(__file__).resolve().parents[1] / "config" / "process_config.yaml"


def load_config(path: str | Path | None = None) -> ProcessConfig:
    """Load and validate the process configuration.

    Args:
        path: Optional explicit path to the YAML. Defaults to
            ``<project_root>/config/process_config.yaml``.

    Returns:
        A validated :class:`ProcessConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        SchemaError: If the config is internally inconsistent.
    """
    cfg_path = Path(path) if path else _default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ProcessConfig(raw=raw, config_dir=cfg_path.parent)


def validate_dataframe(df: pd.DataFrame, config: ProcessConfig,
                       *, require_target: bool = True,
                       check_ranges: bool = True) -> None:
    """Validate a DataFrame against the config contract.

    Checks, in order:
      1. Every declared column is present in the DataFrame.
      2. Every declared column is numeric (coercible to float).
      3. (optional) Non-missing values fall within a tolerant operating band.

    Args:
        df: The DataFrame loaded from a CSV.
        config: The process configuration.
        require_target: If ``False``, the target column may be absent
            (used for inference-time payloads that only carry inputs).
        check_ranges: If ``False``, skip the value-range check (used before
            preprocessing, which is what clips out-of-range values).

    Raises:
        SchemaError: With a clear message naming the first problem found.
    """
    expected = list(config.columns)
    if not require_target:
        expected = [c for c in expected if c != config.target]

    # 1. Column presence.
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(
            f"CSV is missing required column(s): {missing}. "
            f"Present columns: {list(df.columns)}. "
            f"Fix the CSV export or the 'columns:' keys in the config."
        )

    # 2. Dtype / numeric coercion.
    non_numeric: list[str] = []
    for col in expected:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # A column is non-numeric if coercion produced NEW NaNs on non-null cells.
        newly_nan = coerced.isna() & df[col].notna()
        if newly_nan.any():
            non_numeric.append(col)
    if non_numeric:
        raise SchemaError(
            f"Column(s) contain non-numeric values that are not missing "
            f"markers: {non_numeric}. All declared columns must be numeric."
        )

    # 3. Range check (tolerant; preprocessing is what actually clips).
    if check_ranges:
        tol = float(config.section("preprocess").get("outlier_clip_tolerance", 0.0))
        problems: list[str] = []
        for col in expected:
            lo, hi = config.bounds(col)
            band = tol * (hi - lo)
            vals = pd.to_numeric(df[col], errors="coerce")
            out = vals[(vals < lo - band) | (vals > hi + band)]
            if len(out) > 0:
                problems.append(
                    f"'{col}': {len(out)} value(s) outside "
                    f"[{lo - band}, {hi + band}] (e.g. {out.iloc[0]:.3g})"
                )
        if problems:
            raise SchemaError(
                "Out-of-range values detected:\n  " + "\n  ".join(problems)
            )
