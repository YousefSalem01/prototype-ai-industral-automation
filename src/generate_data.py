"""Physics-based synthetic data generator for a rotary kiln / furnace.

WHY NOT random-noise-plus-linear-formula?
    A linear model would let the ML step "cheat": a linear regressor would nail
    it and the optimizer would push every setpoint to a corner of its range.
    Real kilns don't behave that way. Quality is governed by how close each
    controllable setpoint sits to a *condition-dependent optimum*, with genuine
    coupling between variables and asymmetric penalties on either side. That is
    what makes an ML surrogate + search worthwhile, and it is what this
    generator reproduces.

PHYSICS ASSUMPTIONS (documented, deliberately simplified but qualitatively real)
--------------------------------------------------------------------------------
The kiln converts raw feed into clinker. Quality is modelled as the product of
four factors, each in (0, 1], multiplied and scaled to 0-100:

  quality = 100 * thermal * combustion * residence * aeration - defect_penalty

1. THERMAL ADEQUACY  (gas_temperature vs a moving target)
   The material needs a minimum flame temperature to fully calcine. That
   required temperature RISES with:
     - feed_rate        (more mass to heat per unit time)
     - material_moisture (latent heat must first boil off the water)
   and FALLS slightly with ambient_temp (less heat lost to surroundings).
   Deviation is penalised ASYMMETRICALLY: too cold -> under-burnt / raw meal
   (steep penalty); too hot -> wasted fuel, ring formation, thermal defects
   (milder but real). => condition-dependent optimum + diminishing returns.

2. COMBUSTION COMPLETENESS  (oxygen x gas_temperature INTERACTION)
   Complete combustion needs enough excess O2 *for the amount of fuel burned*.
   Hotter flames burn more fuel, so the ideal excess-O2 RISES with
   gas_temperature and with air flow_rate. Too little O2 -> incomplete
   combustion, CO, soot (steep). Too much O2 -> excess cold air quenches the
   flame, energy loss (milder). The optimum O2 depends on gas_temperature =>
   a genuine two-variable interaction, not two independent effects.

3. RESIDENCE TIME  (furnace_speed vs feed_rate)
   Rotation speed sets how long material stays in the hot zone. Faster rotation
   -> shorter residence. The ideal speed RISES with feed_rate (more throughput
   needs faster transport to avoid choking) but too fast -> insufficient
   residence -> under-processed; too slow -> over-burnt / dead-burnt.

4. AERATION / MIXING  (flow_rate vs feed_rate)
   Air flow must match the fuel+material load to keep the bed fluidised and
   evenly heated. Ideal flow RISES with feed_rate. Too low -> hot/cold spots;
   too high -> turbulent heat loss and dust.

5. DEFECT PENALTY  (nonlinear coupling)
   Simultaneously running very hot AND very oxygen-rich promotes NOx and
   clinker defects -- an extra multiplicative penalty that only bites in that
   corner, adding realistic higher-order structure.

DATA-QUALITY ARTIFACTS (so preprocessing has real work to do)
   - Gaussian sensor noise on the quality reading (~1.5 points).
   - ~1% missing values scattered across input sensors (NaN).
   - ~0.5% physically-impossible outlier readings (stuck/faulty sensors:
     negative flow, 5000 degC gas temp, negative oxygen, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ProcessConfig, load_config


# --- tunable physics constants (kept here, not in the CSV contract) ----------
# These are properties of the *simulated plant*, not of the data schema, so
# they intentionally live in code rather than in process_config.yaml.
_MISSING_FRACTION = 0.01     # ~1% of input cells set to NaN
_OUTLIER_FRACTION = 0.005    # ~0.5% of rows get a faulty sensor reading
_NOISE_STD = 1.5             # sensor noise on quality (points)


def _ideal_gas_temperature(feed: np.ndarray, moisture: np.ndarray,
                           ambient: np.ndarray) -> np.ndarray:
    """Condition-dependent optimum flame temperature (degC).

    Rises with feed rate (more mass to calcine) and moisture (latent heat to
    boil off water); eases slightly with a warmer ambient (less shell loss).
    """
    return (1050.0
            + 1.6 * (feed - 50.0)
            + 14.0 * moisture
            - 1.2 * (ambient - 20.0))


def _ideal_oxygen(gas: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Condition-dependent optimum excess O2 (%).

    Couples to flame temperature and air flow: hotter flames burn more fuel and
    need more excess air. This coupling is the genuine O2 x gas_temp interaction.
    """
    return (3.0
            + 0.004 * (gas - 1050.0)
            + 0.0007 * (flow - 1750.0))


def _ideal_speed(feed: np.ndarray) -> np.ndarray:
    """Condition-dependent optimum kiln speed (rpm): rises with throughput."""
    return 1.8 + 0.02 * (feed - 50.0)


def _ideal_flow(feed: np.ndarray) -> np.ndarray:
    """Condition-dependent optimum air flow (m3/h): rises with throughput."""
    return 900.0 + 9.0 * (feed - 50.0)


def _asymmetric_gaussian(x: np.ndarray, ideal: np.ndarray | float,
                         sigma_low: float, sigma_high: float) -> np.ndarray:
    """Bell-shaped response, 1.0 at ``ideal``, decaying on both sides.

    Using a different sigma below vs above the optimum encodes the physical
    asymmetry (e.g. under-burning hurts more than mild over-burning).

    Args:
        x: Observed values.
        ideal: The (possibly per-row) optimum.
        sigma_low: Std-dev governing decay when ``x < ideal``.
        sigma_high: Std-dev governing decay when ``x >= ideal``.

    Returns:
        Array of response factors in (0, 1].
    """
    ideal_arr = np.broadcast_to(ideal, x.shape)
    sigma = np.where(x < ideal_arr, sigma_low, sigma_high)
    return np.exp(-0.5 * ((x - ideal_arr) / sigma) ** 2)


# Fraction of rows that represent deliberate off-spec excursions (trials,
# upsets, commissioning). These give the model coverage of the poor-quality
# region so it learns the full response surface, not just the sweet spot.
_EXCURSION_FRACTION = 0.20


def _sample_conditions(rng: np.random.Generator, n: int,
                       config: ProcessConfig) -> dict[str, np.ndarray]:
    """Sample a realistic operating history.

    Real plant historians are dominated by *reasonable* operation: operators
    hold each setpoint near its (condition-dependent) optimum, with scatter from
    manual control, drift, and lag. We reproduce that by sampling controllables
    around their ideal points given the fixed conditions, then overlaying a
    minority of wide "excursion" rows so the poor-quality region is also
    represented. This keeps the quality distribution realistic (centred high,
    left-tailed) AND gives the surrogate a densely-populated response surface.
    """
    data: dict[str, np.ndarray] = {}

    # Fixed conditions: sample across their full declared range (uncontrolled).
    for name in config.fixed:
        lo, hi = config.bounds(name)
        data[name] = rng.uniform(lo, hi, n)

    feed = data["feed_rate"]
    moisture = data["material_moisture"]
    ambient = data["ambient_temp"]

    # Per-row operator scatter multiplier: tight for normal ops, wide for the
    # excursion fraction.
    is_excursion = rng.random(n) < _EXCURSION_FRACTION
    scatter = np.where(is_excursion, 3.0, 1.0)

    def around_ideal(name: str, ideal: np.ndarray, base_sigma: float) -> np.ndarray:
        lo, hi = config.bounds(name)
        vals = ideal + rng.normal(0.0, base_sigma, n) * scatter
        return np.clip(vals, lo, hi)

    # Sample in dependency order so interactions are reflected in the history:
    # gas & flow first, then O2 (whose optimum depends on gas & flow).
    gas = around_ideal("gas_temperature",
                        _ideal_gas_temperature(feed, moisture, ambient), 55.0)
    flow = around_ideal("flow_rate", _ideal_flow(feed), 220.0)
    o2 = around_ideal("oxygen_pct", _ideal_oxygen(gas, flow), 0.9)
    speed = around_ideal("furnace_speed", _ideal_speed(feed), 0.55)

    data["gas_temperature"] = gas
    data["flow_rate"] = flow
    data["oxygen_pct"] = o2
    data["furnace_speed"] = speed
    return data


def _quality_physics(data: dict[str, np.ndarray],
                     config: ProcessConfig) -> np.ndarray:
    """Compute the noise-free quality score from the physical model.

    See the module docstring for the assumptions behind each factor.
    """
    # Pull inputs by role-derived names so this stays generic to the config's
    # *values*, while the physics itself is intrinsically about these variables.
    gas = data["gas_temperature"]
    o2 = data["oxygen_pct"]
    flow = data["flow_rate"]
    speed = data["furnace_speed"]
    feed = data["feed_rate"]
    moisture = data["material_moisture"]
    ambient = data["ambient_temp"]

    # ---- 1. Thermal adequacy: gas_temperature vs moving target -------------
    # Optimum rises with feed and moisture, eases with a warmer ambient.
    # Penalty is asymmetric: too cold (under-burnt) hurts more than too hot.
    ideal_gas = _ideal_gas_temperature(feed, moisture, ambient)
    thermal = _asymmetric_gaussian(gas, ideal_gas, sigma_low=90.0, sigma_high=150.0)

    # ---- 2. Combustion completeness: oxygen x gas_temperature --------------
    # Ideal excess O2 rises with flame temperature and air flow (more fuel
    # burned). This coupling to `gas` is the genuine interaction effect.
    ideal_o2 = _ideal_oxygen(gas, flow)
    combustion = _asymmetric_gaussian(o2, ideal_o2, sigma_low=1.1, sigma_high=2.2)

    # ---- 3. Residence time: furnace_speed vs feed_rate ---------------------
    residence = _asymmetric_gaussian(speed, _ideal_speed(feed),
                                     sigma_low=0.9, sigma_high=1.1)

    # ---- 4. Aeration / mixing: flow_rate vs feed_rate ----------------------
    aeration = _asymmetric_gaussian(flow, _ideal_flow(feed),
                                    sigma_low=350.0, sigma_high=450.0)

    # ---- Combine multiplicatively (interactions + diminishing returns) -----
    quality = 100.0 * thermal * combustion * residence * aeration

    # ---- 5. High-order defect penalty: hot AND oxygen-rich corner ----------
    # Smooth logistic gates; only bites when both are simultaneously high.
    hot_gate = 1.0 / (1.0 + np.exp(-(gas - 1400.0) / 25.0))
    o2_gate = 1.0 / (1.0 + np.exp(-(o2 - 8.0) / 0.7))
    quality -= 18.0 * hot_gate * o2_gate

    return np.clip(quality, 0.0, 100.0)


def _inject_missing(df: pd.DataFrame, config: ProcessConfig,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Scatter ~1% NaNs across INPUT sensor columns (never the target)."""
    input_cols = config.features
    for col in input_cols:
        mask = rng.random(len(df)) < _MISSING_FRACTION
        df.loc[mask, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, config: ProcessConfig,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Corrupt ~0.5% of rows with physically-impossible sensor readings.

    Faulty/stuck sensors in the field produce values far outside operating
    limits (negative flow, absurd temperatures). Preprocessing must catch and
    winsorise these using the config ranges.
    """
    n_out = int(_OUTLIER_FRACTION * len(df))
    if n_out == 0:
        return df
    idx = rng.choice(len(df), size=n_out, replace=False)
    input_cols = config.features
    faulty_values = {
        "gas_temperature": 5000.0,     # thermocouple short -> absurd high
        "oxygen_pct": -3.0,            # analyser fault -> negative
        "flow_rate": -100.0,           # flow meter reversed
        "furnace_speed": 50.0,         # encoder glitch
        "feed_rate": 9999.0,           # belt scale spike
        "material_moisture": -5.0,     # probe fault
        "ambient_temp": 500.0,         # sensor on fire, apparently
    }
    for i in idx:
        col = rng.choice(input_cols)
        # Prefer a themed impossible value; fall back to a gross range violation.
        lo, hi = config.bounds(col)
        df.iloc[i, df.columns.get_loc(col)] = faulty_values.get(col, hi * 100)
    return df


def generate_dataset(config: ProcessConfig) -> pd.DataFrame:
    """Generate the full synthetic dataset as a DataFrame.

    Args:
        config: The process configuration (drives column names and limits).

    Returns:
        A DataFrame with all feature columns + the target, including injected
        missing values and outliers.
    """
    ds = config.section("dataset")
    n = int(ds.get("n_synthetic_rows", 10000))
    seed = int(ds.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    data = _sample_conditions(rng, n, config)
    quality = _quality_physics(data, config)
    quality = quality + rng.normal(0.0, _NOISE_STD, n)   # sensor noise
    data[config.target] = np.clip(quality, 0.0, 100.0)

    # Assemble in config order (features first, target last).
    ordered = config.features + [config.target]
    df = pd.DataFrame({c: data[c] for c in ordered})

    df = _inject_missing(df, config, rng)
    df = _inject_outliers(df, config, rng)
    return df


def main(config_path: str | None = None) -> Path:
    """CLI entry point: generate the dataset and write it to the configured path.

    Returns:
        The path the CSV was written to.
    """
    config = load_config(config_path)
    df = generate_dataset(config)
    out = config.data_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[generate_data] wrote {len(df):,} rows to {out}")
    print(f"[generate_data] target '{config.target}' "
          f"mean={df[config.target].mean():.2f} "
          f"min={df[config.target].min():.2f} "
          f"max={df[config.target].max():.2f}")
    return out


if __name__ == "__main__":
    main()
