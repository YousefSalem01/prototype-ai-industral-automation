"""Setpoint optimizer built on Optuna.

Given the CURRENT fixed operating conditions (feed rate, moisture, ambient),
search ONLY the controllable setpoints -- each strictly within its config
operating limits -- to maximise (or minimise) the model's predicted quality.

The optimizer is deliberately model-agnostic: it calls ``model.predict`` and
never inspects XGBoost internals, so swapping the surrogate later is trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna

from .model import QualityModel
from .schema import ProcessConfig

# Optuna is chatty by default; quiet it for pipeline output.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    """Outcome of a setpoint optimization."""

    fixed_conditions: dict[str, float]
    current_setpoints: dict[str, float]
    current_quality: float
    recommended_setpoints: dict[str, float]
    predicted_quality: float
    delta: float
    n_trials: int
    within_limits: bool = True
    limit_report: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (used by the API)."""
        return {
            "fixed_conditions": self.fixed_conditions,
            "current_setpoints": self.current_setpoints,
            "current_quality": round(self.current_quality, 3),
            "recommended_setpoints": {
                k: round(v, 4) for k, v in self.recommended_setpoints.items()
            },
            "predicted_quality": round(self.predicted_quality, 3),
            "delta": round(self.delta, 3),
            "n_trials": self.n_trials,
            "within_limits": self.within_limits,
        }


def _extract_fixed(state: dict[str, Any], config: ProcessConfig) -> dict[str, float]:
    """Pull the fixed-condition values out of a full state dict."""
    missing = [c for c in config.fixed if c not in state or state[c] is None]
    if missing:
        raise ValueError(
            f"Cannot optimize: fixed conditions not provided: {missing}. "
            f"The optimizer needs current values for all fixed columns."
        )
    return {c: float(state[c]) for c in config.fixed}


def optimize_setpoints(model: QualityModel, state: dict[str, Any],
                       config: ProcessConfig,
                       n_trials: int | None = None) -> OptimizationResult:
    """Search controllable setpoints to optimise predicted quality.

    Args:
        model: A trained :class:`~src.model.QualityModel`.
        state: Current process state. MUST contain all fixed conditions; may
            optionally contain current controllable setpoints (used to compute
            the improvement delta and a baseline quality).
        config: The process configuration.
        n_trials: Override for the number of Optuna trials.

    Returns:
        An :class:`OptimizationResult` with recommended setpoints, predicted
        quality, and the delta versus the current state.
    """
    opt_cfg = config.section("optimizer")
    n_trials = int(n_trials if n_trials is not None else opt_cfg.get("n_trials", 200))
    seed = int(opt_cfg.get("seed", 42))
    maximize = config.target_direction == "maximize"

    fixed = _extract_fixed(state, config)

    # Baseline: if current setpoints are supplied, score them; else midpoints.
    current_setpoints: dict[str, float] = {}
    for c in config.controllable:
        if c in state and state[c] is not None:
            current_setpoints[c] = float(state[c])
        else:
            lo, hi = config.bounds(c)
            current_setpoints[c] = 0.5 * (lo + hi)
    current_quality = model.predict({**fixed, **current_setpoints})

    def objective(trial: optuna.Trial) -> float:
        candidate = dict(fixed)
        for c in config.controllable:
            lo, hi = config.bounds(c)
            candidate[c] = trial.suggest_float(c, lo, hi)
        return model.predict(candidate)

    sampler = optuna.samplers.TPESampler(seed=seed)
    direction = "maximize" if maximize else "minimize"
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    recommended = {c: float(study.best_params[c]) for c in config.controllable}
    predicted = float(study.best_value)

    # Hard guarantee: recommended setpoints lie within operating limits.
    within = True
    limit_report: dict[str, tuple[float, float]] = {}
    for c in config.controllable:
        lo, hi = config.bounds(c)
        limit_report[c] = (lo, hi)
        val = recommended[c]
        if val < lo - 1e-9 or val > hi + 1e-9:
            within = False
        # Defensive clamp (Optuna respects bounds, but never emit an illegal
        # setpoint to a live plant).
        recommended[c] = min(max(val, lo), hi)

    delta = predicted - current_quality
    if not maximize:
        delta = current_quality - predicted

    return OptimizationResult(
        fixed_conditions=fixed,
        current_setpoints=current_setpoints,
        current_quality=current_quality,
        recommended_setpoints=recommended,
        predicted_quality=predicted,
        delta=delta,
        n_trials=n_trials,
        within_limits=within,
        limit_report=limit_report,
    )
