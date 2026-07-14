"""SHAP explanations: global feature importance and per-recommendation drivers.

Uses ``shap.TreeExplainer``, which is exact and fast for XGBoost. Two views:
  * global_importance  -> mean(|SHAP|) across a data sample (dataset-wide).
  * explain_state      -> signed SHAP for ONE state (why this recommendation).

Explanations are returned as plain dicts/lists so the API can serialise them
and the desktop app can render them without any Python-side coupling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from .model import QualityModel
from .preprocess import prepare_state
from .schema import ProcessConfig


class QualityExplainer:
    """SHAP explainer bound to a trained :class:`~src.model.QualityModel`."""

    def __init__(self, model: QualityModel, config: ProcessConfig) -> None:
        if model.model is None:
            raise RuntimeError("Cannot explain an untrained model.")
        self.model = model
        self.config = config
        self._explainer = shap.TreeExplainer(model.model)

    # ---- global -------------------------------------------------------------

    def global_importance(self, X: pd.DataFrame,
                          max_samples: int = 1000) -> list[dict[str, Any]]:
        """Dataset-wide feature importance via mean absolute SHAP value.

        Args:
            X: Feature frame (e.g. the cleaned training features).
            max_samples: Cap on rows used, for speed on large datasets.

        Returns:
            List of ``{"feature", "role", "mean_abs_shap", "importance_pct"}``
            sorted most-important first.
        """
        Xs = X[self.config.features]
        if len(Xs) > max_samples:
            Xs = Xs.sample(max_samples,
                           random_state=int(self.config.section("dataset")
                                            .get("random_seed", 42)))
        shap_values = self._explainer.shap_values(Xs)
        mean_abs = np.abs(shap_values).mean(axis=0)
        total = float(mean_abs.sum()) or 1.0
        rows = [
            {
                "feature": feat,
                "role": self.config.columns[feat].role,
                "unit": self.config.columns[feat].unit,
                "mean_abs_shap": float(mean_abs[i]),
                "importance_pct": round(100.0 * float(mean_abs[i]) / total, 2),
            }
            for i, feat in enumerate(self.config.features)
        ]
        rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
        return rows

    # ---- local --------------------------------------------------------------

    def explain_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Explain the prediction for a single process state.

        Args:
            state: Mapping of feature name -> value (partial allowed).

        Returns:
            Dict with the base value, the predicted value, and a per-feature
            list of signed SHAP contributions (value, shap, direction).
        """
        X = prepare_state(state, self.config, impute_values=self.model.impute_values)
        shap_values = self._explainer.shap_values(X)[0]
        base = float(np.ravel(self._explainer.expected_value)[0])
        prediction = base + float(np.sum(shap_values))

        contributions = []
        for i, feat in enumerate(self.config.features):
            contrib = float(shap_values[i])
            contributions.append({
                "feature": feat,
                "role": self.config.columns[feat].role,
                "unit": self.config.columns[feat].unit,
                "value": float(X.iloc[0, i]),
                "shap": contrib,
                "direction": "increases" if contrib >= 0 else "decreases",
            })
        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)

        return {
            "base_value": round(base, 3),
            "predicted_quality": round(prediction, 3),
            "contributions": contributions,
        }

    def top_drivers(self, state: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
        """Return the top-``k`` signed drivers for a single state (concise view)."""
        return self.explain_state(state)["contributions"][:k]
