"""Local attribution for an uplift score.

This is NOT SHAP. It is counterfactual median substitution: for each feature,
re-score the customer with that one feature replaced by the population median and
report the change in predicted uplift.

    contribution_j = tau(x) - tau(x with feature j set to median)

Honest about what it is: a one-at-a-time sensitivity, so it ignores interactions
and the contributions do not sum exactly to the prediction the way Shapley values
do. In exchange it needs no extra dependency, runs in milliseconds, and answers
the question a retention manager actually asks -- "what about this customer makes
them worth contacting?"

Swapping in `shap.TreeExplainer` later changes only this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_LABELS, FEATURES


def explain_row(model, row: pd.Series, reference: pd.Series, top_k: int = 6) -> list[dict]:
    base = np.asarray([row[f] for f in FEATURES], dtype=float).reshape(1, -1)
    tau_base = float(model.predict(base)[0])

    variants = np.repeat(base, len(FEATURES), axis=0)
    for j, feat in enumerate(FEATURES):
        variants[j, j] = float(reference[feat])

    tau_counterfactual = model.predict(variants)
    contributions = tau_base - np.asarray(tau_counterfactual, dtype=float)

    drivers = []
    for j, feat in enumerate(FEATURES):
        drivers.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": round(float(row[feat]), 4),
            "contribution": round(float(contributions[j]) * 100, 4),  # in pp
        })

    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return drivers[:top_k]


def global_importance(model, X: pd.DataFrame, reference: pd.Series,
                      sample: int = 800, seed: int = 3) -> list[dict]:
    """Mean absolute local contribution over a sample -- a cheap global view."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X.iloc[idx]

    base = Xs.to_numpy(dtype=float)
    tau_base = np.asarray(model.predict(base), dtype=float)

    rows = []
    for j, feat in enumerate(FEATURES):
        swapped = base.copy()
        swapped[:, j] = float(reference[feat])
        delta = np.abs(tau_base - np.asarray(model.predict(swapped), dtype=float))
        rows.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "importance": round(float(delta.mean()) * 100, 4),
        })

    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows
