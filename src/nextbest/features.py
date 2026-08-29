"""Feature construction.

In production every one of these must be computed strictly from data timestamped
BEFORE the campaign was sent. The simulator generates pre-campaign state only, so
the contract holds by construction here -- `assert_no_leakage` is the guard that
would catch a violation once real event data is wired in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORIES, FEATURES
from .simulate import ground_truth_columns


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame containing exactly FEATURES, in FEATURES order."""
    out = pd.DataFrame(index=df.index)

    out["recency_n"] = df["recency_n"]
    out["frequency_12m"] = df["frequency_12m"]
    out["monetary_12m"] = np.log1p(df["monetary_12m"])
    out["avg_order_value"] = np.log1p(df["avg_order_value"])
    out["tenure_years"] = df["tenure_years"]
    out["engagement"] = df["engagement"]
    out["discount_affinity"] = df["discount_affinity"]
    out["price_tier"] = df["price_tier"]
    out["category_diversity"] = df["category_diversity"]
    out["support_tickets"] = df["support_tickets"]
    out["is_registered"] = df["is_registered"]
    out["spend_trend"] = df["spend_trend"]
    out["promo_share"] = df["promo_share"]
    out["lapse_score"] = df["lapse_score"]

    for c in CATEGORIES:
        out[f"cat_{c}"] = (df["category"] == c).astype(float)

    missing = [c for c in FEATURES if c not in out.columns]
    if missing:
        raise KeyError(f"feature builder is missing: {missing}")
    return out[list(FEATURES)].astype(float)


def assert_no_leakage(feature_frame: pd.DataFrame) -> None:
    """Fail loudly if an outcome or ground-truth column reached the model input."""
    banned = set(ground_truth_columns()) | {"converted", "revenue", "margin", "arm", "treated"}
    overlap = banned.intersection(feature_frame.columns)
    if overlap:
        raise AssertionError(f"LEAKAGE: outcome columns present in features: {sorted(overlap)}")


def population_reference(feature_frame: pd.DataFrame) -> pd.Series:
    """Median profile -- the counterfactual baseline used by the explainer."""
    return feature_frame.median(numeric_only=True)
