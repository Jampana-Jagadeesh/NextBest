"""Synthetic multi-arm randomised campaign with KNOWN ground-truth uplift.

Why simulate when real uplift datasets exist? Because on real data you only ever
observe one arm per customer, so you can never check whether a per-person uplift
estimate is *correct* -- only whether the ranking looks sensible in aggregate.
Here we know each customer's true treatment effect for every offer, so we can
measure an estimator's actual error and pick a learner on evidence.

The generative story deliberately contains all four uplift archetypes:

  persuadable   mid-lapse customers with high discount affinity
  sure thing    frequent, recent, already-converting customers
  lost cause    long-lapsed, disengaged
  sleeping dog  premium full-price regulars, annoyed by discount mail
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (ARMS, CATEGORIES, CHANNELS, CONTROL, MARGIN_RATE, OFFERS,
                     PRODUCTS, SEED)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bell(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian bump in [0, 1] -- used for the 'lapsing but reachable' window."""
    return np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def generate(n: int = 60_000, seed: int = SEED,
             arms: list[str] | None = None) -> pd.DataFrame:
    """Simulate a randomised campaign.

    `arms` selects the experiment design. Default (None) is the full six-arm
    setup -- 25% control plus five offers -- which is what the offer-selection
    layer needs.

    Passing a shorter list gives an evenly split design, e.g. ``arms=["pct10"]``
    for a plain 50/50 A/B test. That matters more than it sounds: uplift is a
    difference between two measured rates, so precision is driven by the size of
    the SMALLEST arm, not by total rows. Splitting 8,000 customers six ways
    leaves ~600 controls and the estimates collapse into noise, while the same
    8,000 split two ways is perfectly workable.
    """
    rng = np.random.default_rng(seed)

    category = rng.choice(CATEGORIES, size=n, p=[0.30, 0.26, 0.18, 0.14, 0.12])
    # what they buy most inside that category, and where they buy it
    product = np.array([rng.choice(PRODUCTS[c]) for c in category])
    channel = np.array([
        rng.choice(CHANNELS, p=([0.55, 0.28, 0.17] if c == "grocery" else
                                [0.30, 0.42, 0.28] if c in ("apparel", "beauty") else
                                [0.24, 0.50, 0.26]))
        for c in category
    ])

    # ---------------------------------------------------------- raw behaviour
    recency_days = np.clip(rng.gamma(shape=2.0, scale=52.0, size=n), 1, 730)
    recency_n = recency_days / 730.0                      # 0 = today, 1 = 2y ago

    tenure_days = np.clip(rng.gamma(2.4, 420.0, n), 30, 4200)
    tenure_years = tenure_days / 365.25

    # frequency falls off with recency and rises with tenure
    freq_lam = np.clip(6.5 * np.exp(-2.1 * recency_n) + 0.35 * tenure_years, 0.2, 24)
    frequency_12m = rng.poisson(freq_lam).astype(float)

    aov_base = {"grocery": 68, "apparel": 95, "home": 130, "beauty": 45, "electronics": 260}
    aov_mu = np.array([aov_base[c] for c in category], dtype=float)
    avg_order_value = np.clip(rng.lognormal(np.log(aov_mu), 0.42), 8, 3000)

    items_base = {"grocery": 11.0, "apparel": 2.4, "home": 2.0, "beauty": 3.1, "electronics": 1.4}
    items_per_order = np.clip(
        rng.poisson([items_base[c] for c in category]) + 1, 1, 40).astype(float)

    monetary_12m = frequency_12m * avg_order_value * rng.uniform(0.85, 1.15, n)

    engagement = np.clip(rng.beta(2.1, 3.0, n) + 0.25 * np.exp(-2.5 * recency_n), 0, 1)
    discount_affinity = np.clip(rng.beta(2.0, 2.6, n), 0, 1)
    promo_share = np.clip(discount_affinity * rng.uniform(0.55, 1.25, n), 0, 1)

    price_tier = rng.choice([1, 2, 3], size=n, p=[0.34, 0.44, 0.22]).astype(float)
    # premium customers skew to lower discount affinity -- this creates sleeping dogs
    discount_affinity = np.clip(discount_affinity - 0.18 * (price_tier - 2), 0, 1)

    category_diversity = np.clip(rng.poisson(1.6 + 0.25 * frequency_12m), 1, 14).astype(float)
    support_tickets = rng.poisson(0.35, n).astype(float)
    is_registered = (rng.uniform(size=n) < _sigmoid(1.2 * (tenure_years - 1.4))).astype(float)
    spend_trend = np.clip(rng.normal(1.0, 0.34, n) * np.exp(-0.9 * recency_n), 0.05, 3.0)

    freq_n = np.clip(frequency_12m / 12.0, 0, 1.6)
    mon_n = np.clip(np.log1p(monetary_12m) / np.log(1 + 4000), 0, 1.5)
    lapse_score = _bell(recency_n, mu=0.30, sigma=0.13)   # the reachable window

    # ------------------------------------------------- baseline P(buy|control)
    z0 = (
        -2.55
        + 1.25 * freq_n
        + 0.85 * engagement
        - 1.65 * recency_n
        + 0.55 * mon_n
        + 0.28 * is_registered
        + 0.22 * spend_trend
        + rng.normal(0, 0.22, n)
    )
    p_control = _sigmoid(z0)

    # ------------------------------------------------------ true uplift terms
    # Positive drivers. The -0.07 intercept matters: without it every customer
    # has some positive effect and the bottom of the ranking never goes negative,
    # which would erase the sleeping-dog quadrant the product is built around.
    persuadable_drive = (
        0.24 * lapse_score + 0.13 * discount_affinity + 0.05 * engagement - 0.07
    )
    # sure things: already near-certain, so there is little room left to move
    headroom = np.clip(1.0 - p_control / 0.60, 0.05, 1.0)
    # sleeping dogs: premium, full-price, still active -> discount mail backfires
    premium_regular = (price_tier >= 3).astype(float) * (1.0 - discount_affinity) * np.clip(freq_n, 0, 1)
    dog_drive = 0.26 * premium_regular + 0.05 * (support_tickets > 1)

    base_tau = persuadable_drive * headroom - dog_drive

    frame: dict[str, np.ndarray] = {}
    for off in OFFERS:
        # not every reward suits every category
        cat_fit = np.ones(n)
        if off.key == "upgrade":                       # premium add-ons
            cat_fit = np.where(np.isin(category, ["electronics", "home"]), 1.35, 0.50)
        elif off.key == "ship":                        # shipping matters where baskets travel
            cat_fit = np.where(np.isin(category, ["apparel", "electronics", "home"]), 1.30, 0.55)
        elif off.key == "bundle":                      # multi-buy suits fast-moving goods
            cat_fit = np.where(np.isin(category, ["grocery", "beauty"]), 1.28, 0.80)

        tau = off.amp * cat_fit * base_tau + rng.normal(0, 0.006, n)
        # a treated probability must stay a probability
        tau = np.clip(tau, -p_control * 0.9, (1.0 - p_control) * 0.9)
        frame[f"tau_{off.key}"] = tau

    # ---------------------------------------------------- randomised assignment
    if arms is None:
        # 25% control, 15% each offer -- a proper multi-arm RCT
        arm_choices = list(ARMS)
        probs = [0.25] + [0.15] * len(OFFERS)
    else:
        unknown = [a for a in arms if a not in [o.key for o in OFFERS]]
        if unknown:
            raise ValueError(f"unknown offer arms: {unknown}")
        arm_choices = [CONTROL] + list(arms)
        probs = [1.0 / len(arm_choices)] * len(arm_choices)

    arm = rng.choice(arm_choices, size=n, p=probs)

    tau_assigned = np.zeros(n)
    for off in OFFERS:
        mask = arm == off.key
        tau_assigned[mask] = frame[f"tau_{off.key}"][mask]

    p_obs = np.clip(p_control + tau_assigned, 0.001, 0.995)
    converted = (rng.uniform(size=n) < p_obs).astype(int)

    order_value = np.where(converted == 1, avg_order_value * rng.uniform(0.7, 1.4, n), 0.0)
    revenue = order_value
    margin = order_value * MARGIN_RATE

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "category": category,
            "product": product,
            "channel": channel,
            "items_per_order": items_per_order,
            "recency_days": recency_days.round(1),
            "recency_n": recency_n,
            "frequency_12m": frequency_12m,
            "monetary_12m": monetary_12m.round(2),
            "avg_order_value": avg_order_value.round(2),
            "tenure_years": tenure_years.round(2),
            "engagement": engagement.round(4),
            "discount_affinity": discount_affinity.round(4),
            "promo_share": promo_share.round(4),
            "price_tier": price_tier,
            "category_diversity": category_diversity,
            "support_tickets": support_tickets,
            "is_registered": is_registered,
            "spend_trend": spend_trend.round(4),
            "lapse_score": lapse_score.round(4),
            "arm": arm,
            "treated": (arm != CONTROL).astype(int),
            "converted": converted,
            "revenue": revenue.round(2),
            "margin": margin.round(2),
            # ---- ground truth. NEVER a model input. Held out for validation only.
            "p_control_true": p_control.round(6),
            **{k: np.round(v, 6) for k, v in frame.items()},
        }
    )
    df["tau_true"] = np.round(tau_assigned, 6)
    return df


def ground_truth_columns() -> list[str]:
    """Columns a model must never see."""
    return ["p_control_true", "tau_true"] + [f"tau_{o.key}" for o in OFFERS]


if __name__ == "__main__":  # pragma: no cover
    d = generate(20_000)
    print(d.groupby("arm")["converted"].agg(["mean", "size"]))
    print("mean true tau by offer:")
    for o in OFFERS:
        print(f"  {o.key:8s} {d[f'tau_{o.key}'].mean():+.4f}")
