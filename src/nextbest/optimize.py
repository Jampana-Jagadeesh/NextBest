"""Offer selection and budget allocation.

An uplift score is not a decision. Turning it into one takes two more steps:

  1. per customer, choose the offer with the highest expected incremental profit
  2. under a finite budget, choose WHICH of those customers to actually contact

Step 2 is a knapsack. Items have a cost (contact + expected redemption) and a
value (expected incremental margin). Because every item's value/cost ratio is
independent of the others, sorting by that ratio and filling greedily is optimal
up to the single boundary item -- the classic fractional-knapsack argument -- so
we do not need an LP solver to be within one customer of the optimum.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import (CONTACT_COST, MARGIN_RATE, OFFER_BY_KEY, OFFER_KEYS,
                     SURE_THING_BASE, UPLIFT_HI, UPLIFT_LO)


def expected_profit_matrix(uplift: pd.DataFrame, order_value: np.ndarray,
                           p_control: np.ndarray | None = None) -> pd.DataFrame:
    """Expected incremental profit per customer per offer.

        E[profit] = uplift * margin - P(buy | contacted) * offer_cost - contact_cost

    Note which term the offer cost attaches to. A discount is redeemed by EVERY
    customer who buys after being contacted, not only by the extra sales it
    caused -- so it is charged against P(buy | contacted), not against uplift.
    Charging it to incremental buyers alone hides the entire cost of discounting
    someone who was going to purchase regardless, which is the waste this whole
    project exists to measure.

    `p_control` is the baseline purchase probability. If omitted the function
    falls back to charging the discount on uplift alone and warns, because that
    understates the cost of every offer.
    """
    order_value = np.asarray(order_value, float)
    margin = order_value * MARGIN_RATE
    out = pd.DataFrame(index=uplift.index)

    if p_control is None:
        warnings.warn(
            "expected_profit_matrix called without p_control: the discount will be "
            "charged only to incremental buyers, which understates offer cost.",
            RuntimeWarning, stacklevel=2)

    for key in OFFER_KEYS:
        cost = OFFER_BY_KEY[key].cost(order_value)
        u = uplift[key].to_numpy()
        if p_control is None:
            out[key] = u * (margin - cost) - CONTACT_COST
        else:
            p_treated = np.clip(np.asarray(p_control, float) + u, 0.0, 1.0)
            out[key] = u * margin - p_treated * cost - CONTACT_COST
    return out


def best_offer(uplift: pd.DataFrame, order_value: np.ndarray,
               allowed: list[str] | None = None,
               p_control: np.ndarray | None = None) -> pd.DataFrame:
    """Pick argmax offer per customer, restricted to `allowed` if given."""
    keys = [k for k in OFFER_KEYS if allowed is None or k in allowed]
    if not keys:
        keys = list(OFFER_KEYS)

    profit = expected_profit_matrix(uplift, order_value, p_control)[keys]
    best_key = profit.idxmax(axis=1)
    best_profit = profit.max(axis=1)
    best_uplift = np.array([uplift.loc[i, k] for i, k in zip(uplift.index, best_key)])

    order_value = np.asarray(order_value, float)
    cost = np.array([OFFER_BY_KEY[k].cost(v) for k, v in zip(best_key, order_value)])

    # What contacting this person really costs: the message, plus the discount
    # redeemed by them if they buy at all -- not only if we caused the purchase.
    if p_control is None:
        redeem = np.clip(best_uplift, 0, None)
    else:
        redeem = np.clip(np.asarray(p_control, float) + best_uplift, 0.0, 1.0)

    return pd.DataFrame({
        "offer": best_key.to_numpy(),
        "offer_uplift": best_uplift,
        "expected_profit": best_profit.to_numpy(),
        "expected_cost": CONTACT_COST + redeem * cost,
    }, index=uplift.index)


def classify_quadrant(uplift: np.ndarray, p_control: np.ndarray,
                      base_threshold: float | None = None) -> np.ndarray:
    """Map (uplift, baseline demand) onto the four archetypes.

    `uplift` must be the effect of a TYPICAL contact -- the mean across offers --
    not the best-offer effect. Ranking by the max over five offers is biased
    upward by construction, and classifying on it labels almost everyone a
    persuadable.
    """
    uplift = np.asarray(uplift, float)
    p_control = np.asarray(p_control, float)
    thr = SURE_THING_BASE if base_threshold is None else float(base_threshold)

    flat = (uplift <= UPLIFT_HI) & (uplift >= -UPLIFT_LO)
    negative = uplift < -UPLIFT_LO
    has_demand = p_control >= thr

    out = np.empty(len(uplift), dtype=object)
    out[uplift > UPLIFT_HI] = "persuadable"
    # A sleeping dog is defined by having something to lose: they would have
    # bought, and contact talks them out of it. Someone who was never going to
    # buy and scores slightly negative is a lost cause and the negative is
    # mostly noise -- classifying on uplift alone conflates the two.
    out[negative & has_demand] = "sleeping_dog"
    out[negative & ~has_demand] = "lost_cause"
    out[flat & has_demand] = "sure_thing"
    out[flat & ~has_demand] = "lost_cause"
    return out


def allocate(scored: pd.DataFrame, budget: float,
             min_uplift: float = 0.0) -> dict:
    """Greedy budget-constrained selection.

    `scored` needs: expected_profit, expected_cost, offer_uplift, margin.
    Customers with non-positive uplift are suppressed outright -- contacting a
    sleeping dog has negative expected value at ANY budget, so it is never a
    question of affordability.
    """
    df = scored.copy()

    eligible = df[(df["offer_uplift"] > min_uplift) & (df["expected_profit"] > 0)]
    suppressed = len(df) - len(eligible)

    if eligible.empty:
        return {
            "selected_ids": [], "n_selected": 0, "n_suppressed": int(suppressed),
            "spend": 0.0, "expected_profit": 0.0,
            "expected_incremental_conversions": 0.0, "budget": float(budget),
            "cutoff_fraction": 0.0,
        }

    # value density: incremental profit per unit of budget consumed
    eligible = eligible.assign(
        density=eligible["expected_profit"] / eligible["expected_cost"].clip(lower=1e-6)
    ).sort_values("density", ascending=False)

    cum_cost = eligible["expected_cost"].cumsum()
    take = cum_cost <= budget
    chosen = eligible[take]

    return {
        "selected_ids": chosen.index.tolist(),
        "n_selected": int(len(chosen)),
        "n_suppressed": int(suppressed),
        "spend": round(float(chosen["expected_cost"].sum()), 2),
        "expected_profit": round(float(chosen["expected_profit"].sum()), 2),
        "expected_incremental_conversions": round(float(chosen["offer_uplift"].sum()), 1),
        "budget": float(budget),
        "cutoff_fraction": round(len(chosen) / max(len(df), 1), 4),
    }
