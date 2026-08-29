"""Uplift evaluation.

There is no per-customer uplift label -- each person is observed in exactly one
arm -- so nothing here is an accuracy metric. Everything works by ranking the
population, then comparing treated to control WITHIN each prefix or bin.
"""
from __future__ import annotations

import numpy as np

from .config import CONTACT_COST


def qini_curve(score: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    """Cumulative incremental conversions as we walk down the ranking.

    At prefix k:  qini(k) = responders_treated(k) - responders_control(k) * Nt(k)/Nc(k)

    The control responders are scaled to the treated head-count so the two arms
    are comparable at every depth.
    """
    score = np.asarray(score, float)
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    order = np.argsort(-score, kind="mergesort")
    t, y = t[order], y[order]

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1.0 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1.0 - t))

    ratio = np.divide(cum_t, cum_c, out=np.zeros_like(cum_t), where=cum_c > 0)
    q = cum_yt - cum_yc * ratio

    n = len(score)
    x = np.arange(1, n + 1, dtype=float) / n
    return {"x": x, "q": q, "n": n}


def qini_coefficient(score, t, y) -> float:
    """Area between the model's Qini curve and random targeting.

    Units are incremental conversions: "this ranking captures N more incremental
    conversions than shuffling the list would, integrated over all depths."
    """
    c = qini_curve(score, t, y)
    x, q = c["x"], c["q"]
    area_model = float(np.trapezoid(q, x))
    area_random = float(q[-1]) / 2.0
    return area_model - area_random


def qini_curve_points(score, t, y, n_points: int = 160) -> list[dict]:
    """Downsampled curve for the API, with the random reference line."""
    c = qini_curve(score, t, y)
    x, q, n = c["x"], c["q"], c["n"]
    idx = np.unique(np.linspace(0, n - 1, min(n_points, n)).astype(int))
    total = float(q[-1])
    return [
        {"x": round(float(x[i]), 4),
         "model": round(float(q[i]), 2),
         "random": round(float(x[i]) * total, 2)}
        for i in idx
    ]


def bootstrap_qini(score, t, y, n_boot: int = 40, seed: int = 11) -> dict:
    """A single Qini number on one split is close to meaningless. Resample."""
    rng = np.random.default_rng(seed)
    score, t, y = np.asarray(score), np.asarray(t), np.asarray(y)
    n = len(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(qini_coefficient(score[idx], t[idx], y[idx]))
    vals = np.array(vals)
    return {
        "mean": round(float(vals.mean()), 2),
        "lo": round(float(np.percentile(vals, 2.5)), 2),
        "hi": round(float(np.percentile(vals, 97.5)), 2),
    }


def uplift_by_decile(score, t, y, bins: int = 10) -> list[dict]:
    """Observed treated-minus-control response rate inside each score bin.

    A working uplift model produces a monotone decreasing staircase, and the
    bottom bins can legitimately go negative -- that is the sleeping dogs.
    """
    score = np.asarray(score, float)
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    order = np.argsort(-score, kind="mergesort")
    t, y = t[order], y[order]
    chunks = np.array_split(np.arange(len(score)), bins)

    out = []
    for i, ch in enumerate(chunks, start=1):
        tt, yy = t[ch], y[ch]
        nt, nc = tt.sum(), (1 - tt).sum()
        rt = float((yy * tt).sum() / nt) if nt else 0.0
        rc = float((yy * (1 - tt)).sum() / nc) if nc else 0.0
        out.append({
            "decile": i,
            "uplift": round((rt - rc) * 100, 3),
            "treated_rate": round(rt * 100, 3),
            "control_rate": round(rc * 100, 3),
            "n": int(len(ch)),
        })
    return out


def profit_curve(score, t, y, margin_per_conversion: float,
                 offer_cost: float, contact_cost: float = CONTACT_COST,
                 n_points: int = 120) -> dict:
    """Money view of the Qini curve.

        profit(k) = incremental_conversions(k) * (margin - offer_cost)
                    - k * contact_cost

    The peak is the number that goes in front of a CFO: contact everyone to the
    left of it, nobody to the right.
    """
    c = qini_curve(score, t, y)
    x, q, n = c["x"], c["q"], c["n"]
    k = np.arange(1, n + 1, dtype=float)

    profit = q * (margin_per_conversion - offer_cost) - k * contact_cost
    best = int(np.argmax(profit))

    idx = np.unique(np.linspace(0, n - 1, min(n_points, n)).astype(int))
    return {
        "points": [
            {"x": round(float(x[i]), 4), "profit": round(float(profit[i]), 2)}
            for i in idx
        ],
        "optimal_fraction": round(float(x[best]), 4),
        "optimal_profit": round(float(profit[best]), 2),
        "profit_at_full": round(float(profit[-1]), 2),
        "incremental_conversions_at_optimum": round(float(q[best]), 1),
    }


def ground_truth_error(pred_tau: np.ndarray, true_tau: np.ndarray) -> dict:
    """Only possible because we simulated. On real data this cannot be computed,
    which is precisely why the simulator earns its place in the pipeline."""
    pred = np.asarray(pred_tau, float)
    true = np.asarray(true_tau, float)
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    if pred.std() < 1e-12 or true.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(pred, true)[0, 1])
    return {"mae": round(mae, 5), "rmse": round(rmse, 5), "corr": round(corr, 4)}


def evaluate_all(score, t, y, true_tau=None, margin_per_conversion: float = 30.0,
                 offer_cost: float = 6.0) -> dict:
    res = {
        "qini": round(qini_coefficient(score, t, y), 2),
        "qini_ci": bootstrap_qini(score, t, y),
        "deciles": uplift_by_decile(score, t, y),
        "curve": qini_curve_points(score, t, y),
        "profit": profit_curve(score, t, y, margin_per_conversion, offer_cost),
    }
    res["top_decile_uplift"] = res["deciles"][0]["uplift"]
    res["bottom_decile_uplift"] = res["deciles"][-1]["uplift"]
    if true_tau is not None:
        res["ground_truth"] = ground_truth_error(score, true_tau)
    return res
