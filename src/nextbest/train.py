"""Training pipeline: simulate -> features -> bake-off -> per-offer models -> score.

Run:  python -m nextbest.train
Everything the API serves is produced here and written to models/ and data/.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import evaluate, explain, features, optimize, simulate
from .config import (CONTROL, DATA_DIR, FEATURES, MARGIN_RATE, MODEL_DIR,
                     OFFER_BY_KEY, OFFER_KEYS, QUADRANT_LABELS, SEED)
from .learners import LEARNERS, HAS_LGBM

# Uplift is a difference of two noisy rates, so decile estimates need volume:
# at 50k the bottom decile still wobbled around zero and the sleeping-dog
# quadrant was not visible. 120k gives a clean monotone staircase.
N_CUSTOMERS = 120_000


def _log(msg: str) -> None:
    print(f"[nextbest] {msg}", flush=True)


def run(n: int = N_CUSTOMERS, seed: int = SEED) -> dict:
    t_start = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _log(f"gradient booster: {'LightGBM' if HAS_LGBM else 'sklearn HistGradientBoosting'}")

    # ------------------------------------------------------------ 1. data
    _log(f"simulating {n:,} customers across a 6-arm randomised campaign")
    raw = simulate.generate(n=n, seed=seed)

    X = features.build(raw)
    features.assert_no_leakage(X)
    reference = features.population_reference(X)

    # the estimand for a pooled treated-vs-control model is the effect averaged
    # over the offers a treated customer could have received
    tau_avg = raw[[f"tau_{k}" for k in OFFER_KEYS]].mean(axis=1).to_numpy()

    y = raw["converted"].to_numpy()
    t = raw["treated"].to_numpy()

    idx_tr, idx_te = train_test_split(
        np.arange(len(raw)), test_size=0.30, random_state=seed, stratify=raw["arm"]
    )

    avg_margin = float(raw.loc[raw["converted"] == 1, "margin"].mean())
    avg_offer_cost = float(np.mean([OFFER_BY_KEY[k].cost(raw["avg_order_value"].mean())
                                    for k in OFFER_KEYS]))

    # -------------------------------------------------------- 2. bake-off
    _log("bake-off: propensity baseline vs four uplift meta-learners")
    bakeoff: list[dict] = []
    fitted: dict[str, object] = {}

    for key, cls in LEARNERS.items():
        t0 = time.time()
        model = cls().fit(X.iloc[idx_tr], t[idx_tr], y[idx_tr])
        pred = model.predict(X.iloc[idx_te])

        res = evaluate.evaluate_all(
            pred, t[idx_te], y[idx_te],
            true_tau=tau_avg[idx_te],
            margin_per_conversion=avg_margin,
            offer_cost=avg_offer_cost,
        )
        fitted[key] = model
        bakeoff.append({
            "key": key,
            "name": cls.name,
            "is_uplift": key != "propensity",
            "qini": res["qini"],
            "qini_ci": res["qini_ci"],
            "top_decile_uplift": res["top_decile_uplift"],
            "bottom_decile_uplift": res["bottom_decile_uplift"],
            "ground_truth": res["ground_truth"],
            "optimal_fraction": res["profit"]["optimal_fraction"],
            "optimal_profit": res["profit"]["optimal_profit"],
            "fit_seconds": round(time.time() - t0, 2),
        })
        _log(f"  {cls.name:22s} qini={res['qini']:8.1f}  "
             f"corr_vs_truth={res['ground_truth']['corr']:+.3f}  "
             f"top-decile={res['top_decile_uplift']:+.2f}pp")

    uplift_only = [b for b in bakeoff if b["is_uplift"]]
    champion_key = max(uplift_only, key=lambda b: b["qini"])["key"]
    champion_name = LEARNERS[champion_key].name
    _log(f"champion: {champion_name}")

    # detailed curves for the two headline models
    champ_pred = fitted[champion_key].predict(X.iloc[idx_te])
    prop_pred = fitted["propensity"].predict(X.iloc[idx_te])

    champ_eval = evaluate.evaluate_all(
        champ_pred, t[idx_te], y[idx_te], tau_avg[idx_te], avg_margin, avg_offer_cost)
    prop_eval = evaluate.evaluate_all(
        prop_pred, t[idx_te], y[idx_te], tau_avg[idx_te], avg_margin, avg_offer_cost)

    # -------------------------------------------- 3. one model per offer arm
    _log("training a champion-class model per offer arm")
    per_offer: dict[str, object] = {}
    offer_stats: list[dict] = []

    for key in OFFER_KEYS:
        sub = raw.index[(raw["arm"] == key) | (raw["arm"] == CONTROL)].to_numpy()
        Xs = X.iloc[sub]
        ts = (raw["arm"].to_numpy()[sub] == key).astype(int)
        ys = y[sub]

        s_tr, s_te = train_test_split(np.arange(len(sub)), test_size=0.30,
                                      random_state=seed, stratify=ts)
        m = LEARNERS[champion_key]().fit(Xs.iloc[s_tr], ts[s_tr], ys[s_tr])
        pred = m.predict(Xs.iloc[s_te])

        truth = raw[f"tau_{key}"].to_numpy()[sub][s_te]
        gt = evaluate.ground_truth_error(pred, truth)
        q = evaluate.qini_coefficient(pred, ts[s_te], ys[s_te])

        per_offer[key] = m
        offer_stats.append({
            "key": key,
            "label": OFFER_BY_KEY[key].label,
            "qini": round(q, 2),
            "corr_vs_truth": gt["corr"],
            "mean_uplift_pp": round(float(truth.mean()) * 100, 3),
            "n_train": int(len(s_tr)),
        })
        _log(f"  {key:8s} qini={q:8.1f}  corr={gt['corr']:+.3f}")

    # ------------------------------------------------------- 4. score everyone
    _log("scoring the full base against every offer")
    uplift = pd.DataFrame(index=raw["customer_id"].to_numpy())
    for key in OFFER_KEYS:
        uplift[key] = per_offer[key].predict(X)

    p_control = fitted[champion_key].predict_control(X)
    order_value = raw["avg_order_value"].to_numpy()

    choice = optimize.best_offer(uplift, order_value, p_control=p_control)

    # The quadrant describes how a customer responds to being contacted at all,
    # so it is built from the mean effect across offers. The best-offer effect is
    # a maximum over five noisy estimates and is biased upward -- classifying on
    # it labels four customers in five a persuadable.
    typical_uplift = uplift[list(OFFER_KEYS)].mean(axis=1).to_numpy()
    quadrant = optimize.classify_quadrant(
        typical_uplift, p_control, base_threshold=float(np.median(p_control))
    )

    scored = pd.DataFrame({
        "customer_id": raw["customer_id"].to_numpy(),
        "category": raw["category"].to_numpy(),
        "recency_days": raw["recency_days"].to_numpy(),
        "frequency_12m": raw["frequency_12m"].to_numpy(),
        "monetary_12m": raw["monetary_12m"].to_numpy(),
        "avg_order_value": order_value,
        "tenure_years": raw["tenure_years"].to_numpy(),
        "engagement": raw["engagement"].to_numpy(),
        "discount_affinity": raw["discount_affinity"].to_numpy(),
        "price_tier": raw["price_tier"].to_numpy(),
        "is_registered": raw["is_registered"].to_numpy(),
        "p_control": p_control,
        "typical_uplift": typical_uplift,
        "quadrant": quadrant,
        "offer": choice["offer"].to_numpy(),
        "offer_uplift": choice["offer_uplift"].to_numpy(),
        "expected_profit": choice["expected_profit"].to_numpy(),
        "expected_cost": choice["expected_cost"].to_numpy(),
        "margin": order_value * MARGIN_RATE,
        "tau_true_avg": tau_avg,
    })
    for key in OFFER_KEYS:
        scored[f"uplift_{key}"] = uplift[key].to_numpy()
    scored = scored.set_index("customer_id", drop=False)

    # ------------------------------------------------------- 5. aggregate views
    quad_counts = scored["quadrant"].value_counts().to_dict()
    quadrants = [
        {"key": k, "label": QUADRANT_LABELS[k], "count": int(quad_counts.get(k, 0)),
         "share": round(quad_counts.get(k, 0) / len(scored) * 100, 2)}
        for k in QUADRANT_LABELS
    ]

    segments = ["Lapsing High-Value", "Core Regulars", "Discount Seekers",
                "New & Curious", "Dormant"]

    money_median = float(scored["monetary_12m"].median())

    def segment_of(r) -> str:
        if r.recency_days > 400:
            return "Dormant"
        if r.tenure_years < 0.9:
            return "New & Curious"
        if r.discount_affinity > 0.55:
            return "Discount Seekers"
        if r.recency_days > 130 and r.monetary_12m > money_median:
            return "Lapsing High-Value"
        return "Core Regulars"

    scored["segment"] = [segment_of(r) for r in scored.itertuples()]

    offer_matrix = []
    for seg in segments:
        sub = scored[scored["segment"] == seg]
        row = {"segment": seg, "n": int(len(sub))}
        for key in OFFER_KEYS:
            row[key] = round(float(sub[f"uplift_{key}"].mean()) * 100, 3) if len(sub) else 0.0
        offer_matrix.append(row)

    _log("computing global uplift drivers")
    importance = explain.global_importance(per_offer["pct10"], X, reference, sample=600)

    # ------------------------------------------------------------ 6. persist
    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_customers": int(len(scored)),
        "booster": "LightGBM" if HAS_LGBM else "sklearn HistGradientBoosting",
        "champion": {"key": champion_key, "name": champion_name},
        "bakeoff": bakeoff,
        "champion_eval": champ_eval,
        "propensity_eval": prop_eval,
        "offers": offer_stats,
        "quadrants": quadrants,
        "offer_matrix": offer_matrix,
        "segments": segments,
        "importance": importance,
        "economics": {
            "avg_margin_per_conversion": round(avg_margin, 2),
            "avg_offer_cost": round(avg_offer_cost, 2),
            "margin_rate": MARGIN_RATE,
        },
        "arm_rates": [
            {"arm": a,
             "n": int((raw["arm"] == a).sum()),
             "conversion_rate": round(float(raw.loc[raw["arm"] == a, "converted"].mean()) * 100, 3)}
            for a in [CONTROL, *OFFER_KEYS]
        ],
        "train_seconds": round(time.time() - t_start, 1),
    }

    scored.to_pickle(DATA_DIR / "scored.pkl")
    X.set_index(raw["customer_id"].to_numpy()).to_pickle(DATA_DIR / "features.pkl")
    raw.to_pickle(DATA_DIR / "raw.pkl")
    joblib.dump(
        {"per_offer": per_offer, "champion": fitted[champion_key],
         "champion_key": champion_key, "reference": reference, "features": list(FEATURES)},
        MODEL_DIR / "artifacts.joblib",
    )
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _log(f"done in {metrics['train_seconds']}s -> models/metrics.json, data/scored.pkl")
    return metrics


if __name__ == "__main__":
    run()
