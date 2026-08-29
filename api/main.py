"""NextBest serving API.

Loads the artifacts produced by `python -m nextbest.train` and exposes them to
the console. Batch scores are precomputed, so every read endpoint is a frame
slice; only /api/customer/{id} and /api/audience do work per request.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nextbest import explain, optimize  # noqa: E402
from nextbest.config import (CONTACT_COST, OFFER_BY_KEY, OFFER_KEYS,  # noqa: E402
                             QUADRANT_LABELS, CATEGORIES)

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

app = FastAPI(title="NextBest", version="1.0.0")

STATE: dict = {}


def _load() -> None:
    missing = [p.name for p in (DATA_DIR / "scored.pkl", MODEL_DIR / "artifacts.joblib",
                                MODEL_DIR / "metrics.json") if not p.exists()]
    if missing:
        raise RuntimeError(
            f"missing artifacts: {missing}. Run `python -m nextbest.train` first."
        )
    STATE["scored"] = pd.read_pickle(DATA_DIR / "scored.pkl")
    STATE["features"] = pd.read_pickle(DATA_DIR / "features.pkl")
    STATE["artifacts"] = joblib.load(MODEL_DIR / "artifacts.joblib")
    STATE["metrics"] = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))


@app.on_event("startup")
def startup() -> None:
    _load()
    print(f"[api] loaded {len(STATE['scored']):,} scored customers", flush=True)


def _py(o):
    """numpy -> json-safe."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ------------------------------------------------------------------ overview
@app.get("/api/overview")
def overview():
    s: pd.DataFrame = STATE["scored"]
    m = STATE["metrics"]

    targeted = s[s["expected_profit"] > 0]
    return {
        "n_customers": int(len(s)),
        "champion": m["champion"],
        "booster": m["booster"],
        "generated_at": m["generated_at"],
        "train_seconds": m["train_seconds"],
        "economics": m["economics"],
        "kpi": {
            "addressable": int(len(targeted)),
            "expected_incremental_conversions": round(float(targeted["offer_uplift"].sum()), 0),
            "expected_incremental_profit": round(float(targeted["expected_profit"].sum()), 0),
            "expected_spend": round(float(targeted["expected_cost"].sum()), 0),
            "suppressed": int((s["offer_uplift"] <= 0).sum()),
            "mean_uplift_pp": round(float(s["typical_uplift"].mean()) * 100, 2),
        },
        "quadrants": m["quadrants"],
        "arm_rates": m["arm_rates"],
        "qini": {
            "champion": m["champion_eval"]["curve"],
            "propensity": m["propensity_eval"]["curve"],
        },
        "deciles": {
            "champion": m["champion_eval"]["deciles"],
            "propensity": m["propensity_eval"]["deciles"],
        },
        "profit": {
            "champion": m["champion_eval"]["profit"],
            "propensity": m["propensity_eval"]["profit"],
        },
        "headline": {
            "champion_qini": m["champion_eval"]["qini"],
            "champion_qini_ci": m["champion_eval"]["qini_ci"],
            "propensity_qini": m["propensity_eval"]["qini"],
            "propensity_qini_ci": m["propensity_eval"]["qini_ci"],
            "champion_top_decile": m["champion_eval"]["top_decile_uplift"],
            "propensity_top_decile": m["propensity_eval"]["top_decile_uplift"],
            "champion_corr": m["champion_eval"]["ground_truth"]["corr"],
            "propensity_corr": m["propensity_eval"]["ground_truth"]["corr"],
        },
    }


@app.get("/api/bakeoff")
def bakeoff():
    m = STATE["metrics"]
    return {"bakeoff": m["bakeoff"], "offers": m["offers"], "champion": m["champion"]}


@app.get("/api/real-data")
def real_data():
    """Bake-off on the Hillstrom RCT. Absent until `python -m nextbest.realdata`."""
    p = MODEL_DIR / "hillstrom_metrics.json"
    if not p.exists():
        return JSONResponse({"available": False,
                             "hint": "run: python -m nextbest.realdata"})
    return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}


@app.get("/api/offer-matrix")
def offer_matrix():
    m = STATE["metrics"]
    return {
        "matrix": m["offer_matrix"],
        "offers": [{"key": k, "label": OFFER_BY_KEY[k].label} for k in OFFER_KEYS],
        "segments": m["segments"],
    }


@app.get("/api/model-health")
def model_health():
    m = STATE["metrics"]
    s: pd.DataFrame = STATE["scored"]
    hist, edges = np.histogram(s["typical_uplift"] * 100, bins=40)
    return {
        "importance": m["importance"],
        "arm_rates": m["arm_rates"],
        "offers": m["offers"],
        "bakeoff": m["bakeoff"],
        "champion_ground_truth": m["champion_eval"]["ground_truth"],
        "distribution": [
            {"x": round(float((edges[i] + edges[i + 1]) / 2), 3), "n": int(hist[i])}
            for i in range(len(hist))
        ],
        "generated_at": m["generated_at"],
        "booster": m["booster"],
        "n_customers": m["n_customers"],
    }


# ----------------------------------------------------------------- customers
@app.get("/api/customers")
def customers(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    quadrant: str | None = None,
    segment: str | None = None,
    category: str | None = None,
    sort: str = "expected_profit",
    q: str | None = None,
):
    s: pd.DataFrame = STATE["scored"]
    view = s

    if quadrant and quadrant != "all":
        view = view[view["quadrant"] == quadrant]
    if segment and segment != "all":
        view = view[view["segment"] == segment]
    if category and category != "all":
        view = view[view["category"] == category]
    if q:
        try:
            view = view[view["customer_id"] == int(q)]
        except ValueError:
            pass

    if sort not in view.columns:
        sort = "expected_profit"
    view = view.sort_values(sort, ascending=False)

    total = int(len(view))
    page = view.iloc[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": [
            {
                "customer_id": int(r.customer_id),
                "category": r.category,
                "segment": r.segment,
                "quadrant": r.quadrant,
                "quadrant_label": QUADRANT_LABELS[r.quadrant],
                "offer": r.offer,
                "offer_label": OFFER_BY_KEY[r.offer].label,
                "uplift_pp": round(float(r.offer_uplift) * 100, 2),
                "typical_uplift_pp": round(float(r.typical_uplift) * 100, 2),
                "p_control_pct": round(float(r.p_control) * 100, 1),
                "expected_profit": round(float(r.expected_profit), 2),
                "expected_cost": round(float(r.expected_cost), 2),
                "monetary_12m": round(float(r.monetary_12m), 0),
                "recency_days": round(float(r.recency_days), 0),
                "frequency_12m": int(r.frequency_12m),
            }
            for r in page.itertuples()
        ],
    }


@app.get("/api/customer/{customer_id}")
def customer(customer_id: int):
    s: pd.DataFrame = STATE["scored"]
    if customer_id not in s.index:
        raise HTTPException(404, f"customer {customer_id} not found")

    r = s.loc[customer_id]
    X: pd.DataFrame = STATE["features"]
    art = STATE["artifacts"]
    row = X.loc[customer_id]

    model = art["per_offer"][r["offer"]]
    drivers = explain.explain_row(model, row, art["reference"], top_k=7)

    per_offer = []
    for k in OFFER_KEYS:
        u = float(r[f"uplift_{k}"])
        cost = OFFER_BY_KEY[k].cost(float(r["avg_order_value"]))
        per_offer.append({
            "key": k,
            "label": OFFER_BY_KEY[k].label,
            "uplift_pp": round(u * 100, 2),
            "cost": round(cost, 2),
            "expected_profit": round(u * (float(r["margin"]) - cost) - CONTACT_COST, 2),
            "chosen": k == r["offer"],
        })

    p_c = float(r["p_control"])
    p_t = p_c + float(r["offer_uplift"])
    return {
        "customer_id": int(customer_id),
        "category": r["category"],
        "segment": r["segment"],
        "quadrant": r["quadrant"],
        "quadrant_label": QUADRANT_LABELS[r["quadrant"]],
        "profile": {
            "recency_days": round(float(r["recency_days"]), 0),
            "frequency_12m": int(r["frequency_12m"]),
            "monetary_12m": round(float(r["monetary_12m"]), 0),
            "avg_order_value": round(float(r["avg_order_value"]), 2),
            "tenure_years": round(float(r["tenure_years"]), 1),
            "engagement": round(float(r["engagement"]), 3),
            "discount_affinity": round(float(r["discount_affinity"]), 3),
            "price_tier": int(r["price_tier"]),
            "is_registered": bool(r["is_registered"]),
        },
        "counterfactual": {
            "p_control_pct": round(p_c * 100, 2),
            "p_treated_pct": round(p_t * 100, 2),
            "uplift_pp": round(float(r["offer_uplift"]) * 100, 2),
        },
        "recommendation": {
            "offer": r["offer"],
            "offer_label": OFFER_BY_KEY[r["offer"]].label,
            "expected_profit": round(float(r["expected_profit"]), 2),
            "expected_cost": round(float(r["expected_cost"]), 2),
            "action": "contact" if float(r["expected_profit"]) > 0 else "suppress",
        },
        "drivers": drivers,
        "offers": per_offer,
    }


# ------------------------------------------------------------------ audience
class AudienceRequest(BaseModel):
    budget: float = Field(25_000, ge=0)
    offers: list[str] | None = None
    segments: list[str] | None = None
    categorys: list[str] | None = None
    min_uplift_pp: float = 0.0
    preview: int = Field(25, ge=0, le=200)


@app.post("/api/audience")
def audience(req: AudienceRequest):
    s: pd.DataFrame = STATE["scored"]
    view = s

    if req.segments:
        view = view[view["segment"].isin(req.segments)]
    if req.categorys:
        view = view[view["category"].isin(req.categorys)]

    # restricting the offer set changes which offer wins per customer, so the
    # argmax has to be recomputed rather than filtered
    if req.offers and set(req.offers) != set(OFFER_KEYS):
        uplift = view[[f"uplift_{k}" for k in OFFER_KEYS]].copy()
        uplift.columns = list(OFFER_KEYS)
        choice = optimize.best_offer(uplift, view["avg_order_value"].to_numpy(),
                                     allowed=req.offers,
                                     p_control=view["p_control"].to_numpy())
        view = view.assign(
            offer=choice["offer"].to_numpy(),
            offer_uplift=choice["offer_uplift"].to_numpy(),
            expected_profit=choice["expected_profit"].to_numpy(),
            expected_cost=choice["expected_cost"].to_numpy(),
        )

    result = optimize.allocate(view, budget=req.budget,
                               min_uplift=req.min_uplift_pp / 100.0)

    ids = result.pop("selected_ids")
    chosen = view.loc[ids] if ids else view.iloc[0:0]

    quad_mix = chosen["quadrant"].value_counts().to_dict() if len(chosen) else {}
    offer_mix = chosen["offer"].value_counts().to_dict() if len(chosen) else {}

    preview = chosen.sort_values("expected_profit", ascending=False).head(req.preview)

    # what a naive "contact everyone in the filter" campaign would do
    naive_cost = float(view["expected_cost"].sum())
    naive_profit = float(view["expected_profit"].sum())

    return {
        **result,
        "pool_size": int(len(view)),
        "quadrant_mix": [
            {"key": k, "label": QUADRANT_LABELS[k], "count": int(quad_mix.get(k, 0))}
            for k in QUADRANT_LABELS
        ],
        "offer_mix": [
            {"key": k, "label": OFFER_BY_KEY[k].label, "count": int(offer_mix.get(k, 0))}
            for k in OFFER_KEYS
        ],
        "naive": {
            "n": int(len(view)),
            "spend": round(naive_cost, 2),
            "profit": round(naive_profit, 2),
        },
        "preview": [
            {
                "customer_id": int(r.customer_id),
                "segment": r.segment,
                "category": r.category,
                "quadrant": r.quadrant,
                "quadrant_label": QUADRANT_LABELS[r.quadrant],
                "offer": r.offer,
                "offer_label": OFFER_BY_KEY[r.offer].label,
                "uplift_pp": round(float(r.offer_uplift) * 100, 2),
                "expected_profit": round(float(r.expected_profit), 2),
            }
            for r in preview.itertuples()
        ],
    }


@app.get("/api/filters")
def filters():
    s: pd.DataFrame = STATE["scored"]
    return {
        "segments": sorted(s["segment"].unique().tolist()),
        "categorys": list(CATEGORIES),
        "quadrants": [{"key": k, "label": v} for k, v in QUADRANT_LABELS.items()],
        "offers": [{"key": k, "label": OFFER_BY_KEY[k].label,
                    "cost_rate": OFFER_BY_KEY[k].cost_rate,
                    "cost_flat": OFFER_BY_KEY[k].cost_flat} for k in OFFER_KEYS],
    }


# -------------------------------------------------------------------- static
STATIC = Path(__file__).resolve().parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
