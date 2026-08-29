"""NextBest — retail customer intelligence.

Serves the scored customer base built by simple/realbuild.py: who your customers are,
what they buy, and what each one should get.

    python simple/app.py
"""
from __future__ import annotations

import json
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Timer

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import realbuild as RB  # noqa: E402

# The real dataset defines its own fields, so the labels and the feature list
# come from the real build rather than from the simulator's config.
FEATURES = RB.FEATURES
CATEGORY_LABELS = RB.MERCH
CHANNEL_LABELS = RB.CHANNEL_LABELS
CATEGORIES = tuple(RB.MERCH)

APP_NAME = "NextBest"
APP_TAGLINE = "Retail Customer Intelligence"
STATIC = HERE / "static"

# Must match simple/realbuild.py, or the customer view and the headline disagree.
MARGIN_RATE = RB.MARGIN_RATE
# Must be the price the stored customer rows were built at, or a customer with
# positive uplift gets told 'no email' because the scorer is using a dearer cost.
CONTACT_COST = RB.CONTACT_COST

STATE: dict = {}


def _load() -> None:
    need = [HERE / "artifacts.joblib", HERE / "customers.pkl", HERE / "metrics.json"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise RuntimeError(f"missing {missing} -- run: python simple/realbuild.py")
    STATE["art"] = joblib.load(need[0])
    STATE["cust"] = pd.read_pickle(need[1])
    STATE["metrics"] = json.loads(need[2].read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    print(f"[app] {len(STATE['cust']):,} customers loaded", flush=True)
    yield


app = FastAPI(title=f"{APP_NAME} - {APP_TAGLINE}", lifespan=lifespan)


# ------------------------------------------------------------------ overview
@app.get("/api/overview")
def overview():
    return {"app": {"name": APP_NAME, "tagline": APP_TAGLINE}, **STATE["metrics"]}


# ----------------------------------------------------------------- customers
@app.get("/api/customers")
def customers(
    action: str | None = None,
    category: str | None = None,
    channel: str | None = None,
    reward: str | None = None,
    q: str | None = None,
    sort: str = "value_of_contact",
    limit: int = Query(40, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    df: pd.DataFrame = STATE["cust"]
    for col, val in (("action", action), ("category", category),
                     ("channel", channel), ("reward", reward)):
        if val and val != "all":
            df = df[df[col] == val]
    if q:
        try:
            df = df[df["customer_id"] == int(q)]
        except ValueError:
            df = df[df["spend_band"].str.contains(q, case=False, na=False)]

    if sort not in df.columns:
        sort = "value_of_contact"
    # months-since-purchase is the one field where smaller is "more recent"
    df = df.sort_values(sort, ascending=(sort == "months_since_purchase"))

    total = int(len(df))
    page = df.iloc[offset: offset + limit]
    return {
        "total": total, "offset": offset, "limit": limit,
        "priced_at": STATE["metrics"]["priced_at"]["customers"],
        "rows": [{
            "customer_id": int(r.customer_id),
            "buys": CATEGORY_LABELS.get(r.category, r.category),
            "spend_band": r.spend_band,
            "spend_12m": round(float(r.spend_12m), 2),
            "months_since_purchase": int(r.months_since_purchase),
            "new_customer": bool(r.new_customer),
            "area": r.area,
            "channel": CHANNEL_LABELS.get(r.channel, r.channel),
            "extra_sales_pp": round(float(r.extra_sales_pp), 1),
            "value_of_contact": round(float(r.value_of_contact), 3),
            "action": r.action,
            "reward": r.reward,
        } for r in page.itertuples()],
    }


@app.get("/api/customer/{customer_id}")
def customer(customer_id: int):
    df: pd.DataFrame = STATE["cust"]
    if customer_id not in df.index:
        raise HTTPException(404, f"customer {customer_id} not found")
    r = df.loc[customer_id]
    met = STATE["metrics"]
    act = met["actions"][r["action"]]
    rew = next((x for x in met["rewards"] if x["key"] == r["reward"]), None)
    return {
        "customer_id": int(customer_id),
        # Only fields this dataset genuinely contains. Nothing derived is shown
        # as though it were observed.
        "known": {
            "months_since_purchase": int(r["months_since_purchase"]),
            "spend_12m": round(float(r["spend_12m"]), 2),
            "spend_band": r["spend_band"],
            "buys": CATEGORY_LABELS.get(r["category"], r["category"]),
            "new_customer": bool(r["new_customer"]),
            "area": r["area"],
            "channel": CHANNEL_LABELS.get(r["channel"], r["channel"]),
        },
        # What actually happened to them in the experiment.
        "happened": {
            "arm": r["arm"],
            "visited": bool(r["visited"]),
            "spent": round(float(r["spent"]), 2),
        },
        "prediction": {
            "buys_alone_pct": round(float(r["buys_alone_pct"]), 1),
            "buys_if_contacted_pct": round(float(r["buys_if_contacted_pct"]), 1),
            "extra_sales_pp": round(float(r["extra_sales_pp"]), 1),
        },
        "decision": {
            "action": r["action"], "label": act["label"], "tone": act["tone"],
            "why": act["why"],
            "reward": rew["label"] if rew else None,
            "reward_why": rew["why"] if rew else None,
            "value": round(float(r["value_of_contact"]), 3),
            "cost": round(float(r["cost_of_contact"]), 2),
        },
    }


# ------------------------------------------------------------ what-if scoring
class WhatIf(BaseModel):
    """The real fields this dataset actually carries."""
    months_since_purchase: float = Field(6, ge=1, le=12)
    spent_last_year: float = Field(250, ge=0, le=3500)
    buys_mens: int = Field(1, ge=0, le=1)
    buys_womens: int = Field(0, ge=0, le=1)
    new_customer: int = Field(0, ge=0, le=1)
    zip_code: str = "Surburban"
    channel: str = "Web"

    @field_validator("zip_code")
    @classmethod
    def _zip(cls, v: str) -> str:
        if v not in ("Surburban", "Rural", "Urban"):
            raise ValueError("zip_code must be Surburban, Rural or Urban")
        return v

    @field_validator("channel")
    @classmethod
    def _chan(cls, v: str) -> str:
        if v not in ("Phone", "Web", "Multichannel"):
            raise ValueError("channel must be Phone, Web or Multichannel")
        return v


def _vector(c: WhatIf) -> np.ndarray:
    """Build the real 12-field model input the artifacts were trained on."""
    hist = float(c.spent_last_year)
    # history_band mirrors the source data's spend brackets (1 = lowest)
    band = 1 + sum(hist > x for x in (100, 200, 350, 500, 750, 1000))
    f = {
        "recency": float(c.months_since_purchase),
        "history_log": float(np.log1p(hist)),
        "mens": float(c.buys_mens),
        "womens": float(c.buys_womens),
        "newbie": float(c.new_customer),
        "history_band": float(band),
    }
    for z in ("Surburban", "Rural", "Urban"):
        f[f"zip_{z.lower()}"] = 1.0 if c.zip_code == z else 0.0
    for ch in ("Phone", "Web", "Multichannel"):
        f[f"chan_{ch.lower()}"] = 1.0 if c.channel == ch else 0.0
    return np.array([[f[k] for k in FEATURES]], dtype=float)


def _demand() -> float:
    """The build splits "no offer needed" from "not worth it" at the median chance
    of visiting unprompted. A hardcoded 0.15 here disagreed with every other view."""
    return float(np.median(STATE["art"]["base_pctrl"]))


@app.post("/api/whatif")
def whatif(c: WhatIf):
    art, met = STATE["art"], STATE["metrics"]
    x = _vector(c)
    best = met["best"]["key"]
    E = met["economics"]
    margin_per_visit = E["margin_per_visit"]

    others, propensity = [], None
    for row in met["models"]:
        val = float(art["models"][row["key"]].predict(x)[0])
        if not row["is_uplift"]:
            propensity = {"name": row["name"], "buy_chance_pct": round(val * 100, 1)}
            continue
        others.append({"name": row["name"], "extra_sales_pp": round(val * 100, 1),
                       "is_best": row["key"] == best})

    champ = art["models"][best]
    uplift = float(champ.predict(x)[0])
    alone = float(champ.predict_control(x)[0])
    treated = float(np.clip(alone + uplift, 0, 1))
    value = uplift * margin_per_visit - CONTACT_COST

    # Which of the two real emails suits this customer. Same rule as realbuild.py:
    # the offer follows what they buy unless the other arm wins by OFFER_GAP.
    up_m = float(art["per_offer"]["mens"].predict(x)[0])
    up_w = float(art["per_offer"]["womens"].predict(x)[0])
    rk = RB.choose_offer(bool(c.buys_mens), bool(c.buys_womens), up_m, up_w)

    if uplift < -0.002:
        key = "suppress"
    elif value > 0:
        key = "contact"
    elif alone >= _demand():
        key = "no_offer"
    else:
        key = "not_worth"
    act = met["actions"][key]
    reward = next((r for r in met["rewards"] if r["key"] == rk), None) if key == "contact" else None

    return {
        "action": key, "label": act["label"], "tone": act["tone"], "why": act["why"],
        "reward": reward["label"] if reward else None,
        "reward_why": reward["why"] if reward else None,
        "buys_alone_pct": round(alone * 100, 1),
        "buys_if_contacted_pct": round(treated * 100, 1),
        "extra_sales_pp": round(uplift * 100, 1),
        "money": {"margin": round(margin_per_visit, 2), "discount": 0.0,
                  "contact": CONTACT_COST, "value": round(value, 2)},
        "emails": {"mens_pp": round(up_m * 100, 1), "womens_pp": round(up_w * 100, 1)},
        "models": others, "propensity": propensity,
    }


# ------------------------------------------------------- live re-pricing
class Assumptions(BaseModel):
    """The only two numbers a marketer actually owns."""
    margin: float = Field(0.45, ge=0.05, le=0.90)
    cost: float = Field(0.30, ge=0.01, le=3.00)


def _price(pick, up, t, y, rev_inc, margin, cost):
    """What a chosen list is really worth at a given margin and contact cost.

    Uses the held-back group inside the chosen set, so the lift is measured, not
    predicted. Only the money is re-derived, which is why no retrain is needed.
    """
    n = int(pick.sum())
    if n == 0:
        return 0.0, 0.0, 0
    tt, yy = t[pick], y[pick]
    nt, nc = int(tt.sum()), int((1 - tt).sum())
    if nt == 0 or nc == 0:
        return 0.0, 0.0, n
    lift = float(yy[tt == 1].mean() - yy[tt == 0].mean())
    extra = lift * n
    return extra * rev_inc * margin - n * cost, extra, n


@app.post("/api/recompute")
def recompute(a: Assumptions):
    art, met = STATE["art"], STATE["metrics"]
    E = art["eval"]
    up, t, y = E["uplift"], E["t"], E["y"]
    rev_inc, n_base = E["rev_per_incremental_visit"], E["n_base"]
    scale = n_base / len(up)

    margin_per_visit = rev_inc * a.margin
    breakeven = E["lift_all"] * margin_per_visit

    everyone = np.ones(len(up), bool)
    blanket, _, _ = _price(everyone, up, t, y, rev_inc, a.margin, a.cost)
    pick = (up * margin_per_visit) > a.cost
    targeted, extra, n_sel = _price(pick, up, t, y, rev_inc, a.margin, a.cost)

    # One definition of the rule, shared with the build. app.js mirrors it for
    # the hosted build; tests/test_core.py pins both.
    tg_scaled = targeted * scale
    verdict, advice = RB.verdict_for(int(n_sel * scale), int(len(up) * scale),
                                     tg_scaled, blanket * scale)

    # the same decision across the whole price range, for the chart
    curve = []
    for c in [round(x * 0.02 + 0.02, 2) for x in range(50)]:
        b, _, _ = _price(everyone, up, t, y, rev_inc, a.margin, c)
        pk = (up * margin_per_visit) > c
        tg, _, ns = _price(pk, up, t, y, rev_inc, a.margin, c)
        curve.append({"cost": c, "blanket": round(b * scale, 2),
                      "targeted": round(tg * scale, 2), "n": int(ns * scale)})

    # what the whole base looks like at this price
    bu, bp = art["base_uplift"], art["base_pctrl"]
    value = bu * margin_per_visit - a.cost
    demand = float(np.median(bp))
    action = np.where(bu < -0.002, "suppress",
                      np.where(value > 0, "contact",
                               np.where(bp >= demand, "no_offer", "not_worth")))
    groups = [{"key": k, "label": met["actions"][k]["label"], "tone": met["actions"][k]["tone"],
               "n": int((action == k).sum()),
               "share": round(float((action == k).mean()) * 100, 1)}
              for k in met["actions"]]

    return {
        "margin": a.margin, "cost": a.cost,
        "breakeven": round(breakeven, 4),
        "margin_per_visit": round(margin_per_visit, 2),
        "blanket": round(blanket * scale, 2),
        "targeted": round(targeted * scale, 2),
        "gain": round((targeted - blanket) * scale, 2),
        "gain_is_avoided_loss": bool(tg_scaled <= 0),
        "n_targeted": int(n_sel * scale),
        "share_targeted": round(n_sel / len(up) * 100, 1),
        "extra_visits": round(extra * scale, 0),
        "verdict": verdict, "advice": advice,
        "curve": curve,
        "groups": groups,
        "n_base": n_base,
    }


# ------------------------------------------------------------------ export
@app.get("/api/export")
def export(action: str = "contact", cost: float = Query(0.30, ge=0.01, le=3.00),
           margin: float = Query(0.45, ge=0.05, le=0.90),
           category: str | None = None, channel: str | None = None,
           reward: str | None = None, q: str | None = None):
    """The list, as a file. Priced at whatever assumptions the caller is using,
    and the assumptions travel in the filename so a downloaded list is never
    ambiguous about what produced it."""
    art, met = STATE["art"], STATE["metrics"]
    df: pd.DataFrame = STATE["cust"].copy()
    E = art["eval"]
    margin_per_visit = E["rev_per_incremental_visit"] * margin

    bu, bp = art["base_uplift"], art["base_pctrl"]
    value = bu * margin_per_visit - cost
    demand = float(np.median(bp))
    df["action"] = np.where(bu < -0.002, "suppress",
                            np.where(value > 0, "contact",
                                     np.where(bp >= demand, "no_offer", "not_worth")))
    df["expected_value"] = np.round(value, 4)

    if action != "all":
        df = df[df["action"] == action]
    # The file has to be the list that was on screen, filters included.
    if category:
        df = df[df["category"] == category]
    if channel:
        df = df[df["channel"] == channel]
    if reward:
        df = df[df["reward"] == reward]
    if q and q.strip().isdigit():
        df = df[df["customer_id"] == int(q.strip())]

    out = df[["customer_id", "months_since_purchase", "spend_12m", "spend_band",
              "buys_mens", "buys_womens", "new_customer", "area", "channel",
              "extra_sales_pp", "expected_value", "action", "reward"]].rename(columns={
        "months_since_purchase": "months_since_last_purchase",
        "spend_12m": "spend_past_year",
        "extra_sales_pp": "predicted_extra_visits_pp",
        "reward": "send_which_email",
    })

    body = out.to_csv(index=False)
    tags = "".join(f"_{t}" for t in (category, channel, reward) if t)
    name = (f"nextbest_{action}{tags}_{len(out)}rows"
            f"_cost{cost:.2f}_margin{int(margin * 100)}pct.csv")
    return Response(
        content=body, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"',
                 "X-Row-Count": str(len(out)),
                 "X-Model": met["best"]["name"]},
    )


@app.get("/api/options")
def options():
    met = STATE["metrics"]
    return {
        "categories": [{"key": k, "label": v} for k, v in CATEGORY_LABELS.items()],
        "channels": [{"key": k, "label": v} for k, v in CHANNEL_LABELS.items()],
        "products": sorted({p["product"] for p in met["products"]}),
        "actions": [{"key": k, "label": v["label"]} for k, v in met["actions"].items()],
        "rewards": [{"key": r["key"], "label": r["label"]} for r in met["rewards"]],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    if not (HERE / "customers.pkl").exists():
        print("[app] no data yet -- building from real customer data")
        import realbuild

        realbuild.main()

    url = "http://127.0.0.1:8050/"
    print(f"\n  {APP_NAME} - {APP_TAGLINE} -> {url}\n  Ctrl+C to stop\n")
    Timer(1.3, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
