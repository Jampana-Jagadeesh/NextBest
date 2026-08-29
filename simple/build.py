#!/usr/bin/env python
"""Builds everything the app serves.

Trains five models on a randomised retail campaign, scores the whole customer
base, and works out what to actually do about each customer.

    python simple/build.py

Writes artifacts.joblib (the models), customers.pkl (the scored base) and
metrics.json (headline numbers plus all the copy). Takes about 25 seconds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from nextbest import evaluate, features, simulate  # noqa: E402
from nextbest.config import (CATEGORY_LABELS, CHANNEL_LABELS, FEATURES,  # noqa: E402
                             OFFER_BY_KEY)
from nextbest.learners import LEARNERS  # noqa: E402

N_ROWS = 20_000
ARM = "pct10"

# Retail economics, stated openly because the answer depends entirely on them.
MARGIN_RATE = 0.45       # gross margin on an order
OFFER_COST_RATE = 0.05   # a 5% discount, redeemed by anyone who buys
CONTACT_COST = 0.45      # cost of reaching one customer

ARTIFACTS = HERE / "artifacts.joblib"
CUSTOMERS = HERE / "customers.pkl"
METRICS = HERE / "metrics.json"

# --------------------------------------------------------------------- copy
PROBLEM = {
    "headline": "Most retail marketing money is spent on people who were going to buy anyway.",
    "body": "Industry studies put 70–90% of trade promotion spend in developed markets in the "
            "“value destroying” bucket once it is measured properly, and 60–80% of retail "
            "promotions never turn ROI positive. The reason is simple: a campaign report shows "
            "revenue, not <i>extra</i> revenue. A discount handed to a customer who had already "
            "decided to buy still lands in the results as a win.",
    "why_hard": "You only ever see what a customer did after you contacted them. You never see "
                "what that same customer would have done if you had left them alone — so the "
                "waste is invisible in every normal report.",
    "solution": "The five models below estimate that missing half of the picture. Each customer "
                "gets one number: how much being contacted actually <b>changes</b> their chance "
                "of buying. That turns a marketing list into four clear groups and tells you what "
                "each group should get.",
}

ACTIONS = {
    "contact": {"label": "Contact them", "tone": "good",
                "why": "Marketing genuinely changes what these customers do. Every sale here is "
                       "one you would not otherwise have had."},
    "no_offer": {"label": "No offer needed", "tone": "neutral",
                 "why": "They were buying regardless. A discount here is money given away on a "
                        "sale you already had — send them content, not a coupon."},
    "not_worth": {"label": "Not worth contacting", "tone": "muted",
                  "why": "They do not respond either way, so the cost of reaching them buys "
                         "nothing."},
    "suppress": {"label": "Leave them alone", "tone": "bad",
                 "why": "They were going to buy, and being marketed to puts them off. Contacting "
                        "these customers costs you the sale."},
}

REWARDS = {
    "discount": {"label": "15% discount",
                 "why": "They buy on promotion and respond to price."},
    "delivery": {"label": "Free delivery",
                 "why": "Price matters, but not enough to need a deep cut."},
    "points": {"label": "Bonus points & early access",
               "why": "They respond to service and status, not discounts. A price cut here would "
                      "only reduce your margin."},
}

# What every uplift model here is predicting -- said once, because all four
# predict the same number and differ only in how they get to it.
PREDICTS = ("For every customer, one number: how much more likely they are to buy "
            "<b>because we contacted them</b>.")

MODEL_INFO = {
    "propensity": {
        "answer": "Ranks customers by how likely they are to buy.",
        "predicts": "How likely a customer is to buy at all. <b>Not the same number.</b>",
        "how": "An ordinary sales-prediction model, run on customers who were contacted.",
        "good": "Fine for forecasting demand and planning stock.",
        "careful": "It cannot tell persuasion from coincidence. Its top customers are your best "
                   "regulars, who were buying anyway.",
        "business": "A chain discounts the 400,000 customers most likely to buy. Revenue looks "
                    "huge, but those people were already coming — about $3.1M of discount went "
                    "to orders it already had.",
        "verdict": "avoid", "verdict_text": "Predicts the wrong thing",
    },
    "s_learner": {
        "answer": "Asks one model twice — contacted, and not contacted.",
        "predicts": PREDICTS,
        "how": "One model, with ‘were they contacted?’ added as one more field. Score the "
               "customer both ways, take the difference.",
        "good": "Cheapest to build, fastest to train, easiest to explain to a non-technical team.",
        "careful": "The model can ignore the contacted field and quietly flatten every score to "
                   "zero. Check the scores actually vary.",
        "business": "A mid-size retailer with one analyst and two weeks. It trains in a second and "
                    "fits on one slide — which is why it ships.",
        "verdict": "good", "verdict_text": "Simple and strong",
    },
    "t_learner": {
        "answer": "Builds two models and subtracts one from the other.",
        "predicts": PREDICTS,
        "how": "One model on contacted customers, one on the held-back group. The gap between "
               "them is the answer.",
        "good": "The most obvious approach, and easy to defend in a review.",
        "careful": "Each model sees only half the data, so their mistakes add up. Noisiest of "
                   "the four.",
        "business": "A grocery chain running a clean 50/50 test on 400,000 customers, where both "
                    "groups are big and equal.",
        "verdict": "ok", "verdict_text": "Simple, but noisy",
    },
    "x_learner": {
        "answer": "Fills in the half of each customer's story you never saw.",
        "predicts": PREDICTS,
        "how": "Estimates what each contacted customer would have done if left alone, then learns "
               "from those estimated gaps.",
        "good": "Holds up when the held-back group is much smaller than the contacted group.",
        "careful": "Five moving parts instead of two — more to go wrong, slower to train.",
        "business": "A chain that will only hold back 3% of customers, because merchandising "
                    "refuses to give up more sales to a test.",
        "verdict": "good", "verdict_text": "Best on real campaigns",
    },
    "transformed": {
        "answer": "Rewrites the maths so one model predicts the change directly.",
        "predicts": PREDICTS,
        "how": "A statistical trick reshapes the outcome, so a single plain model predicts the "
               "effect with no subtraction step.",
        "good": "Fastest to score a very large base.",
        "careful": "The maths divides by the chance of being contacted, so small numbers make it "
                   "jump around. Least stable.",
        "business": "A marketplace rescoring 40 million customers nightly, where finishing before "
                    "the 6am send matters more than the last decimal.",
        "verdict": "ok", "verdict_text": "Fastest, least stable",
    },
}

GLOSSARY = [
    ("Extra sales",
     "How much more likely a customer is to buy because you contacted them. If they would buy 20% "
     "of the time on their own and 32% of the time after an email, the extra is 12 points. It can "
     "be negative, which means contacting them makes things worse."),
    ("Held-back group",
     "A slice of customers deliberately not contacted, so you have something to compare against. "
     "Without one there is nothing to measure extra sales against — no held-back group, no answer."),
    ("Accuracy",
     "How closely a model's scores track each customer's real effect, from -1 to +1. Above 0.5 is "
     "strong; near 0 means the model is not tracking the effect at all. It can only be measured "
     "here because this data is simulated and the true answer is known."),
    ("Top 10% / Bottom 10%",
     "Sort every customer by a model's score, then look at the best and worst tenth. A model that "
     "works shows a big positive at the top and a negative at the bottom."),
    ("Qini",
     "One score for how much better a model's ranking is than contacting people at random. Higher "
     "is better; zero means no better than a shuffled list. Shown with a range because at this "
     "sample size it is noisy."),
]


def _log(m):
    print(f"[build] {m}", flush=True)


def main() -> dict:
    t0_all = time.time()
    _log(f"simulating {N_ROWS:,} customers on a 50/50 test")
    raw = simulate.generate(n=N_ROWS, arms=[ARM])
    X = features.build(raw)
    features.assert_no_leakage(X)

    tau = raw[f"tau_{ARM}"].to_numpy()
    y = raw["converted"].to_numpy()
    t = raw["treated"].to_numpy()
    tr, te = train_test_split(np.arange(len(raw)), test_size=0.30,
                              random_state=1, stratify=raw["arm"])

    # ------------------------------------------------------------- bake-off
    fitted, rows = {}, []
    for key, cls in LEARNERS.items():
        t0 = time.time()
        m = cls().fit(X.iloc[tr], t[tr], y[tr])
        pred = m.predict(X.iloc[te])
        secs = time.time() - t0
        res = evaluate.evaluate_all(pred, t[te], y[te], tau[te])
        fitted[key] = m
        rows.append({
            "key": key, "name": cls.name, "is_uplift": key != "propensity",
            "corr": res["ground_truth"]["corr"], "qini": res["qini"], "ci": res["qini_ci"],
            "top": res["top_decile_uplift"], "bot": res["bottom_decile_uplift"],
            "deciles": [d["uplift"] for d in res["deciles"]], "secs": round(secs, 2),
            **MODEL_INFO[key],
        })
        _log(f"  {cls.name:22s} accuracy={res['ground_truth']['corr']:+.3f}")

    # ------------------------------------------- what each model would have you DO
    # A model is only worth what its decisions earn. So: let each model pick who to
    # contact, then price that choice against the TRUE effect -- which is knowable
    # here only because the data is simulated. A model that overestimates uplift
    # would otherwise flatter itself by scoring its own optimistic guesses.
    true_up = raw[f"tau_{ARM}"].to_numpy()
    true_ctrl = raw["p_control_true"].to_numpy()
    true_treat = np.clip(true_ctrl + true_up, 0, 1)
    _aov = raw["avg_order_value"].to_numpy()
    _margin, _offer = _aov * MARGIN_RATE, _aov * OFFER_COST_RATE
    true_profit = true_up * _margin - true_treat * _offer - CONTACT_COST

    # the best any model could possibly do, and what doing nothing earns
    oracle = float(true_profit[true_profit > 0].sum())
    blanket = float(true_profit.sum())

    for r in rows:
        m = fitted[r["key"]]
        up = m.predict(X)
        pc = m.predict_control(X)
        pt = np.clip(pc + up, 0, 1)
        pick = (up * _margin - pt * _offer - CONTACT_COST) > 0   # its own advice
        earned = float(true_profit[pick].sum())                  # what that advice really earns
        r["decision"] = {
            "contact_n": int(pick.sum()),
            "contact_share": round(float(pick.mean()) * 100, 1),
            "earns": round(earned, 2),
            "of_possible": round(earned / oracle * 100, 1) if oracle > 0 else 0.0,
            "vs_best": 0.0,
            "top_category": (cust_cat := raw["category"].to_numpy()[pick]),
        }
        # which categories its list leans on, in plain terms
        import collections
        cnt = collections.Counter(cust_cat.tolist())
        r["decision"]["top_category"] = (CATEGORY_LABELS.get(cnt.most_common(1)[0][0],
                                                             cnt.most_common(1)[0][0])
                                         if cnt else "—")

    best = max((r for r in rows if r["is_uplift"]), key=lambda r: r["corr"])
    for r in rows:
        r["decision"]["vs_best"] = round(r["decision"]["earns"] - best["decision"]["earns"], 2)
    for r in rows:
        if r["key"] == best["key"]:
            r["verdict"], r["verdict_text"] = "best", "Recommended"
    rows.sort(key=lambda r: -r["corr"])
    _log(f"recommended model: {best['name']}")

    # ------------------------------------------------- score the whole base
    champ = fitted[best["key"]]
    uplift = champ.predict(X)
    p_ctrl = champ.predict_control(X)
    p_treat = np.clip(p_ctrl + uplift, 0, 1)

    aov = raw["avg_order_value"].to_numpy()
    margin = aov * MARGIN_RATE
    offer_cost = aov * OFFER_COST_RATE
    # The discount is redeemed by anyone who buys after being contacted, not only
    # by the extra sales it caused. That asymmetry is the whole cost of
    # discounting customers who were already going to purchase.
    spend = CONTACT_COST + p_treat * offer_cost
    profit = uplift * margin - p_treat * offer_cost - CONTACT_COST

    demand_line = float(np.median(p_ctrl))
    action = np.where(
        (uplift < -0.005) & (p_ctrl >= 0.25), "suppress",
        np.where(profit > 0, "contact",
                 np.where(p_ctrl >= demand_line, "no_offer", "not_worth")))

    aff = raw["discount_affinity"].to_numpy()
    reward = np.where(action != "contact", "",
                      np.where(aff > 0.60, "discount",
                               np.where(aff > 0.35, "delivery", "points")))

    cust = pd.DataFrame({
        "customer_id": raw["customer_id"].to_numpy(),
        "category": raw["category"].to_numpy(),
        "product": raw["product"].to_numpy(),
        "channel": raw["channel"].to_numpy(),
        "orders_12m": raw["frequency_12m"].to_numpy(),
        "items_per_order": raw["items_per_order"].to_numpy(),
        "spend_12m": raw["monetary_12m"].to_numpy(),
        "avg_order": aov,
        "last_seen_days": raw["recency_days"].to_numpy(),
        "years_a_customer": raw["tenure_years"].to_numpy(),
        "registered": raw["is_registered"].to_numpy().astype(int),
        "buys_on_promo": raw["discount_affinity"].to_numpy(),
        "buys_alone_pct": p_ctrl * 100,
        "buys_if_contacted_pct": p_treat * 100,
        "extra_sales_pp": uplift * 100,
        "value_of_contact": profit,
        "cost_of_contact": spend,
        "action": action,
        "reward": reward,
    }).set_index("customer_id", drop=False)
    cust.to_pickle(CUSTOMERS)

    # --------------------------------------------------------- the headline
    n = len(cust)
    grp = {k: cust[cust["action"] == k] for k in ACTIONS}
    total_spend = float(spend.sum())
    good_spend = float(grp["contact"]["cost_of_contact"].sum())
    wasted = total_spend - good_spend
    profit_all = float(profit.sum())
    profit_targeted = float(grp["contact"]["value_of_contact"].sum())

    answer = {
        "waste_pct": round(wasted / total_spend * 100, 1),
        "wasted_spend": round(wasted, 2),
        "total_spend": round(total_spend, 2),
        "profit_blanket": round(profit_all, 2),
        "profit_targeted": round(profit_targeted, 2),
        "gain": round(profit_targeted - profit_all, 2),
        "n": n,
        "groups": [{
            "key": k, "label": ACTIONS[k]["label"], "tone": ACTIONS[k]["tone"],
            "why": ACTIONS[k]["why"], "n": int(len(g)),
            "share": round(len(g) / n * 100, 1),
            "spend": round(float(g["cost_of_contact"].sum()), 2),
            "value": round(float(g["value_of_contact"].sum()), 2),
        } for k, g in grp.items()],
        "assumptions": {
            "margin_rate": MARGIN_RATE, "offer_rate": OFFER_COST_RATE,
            "contact_cost": CONTACT_COST,
            "avg_order_value": round(float(aov.mean()), 2),
            "breakeven_pp": round((demand_line * OFFER_COST_RATE * float(np.median(aov))
                                   + CONTACT_COST) / (MARGIN_RATE * float(np.median(aov))) * 100, 1),
        },
        "scaled": {"base": 1_000_000,
                   "wasted": round(wasted / n * 1_000_000),
                   "gain": round((profit_targeted - profit_all) / n * 1_000_000)},
    }
    _log(f"answer: {answer['waste_pct']}% of spend wasted, "
         f"{len(grp['contact']):,} customers worth contacting")

    # ---------------------------------------------------------- aggregates
    def mix(col, labels):
        out = []
        for k, g in cust.groupby(col):
            out.append({
                "key": k, "label": labels.get(k, k), "n": int(len(g)),
                "share": round(len(g) / n * 100, 1),
                "spend": round(float(g["spend_12m"].sum())),
                "avg_order": round(float(g["avg_order"].mean()), 2),
                "contact_n": int((g["action"] == "contact").sum()),
                "extra_sales_pp": round(float(g["extra_sales_pp"].mean()), 2),
            })
        return sorted(out, key=lambda r: -r["n"])

    rewards = []
    for k, meta in REWARDS.items():
        g = cust[cust["reward"] == k]
        rewards.append({"key": k, "label": meta["label"], "why": meta["why"],
                        "n": int(len(g)),
                        "value": round(float(g["value_of_contact"].sum()), 2),
                        "extra_sales": round(float(g["extra_sales_pp"].sum() / 100), 1)})

    # ---------------------------------------------------------------- products
    # For each product: how it sells, and -- using the winning model -- whether
    # its buyers can be moved at all, which reward moves them, and what targeting
    # them is worth. That is what turns "this product sells badly" into an action.
    # Compared against other CONTACTABLE customers, not the whole base. The
    # contactable group is lapsed and promo-led by definition, so measuring it
    # against everyone just re-describes the selection rule for every product.
    _ref = cust[cust["action"] == "contact"]
    BASE = {"seen": float(_ref["last_seen_days"].mean()),
            "orders": float(_ref["orders_12m"].mean()),
            "promo": float(_ref["buys_on_promo"].mean())}

    products = []
    for k, g in cust.groupby("product"):
        worth = g[g["action"] == "contact"]
        promo = float(g["buys_on_promo"].mean())
        # The reward is whatever the model actually assigned to most of this
        # product's contactable buyers, not a second guess from the average --
        # averaging discount affinity across a product flattens the differences.
        rk = (worth["reward"].mode().iloc[0] if len(worth) and not worth["reward"].mode().empty
              else ("discount" if promo > 0.55 else "delivery" if promo > 0.38 else "points"))
        # Who to send that reward to: the business team needs a description of the
        # customer, not a customer id. Built from the contactable buyers only.
        aud = worth if len(worth) >= 20 else g
        seen, ords, pr = (float(aud["last_seen_days"].mean()),
                          float(aud["orders_12m"].mean()),
                          float(aud["buys_on_promo"].mean()))
        # Described relative to the rest of the base, not against fixed cut-offs.
        # Absolute thresholds labelled every product identically, because the
        # contactable population is fairly uniform -- the differences are only
        # visible as "more lapsed than average", "keener on promotions", and so on.
        when = ("Longest-lapsed" if seen > BASE["seen"] * 1.08 else
                "Recently active" if seen < BASE["seen"] * 0.92 else "Mid-lapse")
        kind = ("deal-driven" if pr > BASE["promo"] * 1.05 else
                "price-resistant" if pr < BASE["promo"] * 0.95 else "middle-of-road")
        freq = ("high-frequency " if ords > BASE["orders"] * 1.10 else
                "low-frequency " if ords < BASE["orders"] * 0.90 else "")
        chan = aud["channel"].mode().iloc[0] if len(aud) else "online"

        products.append({
            "audience": f"{when} {freq}{kind}".replace("  ", " "),
            "audience_detail": (f"last bought {seen:.0f} days ago, {ords:.1f} orders a year, "
                                f"buy on promotion {pr * 100:.0f}% of the time, mostly "
                                f"{CHANNEL_LABELS.get(chan, chan).lower()}"),
            "audience_seen": round(seen),
            "audience_orders": round(ords, 1),
            "audience_promo": round(pr * 100),
            "audience_channel": CHANNEL_LABELS.get(chan, chan),
            "product": k,
            "category": CATEGORY_LABELS.get(g["category"].iloc[0], g["category"].iloc[0]),
            "n": int(len(g)),
            "spend": round(float(g["spend_12m"].sum())),
            "avg_order": round(float(g["avg_order"].mean()), 2),
            "orders": round(float(g["orders_12m"].mean()), 1),
            "contact_n": int(len(worth)),
            "contact_share": round(len(worth) / max(len(g), 1) * 100, 1),
            # how movable this product's buyers are, on average
            "extra_sales_pp": round(float(g["extra_sales_pp"].mean()), 2),
            "buys_on_promo_pct": round(promo * 100),
            # the reward that suits this product's audience, and what it is worth
            "reward": rk,
            "reward_label": REWARDS[rk]["label"],
            "reward_why": REWARDS[rk]["why"],
            "gain": round(float(worth["value_of_contact"].sum()), 2),
            "extra_sales": round(float(worth["extra_sales_pp"].sum() / 100), 1),
        })
    products.sort(key=lambda r: -r["spend"])

    # Ranking purely by total sales just re-discovers that cheap categories total
    # less, so each product also carries how much of its sales marketing can move.
    for r in products:
        r["gain_per_100"] = round(r["gain"] / max(r["n"], 1) * 100, 2)
        r["headroom_pct"] = round(r["gain"] / max(r["spend"], 1) * 100, 3)

    weak = sorted(products, key=lambda r: r["spend"])[:5]
    for w in weak:
        # A product nobody responds to is a range or pricing problem, not a
        # marketing one -- saying so is more useful than recommending an offer.
        w["fixable"] = w["extra_sales_pp"] > 1.0 and w["contact_n"] >= 20

    opportunity = sorted(products, key=lambda r: -r["gain_per_100"])[:5]
    product_focus = {
        "best": products[0],
        "worst": weak[0],
        "top5": products[:5],
        "weak5": weak,
        "opportunity": opportunity,
        "fixable_n": sum(1 for w in weak if w["fixable"]),
        "note": ("Lowest total sales tends to follow basket size, not customer interest -- "
                 "cheaper categories will always total less. The opportunity ranking below "
                 "corrects for that by asking how much of each product's sales marketing can "
                 "actually move."),
    }

    metrics = {
        "app": {"name": "NextBest", "tagline": "Retail Customer Intelligence"},
        "problem": PROBLEM,
        "answer": answer,
        "models": rows,
        "oracle": {"best_possible": round(oracle, 2), "blanket": round(blanket, 2)},
        "best": {"key": best["key"], "name": best["name"]},
        "predicts": PREDICTS,
        "actions": ACTIONS,
        "rewards": rewards,
        "categories": mix("category", CATEGORY_LABELS),
        "channels": mix("channel", CHANNEL_LABELS),
        "products": products,
        "product_focus": product_focus,
        "glossary": GLOSSARY,
        "base": {
            "n": n,
            "revenue": round(float(cust["spend_12m"].sum())),
            "avg_order": round(float(cust["avg_order"].mean()), 2),
            "avg_orders": round(float(cust["orders_12m"].mean()), 1),
            "n_test": int(len(te)),
            "control_rate": round(float(raw.loc[raw["arm"] == "control", "converted"].mean()) * 100, 1),
            "treated_rate": round(float(raw.loc[raw["arm"] != "control", "converted"].mean()) * 100, 1),
            "offer_tested": OFFER_BY_KEY[ARM].label,
        },
        "generated_at": time.strftime("%d %b %Y, %H:%M"),
        "seconds": round(time.time() - t0_all, 1),
    }

    joblib.dump({"models": fitted, "features": list(FEATURES)}, ARTIFACTS)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log(f"wrote artifacts, customers.pkl and metrics.json in {metrics['seconds']}s")
    return metrics


if __name__ == "__main__":
    main()
