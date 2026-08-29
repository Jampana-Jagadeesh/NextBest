#!/usr/bin/env python
"""Builds the app from REAL customer data.

Source: Hillstrom MineThatData, 2008 — 64,000 customers of a real retailer,
randomly split into three groups: men's email, women's email, no email. Because
that split was random, the difference between the groups is a genuine causal
effect, not a correlation. That is what makes any of this measurable.

    python simple/realbuild.py

Nothing here is simulated. Every customer, purchase, email and outcome is real.
The trade-off is that no per-customer "true answer" exists, so models are ranked
by what their decisions actually earned on held-out customers, which is the
measure a business cares about anyway.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from nextbest import evaluate  # noqa: E402
from nextbest.learners import LEARNERS  # noqa: E402

CSV = ROOT / "data" / "hillstrom.csv"
URL = ("http://www.minethatdata.com/"
       "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

ARTIFACTS = HERE / "artifacts.joblib"
CUSTOMERS = HERE / "customers.pkl"
METRICS = HERE / "metrics.json"

# ---------------------------------------------------------------- economics
# All three are observable in the data itself, except the two marked assumed.
MARGIN_RATE = 0.45        # assumed: gross margin on apparel retail
CONTACT_COST = 0.08       # observed-scale cost of one plain email (list + send + creative)
# Contact methods a retailer actually chooses between, priced per customer.
CONTACT_OPTIONS = [
    {"key": "email", "label": "Plain email", "cost": 0.08,
     "note": "list, send and creative, spread over the base"},
    {"key": "email_promo", "label": "Email + 10% off", "cost": 0.30,
     "note": "the send, plus the discount redeemed by everyone who buys"},
    {"key": "sms", "label": "SMS", "cost": 0.45,
     "note": "per-message carrier cost"},
    {"key": "mail", "label": "Direct mail", "cost": 0.85,
     "note": "print, postage and fulfilment"},
]
RANK_AT = 0.30            # models are compared at a cost where the choice matters
OUTCOME = "visit"         # modelled outcome -- see note in main()

FEATURES = ("recency", "history_log", "mens", "womens", "newbie", "history_band",
            "zip_surburban", "zip_rural", "zip_urban",
            "chan_phone", "chan_web", "chan_multichannel")

MERCH = {"mens": "Men's merchandise", "womens": "Women's merchandise", "both": "Both"}

# How much better the mismatched campaign has to look before it overrides what the
# customer actually buys. app.py's single-customer scorer uses the same constant.
OFFER_GAP = 0.02

# Below this share of the base a "campaign" is a rounding error, not a campaign.
MIN_SHARE = 0.005


def choose_offer(buys_mens: bool, buys_womens: bool,
                 up_mens: float, up_womens: float) -> str:
    """Which of the two real campaigns to send.

    The men's arm lifted more overall (7.66pp vs 4.52pp), so picking the argmax
    per customer handed the men's email to 70% of womenswear buyers. That is
    defensible statistically and indefensible commercially: nobody can explain
    to a merchandising team why women's-only buyers get the men's campaign.

    So the offer follows what the customer actually buys, and the other arm has
    to beat it by more than OFFER_GAP before it overrides that. Customers who
    buy both have no natural match, so they take the argmax.
    """
    if buys_mens and buys_womens:
        return "mens" if up_mens >= up_womens else "womens"
    buys = "mens" if buys_mens else "womens"
    other = "womens" if buys == "mens" else "mens"
    up_buys, up_other = (up_mens, up_womens) if buys == "mens" else (up_womens, up_mens)
    return other if up_other > up_buys + OFFER_GAP else buys


def verdict_for(n_sel: int, n_base: int, targeted: float,
                blanket: float) -> tuple[str, str]:
    """Should this campaign run, and to whom?

    Previously any list that beat blanket sending by 2% came back as "target",
    which recommended campaigns that still lost money and a direct-mail list of
    34 people. Beating a worse option is not the same as being worth doing.
    """
    if n_sel == 0:
        return "stop", "Do not run it. Nobody is worth this cost."
    if targeted <= 0:
        return "stop", "Do not run it at this price. Even the best list still loses money."
    if n_sel < n_base * MIN_SHARE:
        return "stop", (f"Only {n_sel:,} customers clear this cost "
                        f"-- too few to run as a campaign.")
    if targeted - blanket > abs(blanket) * 0.02 and n_sel < n_base * 0.95:
        return "target", f"Target {n_sel / n_base * 100:.0f}% of the base."
    return "blanket", "Contact everyone. A model adds nothing here."
CHANNEL_LABELS = {"Phone": "Phone", "Web": "Web", "Multichannel": "Phone + Web"}

ACTIONS = {
    "contact": {"label": "Email them", "tone": "good",
                "why": "The email genuinely changes what these customers do. Every visit here is "
                       "one you would not otherwise have had."},
    "no_offer": {"label": "No email needed", "tone": "neutral",
                 "why": "They were coming back anyway. Emailing them adds cost without adding "
                        "visits."},
    "not_worth": {"label": "Not worth emailing", "tone": "muted",
                  "why": "They do not respond either way, so the send cost buys nothing."},
    "suppress": {"label": "Leave them alone", "tone": "bad",
                 "why": "These customers respond negatively. Emailing them reduces the chance "
                        "they come back."},
}

PROJECT = {
    "kicker": "AI project · 5 machine learning models",

    "problem_title": "Marketing money goes to people who were coming back anyway.",
    "problem_body": "Campaign reports show revenue, not <b>extra</b> revenue. You cannot tell a "
                    "sale you caused from one that was already coming.",
    "problem_cost": "Measured on this campaign: it paid — but only because email is nearly "
                    "free. Raise the cost of reaching someone and it stops paying. "
                    "The <b>exact price where it flips</b> is below.",

    "solution_title": "Decide who to contact, and who to leave alone.",
    "solution_body": "Five AI models score every customer on whether contact actually "
                     "<b>changes</b> what they do. The one that earns most runs the next two tabs.",
    "deliverables": [
        ["Send list", "worth contacting"],
        ["Suppression list", "buy less if contacted"],
        ["Offer per customer", "which of the two"],
        ["Price ceiling", "when to stop"],
    ],
}

PROBLEM = {
    "headline": "You can see what a campaign earned. You cannot see what it caused.",
    "body": "Without a held-back group, the emailed group's return rate is just a number with "
            "nothing to compare it against — and it gets reported as a success either way.",
    "why_hard": "Most retailers email everyone, so there is no held-back group and no way to tell "
                "a visit you caused from one that was coming anyway. This dataset has one, which "
                "is why real numbers can be put on the page.",
    "solution": "The models below score each real customer on how much the email changed their "
                "behaviour, then decide who to email and which of the two emails to send.",
    "what_we_do": "",
    "what_you_get": [],
}

REWARD_META = {
    "mens": {"label": "Men's email", "why": "Responds more to the men's merchandise campaign."},
    "womens": {"label": "Women's email", "why": "Responds more to the women's merchandise campaign."},
}

MODEL_INFO = {
    "propensity": {
        "answer": "Ranks customers by how likely they are to visit.",
        "predicts": "How likely a customer is to visit at all. <b>Not the same number.</b>",
        "how": "An ordinary response model, trained on the customers who were emailed.",
        "good": "Fine for forecasting traffic and planning stock.",
        "careful": "It cannot tell persuasion from coincidence. Its top customers are your most "
                   "recent buyers, who were coming back anyway.",
        "business": "The model most retail teams already run. It puts nearly everyone on the list, "
                    "because nearly everyone has some chance of visiting.",
        "verdict": "avoid", "verdict_text": "Predicts the wrong thing",
    },
    "s_learner": {
        "answer": "Asks one model twice — emailed, and not emailed.",
        "predicts": "",
        "how": "One model, with 'were they emailed?' added as one more field. Score the customer "
               "both ways, take the difference.",
        "good": "Cheapest to build and easiest to explain to a non-technical team.",
        "careful": "It can ignore the emailed field and flatten every score to zero. Check the "
                   "scores actually vary.",
        "business": "A team with one analyst and the customer data they already have.",
        "verdict": "good", "verdict_text": "Simple and strong",
    },
    "t_learner": {
        "answer": "Builds two models and subtracts one from the other.",
        "predicts": "",
        "how": "One model on emailed customers, one on the held-back group. The gap is the answer.",
        "good": "The most obvious approach, and easy to defend in a review.",
        "careful": "Each model sees only part of the data, so their mistakes add up.",
        "business": "Works here because the holdout is a full third of the base — larger than most "
                    "retailers would ever allow.",
        "verdict": "ok", "verdict_text": "Simple, but noisy",
    },
    "x_learner": {
        "answer": "Fills in the half of each customer's story you never saw.",
        "predicts": "",
        "how": "Estimates what each emailed customer would have done if left alone, then learns "
               "from those estimated gaps.",
        "good": "Holds up when the held-back group is much smaller than the emailed group.",
        "careful": "Five moving parts instead of two — more to go wrong, slower to train.",
        "business": "The one to reach for once the holdout shrinks to a realistic 5%, which is all "
                    "most merchandising teams will agree to.",
        "verdict": "good", "verdict_text": "Best when the holdout is small",
    },
    "transformed": {
        "answer": "Rewrites the maths so one model predicts the change directly.",
        "predicts": "",
        "how": "A statistical trick reshapes the outcome so a single plain model predicts the "
               "effect, with no subtraction step.",
        "good": "Fastest to score a very large base.",
        "careful": "The maths divides by the chance of being emailed, so small numbers make it "
                   "jump around.",
        "business": "For scoring millions of customers before a morning send.",
        "verdict": "ok", "verdict_text": "Fastest, least stable",
    },
}


# --------------------------------------------------------------- ML reference
# Written for the data team: definitions, notation and the assumptions the
# estimates rest on. Nothing here is needed to read the first three tabs.
REFERENCE = {
    # Every symbol used anywhere on this tab, in plain words. Without this the
    # page is only readable by someone who already knows the material.
    "notation": [
        ["Y", "what the customer actually did", "1 if they came back, 0 if not"],
        ["Y(1)", "what they would do if contacted", "one of two possible worlds"],
        ["Y(0)", "what they would do if left alone", "the other one — you never see both"],
        ["T", "were they contacted?", "1 = emailed, 0 = held back"],
        ["X", "everything we know about them", "recency, spend, what they buy, channel…"],
        ["x", "one particular customer's details", "a specific set of values for X"],
        ["τ (tau)", "the effect — what contact changed",
         "the number every model is trying to predict"],
        ["E[ … ]", "the average of …",
         "read “E[A | B]” as “the average A, among customers like B”"],
        ["μ (mu)", "a model of the outcome",
         "predicts Y. μ₁ is fitted on contacted customers, μ₀ on held-back"],
        ["̂ (hat)", "an estimate, not the truth",
         "μ̂ is what the model thinks; μ is the real thing"],
        ["e(x)", "chance this customer was contacted",
         "0.667 here, because two arms in three received email"],
        ["⪰", "is independent of", "knowing one tells you nothing about the other"],
        ["π (pi)", "a targeting policy", "a rule for deciding who goes on the list"],
        ["S", "the set a policy picked", "|S| means how many customers are in it"],
    ],
    "setup": {
        "title": "The estimation problem",
        "body": "Every customer <i>i</i> has two potential outcomes: <b>Y<sub>i</sub>(1)</b>, what "
                "they do if contacted, and <b>Y<sub>i</sub>(0)</b>, what they do if not. The "
                "individual treatment effect is the difference between them. Exactly one is ever "
                "observed, so the individual effect is never identified — this is the "
                "<b>fundamental problem of causal inference</b>. What can be identified is the "
                "conditional average effect over customers who look alike.",
        "eq": [
            ["Individual treatment effect", "τᵢ  =  Yᵢ(1) − Yᵢ(0)", "What contact did to this one person. Never observable, because only one of the two worlds ever happens."],
            ["What we estimate (CATE)", "τ(x)  =  E[ Y(1) − Y(0) | X = x ]", "The same difference, averaged over every customer who looks like this one. Knowable, because among lookalikes some were contacted and some were not."],
            ["Observed outcome", "Yᵢ  =  Tᵢ·Yᵢ(1) + (1−Tᵢ)·Yᵢ(0)", "Your data holds one outcome per customer. The missing half is what all the modelling is for."],
        ],
    },
    "assumptions": {
        "title": "What has to be true",
        "items": [
            ["Nothing hidden decided who got the email", "unconfoundedness:  ( Y(1), Y(0) ) ⫫ T | X",
             "Treatment is independent of the outcomes given covariates. Guaranteed here by "
             "randomisation — the retailer assigned the three arms at random, which is why no "
             "adjustment for selection is needed."],
            ["Everyone could have gone either way", "overlap:  0 < e(x) < 1  for all x",
             "Every customer had a non-zero chance of both being contacted and being held back. "
             "Satisfied by design: e(x) ≈ 0.667 for every customer in this experiment."],
            ["One customer's email cannot affect another", "SUTVA:  no interference, one version of treatment",
             "One customer's email does not change another customer's behaviour, and every email "
             "in an arm is the same email. Reasonable for direct mail; would need care for "
             "referral or social campaigns."],
        ],
    },
    "learners": {
        "propensity": {
            "formal": "Response model (not a causal estimator)",
            "eq": "μ̂₁(x) = P̂( Y = 1 | X = x, T = 1 )",
            "reads": "Among contacted customers who look like this one, what share came back? That is a chance of buying — not a change in behaviour.",
            "detail": "Fits the conditional response among the treated only. It estimates a "
                      "probability, not a difference, so it carries no counterfactual and answers "
                      "a different question. Included as the baseline because it is what most "
                      "retail stacks actually deploy.",
            "cite": "",
        },
        "s_learner": {
            "formal": "S-Learner (single learner)",
            "eq": "μ̂(x, t) fitted on pooled data\nτ̂(x) = μ̂(x, 1) − μ̂(x, 0)",
            "reads": "Train one model on everybody, with 'was contacted' as an input. Ask it about this customer twice — once pretending they were emailed, once pretending not — and subtract the two answers.",
            "detail": "One base learner over the pooled sample with treatment as an ordinary "
                      "feature. Low variance because it uses all the data, but biased toward zero "
                      "whenever regularisation or a tree's split criterion treats T as "
                      "unimportant — the effect is then simply not represented.",
            "cite": "Künzel et al. (2019), PNAS",
        },
        "t_learner": {
            "formal": "T-Learner (two learners)",
            "eq": "μ̂₁(x) on treated,  μ̂₀(x) on control\nτ̂(x) = μ̂₁(x) − μ̂₀(x)",
            "reads": "Train one model only on emailed customers and another only on held-back ones. Ask both about this customer, and take the gap between their answers.",
            "detail": "Two independent base learners, differenced. Unbiased in the sense that "
                      "neither model can suppress the effect, but the variances add rather than "
                      "cancel, and each learner sees only its own arm. Degrades sharply when the "
                      "arms are unbalanced.",
            "cite": "Künzel et al. (2019), PNAS",
        },
        "x_learner": {
            "formal": "X-Learner (cross-fitted imputation)",
            "eq": "D¹ᵢ = Yᵢ − μ̂₀(Xᵢ)   for treated i\n"
                  "D⁰ᵢ = μ̂₁(Xᵢ) − Yᵢ   for control i\n"
                  "τ̂(x) = g(x)·τ̂₀(x) + (1 − g(x))·τ̂₁(x)",
            "reads": "For each emailed customer, guess what they would have done unemailed and subtract — that is their estimated effect. Do the mirror image for the held-back ones. Then train a model on those estimated effects directly, and blend the two, leaning on whichever group has more data.",
            "detail": "Imputes each unit's effect using the opposite arm's outcome model, fits a "
                      "second-stage regression to those imputed effects in each arm, then blends "
                      "the two by the propensity g(x). The blend weights the arm with more data "
                      "more heavily, which is why it holds up under severe imbalance — the case "
                      "that matters in production, where holdouts are small.",
            "cite": "Künzel, Sekhon, Bickel & Yu (2019), PNAS 116(10)",
        },
        "transformed": {
            "formal": "Transformed / modified outcome",
            "eq": "Y*ᵢ = Yᵢ · ( Tᵢ − e(Xᵢ) ) / ( e(Xᵢ)·(1 − e(Xᵢ)) )\n"
                  "E[ Y* | X = x ] = τ(x)",
            "reads": "Rescale each outcome by how surprising it was that this customer got emailed. The algebra works out so the average of the rescaled number IS the effect — so one ordinary model predicts it directly, with no subtraction step.",
            "detail": "An algebraic reweighting whose conditional mean is the treatment effect "
                      "itself, so a single ordinary regression estimates τ(x) directly with no "
                      "differencing step. Unbiased but high variance: the transform divides by "
                      "e(x)(1−e(x)), so any region of thin overlap inflates the target. The "
                      "propensity-corrected generalisation of the Lai / Kane class transformation.",
            "cite": "Athey & Imbens (2016), PNAS 113(27)",
        },
    },
    "evaluation": {
        "title": "How these are scored, and why not with accuracy",
        "body": "There is no per-customer label to be accurate against, because τᵢ is never "
                "observed. Every valid metric therefore works by ranking the population and "
                "comparing treated to control <i>within</i> each prefix or bin, where "
                "randomisation makes the comparison causal.",
        "eq": [
            ["Qini at depth k", "Q(k) = Y₁(k) − Y₀(k) · N₁(k) / N₀(k)",
             "Walk down the ranked list. At each depth count responses among the contacted, then subtract what the held-back did — scaled so both groups count equally. What is left is the extra responses the ranking found."],
            ["Qini coefficient", "∫ Q(k) dk − area under random targeting",
             "The area between that curve and the straight line you would get by shuffling the list. One number for how much better than random the ranking is."],
            ["Offline policy value", "V(π) = Σ [ Ȳ₁(S) − Ȳ₀(S) ]·|S| · m − |S|·c",
             "Take the customers a model chose. Compare what the contacted ones did with the held-back ones inside that group, multiply by margin, subtract the contact cost. This is money — and it is what ranks the models on tab 1."],
        ],
        "invalid": ["Accuracy", "F1", "AUC-ROC", "log loss", "RMSE against Y"],
        "invalid_why": "All of these score a prediction of Y, not of τ. A model can maximise every "
                       "one of them while ranking uplift no better than chance — which is exactly "
                       "what the propensity baseline on the previous tab demonstrates.",
    },
    "stack": [
        ["Base learner", "sklearn HistGradientBoostingClassifier / Regressor",
         "Same algorithm family as LightGBM, no build toolchain required."],
        ["Meta-learners", "implemented directly on scikit-learn",
         "Roughly twenty lines each; deliberately not a wrapper around causalml."],
        ["Split", "65 / 35 stratified on treatment arm",
         "Models never see the customers they are scored on."],
        ["Selection", "offline policy value on held-out customers",
         "Ranked by realised money, not by a fit statistic."],
        ["Uncertainty", "bootstrap, 40 resamples",
         "Qini on one split is close to meaningless without an interval."],
    ],
}

PREDICTS = ("For every customer, one number: how much more likely they are to come back "
            "<b>because we emailed them</b>.")

GLOSSARY = [
    ("Extra visits",
     "How much more likely a customer is to come back because you emailed them. The emailed group "
     "here visited 16.7% of the time and the held-back group 10.6%, so the campaign caused 6.1 "
     "points of extra visits on average. Per customer it can be higher, lower, or negative."),
    ("Held-back group",
     "21,306 customers deliberately not emailed. They are the only reason any of this is "
     "measurable — without them, the emailed group's 16.7% would just look like success."),
    ("Realised value",
     "Let a model choose who to email, then look at what those exact customers actually did. "
     "Because the split was random, comparing the emailed and held-back people inside that chosen "
     "group gives a true incremental result, not a prediction."),
    ("Qini",
     "One score for how much better a model's ranking is than emailing people at random. Higher "
     "is better; zero means no better than a shuffled list. Shown with a range, because on real "
     "data it is noisy."),
    ("Why no accuracy score",
     "On real customers there is no per-person truth to check against — you see what someone did "
     "after being emailed, never what they would have done otherwise. So models are judged on "
     "what their decisions actually earned, which is the number that matters anyway."),
]


def _log(m):
    print(f"[real] {m}", flush=True)


def load() -> pd.DataFrame:
    if not CSV.exists():
        CSV.parent.mkdir(parents=True, exist_ok=True)
        _log("downloading Hillstrom (real retailer, randomised email test)")
        urllib.request.urlretrieve(URL, CSV)
    return pd.read_csv(CSV)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=raw.index)
    X["recency"] = raw["recency"].astype(float)              # months since last purchase
    X["history_log"] = np.log1p(raw["history"].astype(float))  # spend in the past year
    X["mens"] = raw["mens"].astype(float)
    X["womens"] = raw["womens"].astype(float)
    X["newbie"] = raw["newbie"].astype(float)
    X["history_band"] = raw["history_segment"].str.extract(r"^(\d)").astype(float)
    for z in ("Surburban", "Rural", "Urban"):
        X[f"zip_{z.lower()}"] = (raw["zip_code"] == z).astype(float)
    for c in ("Phone", "Web", "Multichannel"):
        X[f"chan_{c.lower()}"] = (raw["channel"] == c).astype(float)
    return X[list(FEATURES)].astype(float)


def realised_value(pick: np.ndarray, t: np.ndarray, y: np.ndarray,
                   rev_per_incremental_visit: float) -> tuple[float, float]:
    """What a chosen list ACTUALLY earned, using the real held-back group.

    Inside the chosen set, compare the people who happened to be emailed with the
    people who happened to be held back. The split was random, so that difference
    is a true causal effect -- no model, no assumption. This is offline policy
    evaluation, and it is the honest way to grade a targeting decision.
    """
    n = int(pick.sum())
    if n == 0:
        return 0.0, 0.0
    tt, yy = t[pick], y[pick]
    nt, nc = int(tt.sum()), int((1 - tt).sum())
    if nt == 0 or nc == 0:
        return 0.0, 0.0
    lift = yy[tt == 1].mean() - yy[tt == 0].mean()      # extra visit rate this list produced
    extra_visits = lift * n
    value = extra_visits * rev_per_incremental_visit * MARGIN_RATE - n * CONTACT_COST
    return float(value), float(extra_visits)


def main() -> dict:
    t0 = time.time()
    raw = load()
    _log(f"{len(raw):,} real customers, {(raw.segment == 'No E-Mail').sum():,} held back")

    X = build_features(raw)
    t = (raw["segment"] != "No E-Mail").astype(int).to_numpy()
    y = raw[OUTCOME].astype(int).to_numpy()

    # Revenue actually observed per visit -- the bridge from visits to money.
    rev_per_visit = float(raw["spend"].sum() / max(raw["visit"].sum(), 1))
    # An INCREMENTAL visit is not worth the site average. The experiment gives both
    # numbers directly, so the value of a caused visit is the caused spend divided
    # by the caused visits. Using the site average understated the case by ~37%.
    _lift_v = float(raw.loc[raw.segment != "No E-Mail", OUTCOME].mean()
                    - raw.loc[raw.segment == "No E-Mail", OUTCOME].mean())
    _lift_s = float(raw.loc[raw.segment != "No E-Mail", "spend"].mean()
                    - raw.loc[raw.segment == "No E-Mail", "spend"].mean())
    rev_per_incremental_visit = _lift_s / max(_lift_v, 1e-9)
    _log(f"observed: {raw[OUTCOME].mean():.2%} {OUTCOME} rate, ${rev_per_visit:.2f} revenue per visit")

    tr, te = train_test_split(np.arange(len(raw)), test_size=0.35,
                              random_state=7, stratify=raw["segment"])

    # ------------------------------------------------------------- bake-off
    fitted, rows = {}, []
    for key, cls in LEARNERS.items():
        t_start = time.time()
        m = cls().fit(X.iloc[tr], t[tr], y[tr])
        pred = m.predict(X.iloc[te])
        secs = time.time() - t_start
        res = evaluate.evaluate_all(pred, t[te], y[te], None)   # no ground truth on real data
        fitted[key] = m
        rows.append({
            "key": key, "name": cls.name, "is_uplift": key != "propensity",
            "qini": res["qini"], "ci": res["qini_ci"],
            "top": res["top_decile_uplift"], "bot": res["bottom_decile_uplift"],
            "deciles": [d["uplift"] for d in res["deciles"]],
            "secs": round(secs, 2), "corr": None,
            **MODEL_INFO[key],
        })
        if not rows[-1]["predicts"]:
            rows[-1]["predicts"] = PREDICTS
        _log(f"  {cls.name:22s} qini={res['qini']:7.1f}  top={res['top_decile_uplift']:+.2f}pp")

    # ------------------------------- what each model's decisions really earned
    # Judged on held-out customers only, so no model is graded on data it saw.
    Xte, tte, yte = X.iloc[te], t[te], y[te]
    margin_per_visit = rev_per_incremental_visit * MARGIN_RATE

    def value_at(pick, cost):
        """Re-price a chosen list at an arbitrary cost per contact."""
        v, extra = realised_value(pick, tte, yte, rev_per_incremental_visit)
        n_sel = int(pick.sum())
        return v + n_sel * CONTACT_COST - n_sel * cost, extra

    for r in rows:
        up = fitted[r["key"]].predict(Xte)
        # contact whoever the model thinks clears the cost of reaching them
        pick = (up * margin_per_visit) > RANK_AT
        val, extra = value_at(pick, RANK_AT)
        r["decision"] = {
            "contact_n": int(pick.sum()),
            "contact_share": round(float(pick.mean()) * 100, 1),
            "earns": round(val, 2),
            "extra_visits": round(extra, 1),
            "vs_best": 0.0, "of_possible": 0.0,
        }

    # reference points, on the same held-out customers
    everyone = np.ones(len(te), bool)
    all_val, all_extra = value_at(everyone, RANK_AT)
    all_val_asrun, _ = realised_value(everyone, tte, yte, rev_per_incremental_visit)
    # best achievable by ranking: walk the true decile curve and stop where it stops paying
    best_model_key = max((r for r in rows if r["is_uplift"]),
                         key=lambda r: r["decision"]["earns"])["key"]
    ordered = np.argsort(-fitted[best_model_key].predict(Xte))
    ceiling, best_k = -1e18, 0
    for k in range(500, len(te) + 1, 500):
        mask = np.zeros(len(te), bool)
        mask[ordered[:k]] = True
        v, _ = value_at(mask, RANK_AT)
        if v > ceiling:
            ceiling, best_k = v, k

    for r in rows:
        r["decision"]["vs_best"] = round(
            r["decision"]["earns"] - max(x["decision"]["earns"] for x in rows), 2)
        r["decision"]["of_possible"] = round(r["decision"]["earns"] / ceiling * 100, 1) if ceiling > 0 else 0.0

    rows.sort(key=lambda r: -r["decision"]["earns"])
    best = rows[0]
    # Verdicts come from what actually happened on this data. The previous badges
    # were carried over from the simulated run, so a model with negative Qini was
    # still described as merely "noisy".
    for r in rows:
        if r["key"] == best["key"]:
            r["verdict"], r["verdict_text"] = "best", "Best of the five here"
        elif not r["is_uplift"]:
            r["verdict"], r["verdict_text"] = "avoid", "Predicts the wrong thing"
        elif r["qini"] < 0:
            r["verdict"], r["verdict_text"] = "avoid", "Failed on this data"
        elif r["decision"]["earns"] < 0:
            r["verdict"], r["verdict_text"] = "ok", "Loses money at this price"
        else:
            r["verdict"], r["verdict_text"] = "good", "Works, but behind"

    prop = next(r for r in rows if not r["is_uplift"])
    ci_overlap = (best["ci"]["lo"] < prop["ci"]["hi"] and prop["ci"]["lo"] < best["ci"]["hi"])
    per_customer = best["decision"]["earns"] / len(te)
    honesty = {
        "ci_overlap": bool(ci_overlap),
        "champion_ci": best["ci"], "baseline_ci": prop["ci"],
        "per_customer": round(per_customer, 5),
        "verdict": ("On this dataset the recommended model's advantage over the baseline is "
                    "<b>not statistically distinguishable</b>. Its Qini interval "
                    f"[{best['ci']['lo']:.0f}, {best['ci']['hi']:.0f}] overlaps the baseline's "
                    f"[{prop['ci']['lo']:.0f}, {prop['ci']['hi']:.0f}], and the winning margin is "
                    f"${per_customer:.5f} per customer — which is noise, not a result. "
                    "Hillstrom is almost certainly too small and too weak-signal to support "
                    "per-customer targeting. Reporting that is a stronger finding than claiming "
                    "a win.") if ci_overlap else "",
    }
    _log(f"recommended on real data: {best['name']} "
         f"(earns ${best['decision']['earns']:,.0f} on {len(te):,} held-out customers)")

    # ------------------------------------------------- what a contact may cost
    # The campaign's own economics: +6.09 points of visits, each visit worth
    # rev_per_visit * margin. That product is the most a contact can be worth,
    # and therefore the exact price above which blanket sending loses money.
    # Measured across the WHOLE experiment, not just the held-out third: this is
    # the campaign's own result, not a statement about model performance. Using
    # the held-out figure here made the headline and the break-even disagree.
    lift_all = float(y[t == 1].mean() - y[t == 0].mean())
    spend_lift = float(raw.loc[raw.segment != "No E-Mail", "spend"].mean()
                       - raw.loc[raw.segment == "No E-Mail", "spend"].mean())
    value_per_contact = lift_all * margin_per_visit
    best_up = fitted[best_model_key].predict(Xte)

    cost_curve = []
    for opt in CONTACT_OPTIONS:
        c = opt["cost"]
        bl, _ = value_at(everyone, c)
        pick = (best_up * margin_per_visit) > c
        tg, tg_extra = value_at(pick, c)
        n_sel = int(pick.sum())
        verdict, advice = verdict_for(n_sel, len(te), tg, bl)
        cost_curve.append({
            **opt,
            "blanket": round(bl, 2),
            "targeted": round(tg, 2),
            "n_targeted": n_sel,
            "share_targeted": round(n_sel / len(te) * 100, 1),
            "gain": round(tg - bl, 2),
            "gain_is_avoided_loss": bool(tg <= 0),
            "verdict": verdict, "advice": advice,
        })

    # --------------------------------------- which of the two emails suits whom
    champ_cls = LEARNERS[best["key"]]
    per_offer = {}
    for arm, key in (("Mens E-Mail", "mens"), ("Womens E-Mail", "womens")):
        sub = raw.index[(raw["segment"] == arm) | (raw["segment"] == "No E-Mail")].to_numpy()
        ts = (raw["segment"].to_numpy()[sub] == arm).astype(int)
        per_offer[key] = champ_cls().fit(X.iloc[sub], ts, y[sub])
        _log(f"  trained the {key} email model on {len(sub):,} real customers")

    # ------------------------------------------------ score every real customer
    champ = fitted[best["key"]]
    up_all = champ.predict(X)
    p_ctrl = champ.predict_control(X)
    up_m = per_offer["mens"].predict(X)
    up_w = per_offer["womens"].predict(X)

    merch = np.where((raw["mens"] == 1) & (raw["womens"] == 1), "both",
                     np.where(raw["mens"] == 1, "mens", "womens"))

    # Sending the men's campaign to a womenswear buyer needs a real reason. The
    # men's arm lifted more overall (7.66pp vs 4.52pp), so an unconstrained argmax
    # handed the men's email to 70% of womenswear buyers -- defensible statistically,
    # indefensible commercially. Require the mismatched arm to win by a real margin.
    which = np.array([choose_offer(bm, bw, m, w) for bm, bw, m, w
                      in zip(raw["mens"] == 1, raw["womens"] == 1, up_m, up_w)])
    up_best = np.maximum(up_m, up_w)
    value = up_all * margin_per_visit - CONTACT_COST   # priced at the real send cost
    spend_line = np.full(len(raw), CONTACT_COST)   # the campaign as actually run

    demand = float(np.median(p_ctrl))
    action = np.where(up_all < -0.002, "suppress",
                      np.where(value > 0, "contact",
                               np.where(p_ctrl >= demand, "no_offer", "not_worth")))
    reward = np.where(action == "contact", which, "")

    product = np.array([f"{MERCH[m]} · {seg.split(') ')[-1]}"
                        for m, seg in zip(merch, raw["history_segment"])])

    cust = pd.DataFrame({
        "customer_id": np.arange(1, len(raw) + 1),
        # ---- real Hillstrom fields, unaltered
        "months_since_purchase": raw["recency"].to_numpy(),
        "spend_12m": raw["history"].to_numpy(),
        "spend_band": raw["history_segment"].str.replace(r"^\d\) ", "", regex=True).to_numpy(),
        "buys_mens": raw["mens"].to_numpy(),
        "buys_womens": raw["womens"].to_numpy(),
        "new_customer": raw["newbie"].to_numpy(),
        "area": raw["zip_code"].replace({"Surburban": "Suburban"}).to_numpy(),
        "channel": raw["channel"].to_numpy(),
        # ---- derived labels, clearly derived
        "category": merch,
        "product": product,
        # ---- model output
        "buys_alone_pct": p_ctrl * 100,
        "buys_if_contacted_pct": np.clip(p_ctrl + up_all, 0, 1) * 100,
        "extra_sales_pp": up_all * 100,
        "value_of_contact": value,
        "cost_of_contact": spend_line,
        "action": action,
        "reward": reward,
        # ---- what actually happened to them in the experiment
        "arm": raw["segment"].to_numpy(),
        "visited": raw["visit"].to_numpy(),
        "spent": raw["spend"].to_numpy(),
    }).set_index("customer_id", drop=False)
    cust.to_pickle(CUSTOMERS)

    # --------------------------------------------------------------- headline
    n = len(cust)
    grp = {k: cust[cust["action"] == k] for k in ACTIONS}
    total_spend = float(spend_line.sum())
    good_spend = float(grp["contact"]["cost_of_contact"].sum())
    wasted = total_spend - good_spend

    scale = n / len(te)                       # held-out result, expressed over the full base
    profit_targeted = best["decision"]["earns"] * scale
    profit_blanket = all_val * scale
    for row in cost_curve:                    # same, for every contact method
        row["blanket"] = round(row["blanket"] * scale, 2)
        row["targeted"] = round(row["targeted"] * scale, 2)
        row["gain"] = round(row["gain"] * scale, 2)
        row["n_targeted"] = int(round(row["n_targeted"] * scale))

    answer = {
        "waste_pct": round(wasted / total_spend * 100, 1),
        "wasted_spend": round(wasted, 2),
        "total_spend": round(total_spend, 2),
        "profit_blanket": round(profit_blanket, 2),
        "profit_targeted": round(profit_targeted, 2),
        "gain": round(profit_targeted - profit_blanket, 2),
        "n": n,
        "groups": [{
            "key": k, "label": ACTIONS[k]["label"], "tone": ACTIONS[k]["tone"],
            "why": ACTIONS[k]["why"], "n": int(len(g)),
            "share": round(len(g) / n * 100, 1),
            "spend": round(float(g["cost_of_contact"].sum()), 2),
            "value": round(float(g["value_of_contact"].sum()), 2),
        } for k, g in grp.items()],
        "assumptions": {
            "margin_rate": MARGIN_RATE, "offer_rate": 0.0,
            "contact_cost": CONTACT_COST,
            "avg_order_value": round(float(raw.loc[raw.conversion == 1, "spend"].mean()), 2),
            "rev_per_visit": round(rev_per_visit, 2),
            "breakeven_pp": round(CONTACT_COST / margin_per_visit * 100, 2),
        },
        "scaled": {"base": 1_000_000,
                   "wasted": round(wasted / n * 1_000_000),
                   "gain": round((profit_targeted - profit_blanket) / n * 1_000_000)},
    }
    _log(f"answer: emailing everyone earns ${profit_blanket:,.0f}; "
         f"targeting earns ${profit_targeted:,.0f}")

    # -------------------------------------------------------------- roll-ups
    def mix(col, labels):
        out = []
        for k, g in cust.groupby(col):
            out.append({"key": k, "label": labels.get(k, k), "n": int(len(g)),
                        "share": round(len(g) / n * 100, 1),
                        "spend": round(float(g["spend_12m"].sum())),
                        "avg_order": round(float(g["spend_12m"].mean()), 2),
                        "contact_n": int((g["action"] == "contact").sum()),
                        "extra_sales_pp": round(float(g["extra_sales_pp"].mean()), 2)})
        return sorted(out, key=lambda r: -r["n"])

    _w = cust[cust["action"] == "contact"]
    _ok = int(((_w["category"] == "both")
               | (_w["category"] == _w["reward"])).sum())
    offer_match = {
        "matched": _ok, "total": int(len(_w)),
        "share": round(_ok / max(len(_w), 1) * 100, 1),
        "gap_pp": OFFER_GAP * 100,
        "note": ("The offer follows what the customer actually buys. The other campaign "
                 f"has to beat it by more than {OFFER_GAP * 100:.0f} points before it overrides that."),
    }

    rewards = []
    for k, meta in REWARD_META.items():
        g = cust[cust["reward"] == k]
        rewards.append({"key": k, "label": meta["label"], "why": meta["why"], "n": int(len(g)),
                        "value": round(float(g["value_of_contact"].sum()), 2),
                        "extra_sales": round(float(g["extra_sales_pp"].sum() / 100), 1)})

    BASE = {"seen": float(cust["months_since_purchase"].mean())}
    products = []
    for k, g in cust.groupby("product"):
        worth = g[g["action"] == "contact"]
        aud = worth if len(worth) >= 20 else g
        seen = float(aud["months_since_purchase"].mean())
        rk = worth["reward"].mode().iloc[0] if len(worth) and not worth["reward"].mode().empty else "mens"
        products.append({
            "product": k, "category": MERCH.get(g["category"].iloc[0], g["category"].iloc[0]),
            "n": int(len(g)), "spend": round(float(g["spend_12m"].sum())),
            "avg_order": round(float(g["spend_12m"].mean()), 2),
            "per_buyer": round(float(g["spend_12m"].mean()), 2),
            "contact_n": int(len(worth)),
            "contact_share": round(len(worth) / max(len(g), 1) * 100, 1),
            "extra_sales_pp": round(float(g["extra_sales_pp"].mean()), 2),
            "buys_on_promo_pct": 0,
            "reward": rk, "reward_label": REWARD_META[rk]["label"],
            "reward_why": REWARD_META[rk]["why"],
            "gain": round(float(worth["value_of_contact"].sum()), 2),
            "extra_sales": round(float(worth["extra_sales_pp"].sum() / 100), 1),
            "audience": ("Longest-lapsed" if seen > BASE["seen"] * 1.08 else
                         "Recently active" if seen < BASE["seen"] * 0.92 else "Mid-lapse")
                        + " buyers",
            "audience_detail": f"last bought {seen:.0f} months ago, "
                               f"${float(aud['spend_12m'].mean()):,.0f} spent in the past year",
        })
    products.sort(key=lambda r: -r["spend"])
    for r in products:
        r["gain_per_100"] = round(r["gain"] / max(r["n"], 1) * 100, 2)
    weak = sorted(products, key=lambda r: r["spend"])[:5]
    for w in weak:
        w["fixable"] = w["extra_sales_pp"] > 0.5 and w["contact_n"] >= 20

    metrics = {
        "app": {"name": "NextBest", "tagline": "Retail Customer Intelligence"},
        "data_is_real": True,
        "source": {
            "name": "Hillstrom MineThatData (2008)",
            "detail": f"{len(raw):,} real customers of a real retailer, randomly split into "
                      f"men's email, women's email and no email.",
            "url": URL,
        },
        "project": PROJECT, "reference": REFERENCE, "problem": PROBLEM, "predicts": PREDICTS, "answer": answer,
        "honesty": honesty,
        "priced_at": {
            "customers": CONTACT_COST,   # the campaign as it was actually run
            "bakeoff": RANK_AT,          # a what-if price where targeting matters
            "note": "Customer rows are priced at the real send cost. The model comparison is "
                    "priced higher on purpose: at the real cost every model says 'email everyone', "
                    "so there is nothing to compare.",
        },
        "economics": {
            "lift_pp": round(lift_all * 100, 2),
            "spend_lift": round(spend_lift, 3),
            "n_emailed": int((t == 1).sum()),
            "n_held_back": int((t == 0).sum()),
            "rate_emailed": round(float(y[t == 1].mean()) * 100, 1),
            "rate_held_back": round(float(y[t == 0].mean()) * 100, 1),
            "rev_per_visit": round(rev_per_visit, 2),
            "rev_per_incremental_visit": round(rev_per_incremental_visit, 2),
            "margin_rate": MARGIN_RATE,
            "margin_per_visit": round(margin_per_visit, 2),
            "value_per_contact": round(value_per_contact, 4),
            "as_run_cost": CONTACT_COST,
            "as_run_profit": round(all_val_asrun * (n / len(te)), 2),
            "ranked_at": RANK_AT,
        },
        "cost_curve": cost_curve,
        "models": rows, "best": {"key": best["key"], "name": best["name"]},
        "oracle": {"best_possible": round(ceiling * scale, 2),
                   "blanket": round(profit_blanket, 2)},
        "actions": ACTIONS, "rewards": rewards, "offer_match": offer_match,
        "categories": mix("category", MERCH), "channels": mix("channel", CHANNEL_LABELS),
        "products": products,
        "product_focus": {
            "best": products[0], "worst": weak[0], "top5": products[:5], "weak5": weak,
            "opportunity": sorted(products, key=lambda r: -r["gain_per_100"])[:5],
            "fixable_n": sum(1 for w in weak if w["fixable"]),
            "scored_by": best["name"],
            "offer_scored_by": f"one {best['name']} per email arm",
            "note": "Ranked by TOTAL money spent in the past year, which mostly tracks how many "
                    "buyers a group has — not how valuable each one is. Check the per-buyer "
                    "column before acting: the smallest group by total sales is often the most "
                    "valuable per head."
                    "The opportunity table below ranks by what an email is worth per customer "
                    "",
        },
        "glossary": GLOSSARY,
        "base": {
            "n": n, "revenue": round(float(cust["spend_12m"].sum())),
            "avg_order": round(float(raw.loc[raw.conversion == 1, "spend"].mean()), 2),
            "n_test": int(len(te)),
            "control_rate": round(float(raw.loc[raw.segment == "No E-Mail", OUTCOME].mean()) * 100, 1),
            "treated_rate": round(float(raw.loc[raw.segment != "No E-Mail", OUTCOME].mean()) * 100, 1),
            "outcome_word": "came back",
            "conversion_rate": round(float(raw["conversion"].mean()) * 100, 2),
            "offer_tested": "Men's email vs women's email vs no email",
        },
        "generated_at": time.strftime("%d %b %Y, %H:%M"),
        "seconds": round(time.time() - t0, 1),
    }

    # Everything needed to re-price any policy at a new margin or contact cost
    # WITHOUT retraining: the champion's held-out scores plus the arms and
    # outcomes for those same customers. Re-pricing is then pure arithmetic.
    joblib.dump({
        "models": fitted, "per_offer": per_offer, "features": list(FEATURES),
        "eval": {
            "uplift": best_up.astype("float32"),
            "t": tte.astype("int8"),
            "y": yte.astype("int8"),
            "rev_per_incremental_visit": rev_per_incremental_visit,
            "lift_all": lift_all,
            "n_base": int(len(raw)),
        },
        "base_uplift": up_all.astype("float32"),
        "base_pctrl": p_ctrl.astype("float32"),
    }, ARTIFACTS)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log(f"wrote real-data artifacts in {metrics['seconds']}s")
    return metrics


if __name__ == "__main__":
    main()
