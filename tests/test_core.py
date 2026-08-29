"""Tests for the parts that transfer to a real deployment.

The simulator earns its keep here rather than in the product: because it plants a
known treatment effect, the learners can be checked against truth — something no
real dataset allows. Everything else is arithmetic that must hold regardless of
data.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nextbest import evaluate, features, optimize, simulate  # noqa: E402
from nextbest.config import CONTACT_COST, MARGIN_RATE, OFFER_BY_KEY, OFFER_KEYS  # noqa: E402
from nextbest.learners import LEARNERS  # noqa: E402

SEED = 11


@pytest.fixture(scope="module")
def sim():
    """A two-arm randomised campaign with the true per-customer effect known."""
    raw = simulate.generate(n=14_000, seed=SEED, arms=["pct10"])
    X = features.build(raw)
    return {
        "raw": raw, "X": X,
        "t": raw["treated"].to_numpy(),
        "y": raw["converted"].to_numpy(),
        "tau": raw["tau_pct10"].to_numpy(),        # ground truth, never a model input
    }


# ----------------------------------------------------------------- simulator
def test_randomisation_is_balanced(sim):
    """Arms must be assigned independently of the outcome, or nothing downstream
    is a causal estimate."""
    raw = sim["raw"]
    share = raw["treated"].mean()
    assert 0.45 < share < 0.55, f"arms not balanced: {share:.3f}"
    # covariates should look the same in both arms
    for col in ("recency_n", "engagement", "discount_affinity"):
        a = raw.loc[raw.treated == 1, col].mean()
        b = raw.loc[raw.treated == 0, col].mean()
        assert abs(a - b) < 0.02, f"{col} differs across arms: {a:.3f} vs {b:.3f}"


def test_simulator_contains_all_four_archetypes(sim):
    """If the data has no negative-effect customers there is nothing to suppress,
    and the product's central claim cannot be demonstrated."""
    tau = sim["tau"]
    assert (tau > 0.05).mean() > 0.10, "too few persuadables"
    assert (tau < 0).mean() > 0.05, "no customers harmed by contact"


def test_ground_truth_never_reaches_the_model(sim):
    with pytest.raises(AssertionError, match="LEAKAGE"):
        bad = sim["X"].copy()
        bad["converted"] = sim["y"]
        features.assert_no_leakage(bad)


# ------------------------------------------------------------------ learners
@pytest.mark.parametrize("key", ["s_learner", "t_learner", "x_learner", "transformed"])
def test_uplift_learners_recover_the_true_effect(sim, key):
    """Each uplift learner should correlate with the planted effect. This is the
    check that is impossible on real data, and the whole reason to simulate."""
    m = LEARNERS[key]().fit(sim["X"], sim["t"], sim["y"])
    pred = m.predict(sim["X"])
    corr = float(np.corrcoef(pred, sim["tau"])[0, 1])
    assert corr > 0.25, f"{key} barely tracks the true effect: r={corr:.3f}"


def test_propensity_model_does_not_recover_the_effect(sim):
    """The project's central claim: a response model is not an uplift model.
    If this ever starts passing as an uplift learner, the argument collapses."""
    m = LEARNERS["propensity"]().fit(sim["X"], sim["t"], sim["y"])
    pred = m.predict(sim["X"])
    corr = abs(float(np.corrcoef(pred, sim["tau"])[0, 1]))
    assert corr < 0.25, f"propensity unexpectedly tracks uplift: r={corr:.3f}"


def test_predictions_actually_vary(sim):
    """An S-Learner that ignores the treatment column returns a flat zero and
    still looks fine on some metrics. Catch that here."""
    m = LEARNERS["s_learner"]().fit(sim["X"], sim["t"], sim["y"])
    pred = m.predict(sim["X"])
    assert pred.std() > 0.005, "uplift predictions are effectively constant"


# ---------------------------------------------------------------- evaluation
def test_qini_rewards_a_good_ranking_and_not_a_shuffled_one(sim):
    rng = np.random.default_rng(SEED)
    good = evaluate.qini_coefficient(sim["tau"], sim["t"], sim["y"])       # perfect ranking
    shuffled = evaluate.qini_coefficient(rng.permutation(sim["tau"]), sim["t"], sim["y"])
    assert good > 0, "a perfect ranking scored non-positive Qini"
    assert good > shuffled, "Qini did not prefer the true ranking to a shuffled one"


def test_qini_scales_control_to_the_treated_head_count():
    """The control arm must be rescaled at every depth, otherwise an unbalanced
    split silently inflates or deflates the curve."""
    n = 400
    t = np.array([1] * 300 + [0] * 100)          # deliberately 3:1
    y = np.zeros(n)
    y[:60] = 1                                    # 20% of treated respond
    y[300:320] = 1                                # 20% of control respond
    score = np.linspace(1, 0, n)
    q = evaluate.qini_curve(score, t, y)["q"][-1]
    # equal response rates => no incremental effect, whatever the arm sizes
    assert abs(q) < 1e-6, f"unbalanced arms leaked into Qini: {q}"


def test_decile_lift_is_ordered_for_a_perfect_ranking(sim):
    d = [x["uplift"] for x in evaluate.uplift_by_decile(sim["tau"], sim["t"], sim["y"])]
    assert d[0] > d[-1], "top decile should out-lift the bottom under a perfect ranking"


# ----------------------------------------------------------------- economics
def test_discount_is_charged_to_everyone_who_buys_not_only_extra_buyers():
    """The bug that hid the entire cost of discounting a customer who was going
    to purchase anyway. A sure thing -- high baseline, no uplift -- must price as
    a loss."""
    up = pd.DataFrame({k: [0.0] for k in OFFER_KEYS})     # zero incremental effect
    aov = np.array([100.0])
    p_control = np.array([0.80])                           # buys 80% of the time regardless

    correct = optimize.expected_profit_matrix(up, aov, p_control)["pct10"].iloc[0]
    offer_cost = OFFER_BY_KEY["pct10"].cost(100.0)
    assert correct == pytest.approx(-0.80 * offer_cost - CONTACT_COST), \
        "discount was not charged against P(buy | contacted)"
    assert correct < 0, "discounting a sure thing must show as a loss"

    with pytest.warns(RuntimeWarning, match="understates offer cost"):
        naive = optimize.expected_profit_matrix(up, aov)["pct10"].iloc[0]
    assert naive > correct, "the naive costing should look better than it is"


def test_profit_rises_with_uplift_and_falls_with_offer_cost():
    up = pd.DataFrame({k: [0.02, 0.20] for k in OFFER_KEYS})
    aov = np.array([100.0, 100.0])
    p = np.array([0.10, 0.10])
    prof = optimize.expected_profit_matrix(up, aov, p)["pct10"].to_numpy()
    assert prof[1] > prof[0], "more uplift must be worth more"


def test_allocation_never_exceeds_the_budget_and_skips_negative_uplift():
    rng = np.random.default_rng(SEED)
    n = 500
    scored = pd.DataFrame({
        "offer_uplift": rng.normal(0.02, 0.05, n),
        "expected_profit": rng.normal(1.0, 2.0, n),
        "expected_cost": rng.uniform(0.4, 1.2, n),
    })
    res = optimize.allocate(scored, budget=100.0)
    assert res["spend"] <= 100.0 + 1e-9, "allocation overspent the budget"
    chosen = scored.loc[res["selected_ids"]] if res["selected_ids"] else scored.iloc[0:0]
    assert (chosen["offer_uplift"] > 0).all(), "a negative-uplift customer was selected"
    assert (chosen["expected_profit"] > 0).all(), "a loss-making customer was selected"


def test_quadrants_use_both_axes():
    """A slightly negative score at low baseline demand is a lost cause, not a
    sleeping dog -- classifying on uplift alone conflates the two."""
    up = np.array([0.10, -0.03, -0.03, 0.00])
    pc = np.array([0.20, 0.40, 0.05, 0.40])
    q = optimize.classify_quadrant(up, pc, base_threshold=0.30)
    assert list(q) == ["persuadable", "sleeping_dog", "lost_cause", "sure_thing"]


# --------------------------------------------------------------- consistency
def test_readme_margin_matches_the_code():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**{int(MARGIN_RATE * 100)}%** gross margin" in txt, \
        "README and config.py disagree about gross margin"


# ------------------------------------------------- the two decision rules
# These live in simple/realbuild.py because they are product rules, not model
# maths, but they are the rules a marketer actually reads off the screen. Both
# were wrong in a way the model metrics could not catch, so they are pinned here.
sys.path.insert(0, str(ROOT / "simple"))
import realbuild as RB  # noqa: E402


def test_the_offer_follows_what_the_customer_buys():
    """The men's arm lifted more overall, so an unconstrained argmax sent the
    men's campaign to 70% of womenswear buyers. A small edge must not override
    what someone actually purchases."""
    small = RB.OFFER_GAP / 2
    assert RB.choose_offer(False, True, up_mens=0.05 + small, up_womens=0.05) == "womens"
    assert RB.choose_offer(True, False, up_mens=0.05, up_womens=0.05 + small) == "mens"


def test_a_big_enough_edge_still_overrides_the_category():
    """The rule is a floor on the evidence, not a ban. A genuinely large gap
    should still win, or the model is not being used at all."""
    big = RB.OFFER_GAP * 3
    assert RB.choose_offer(False, True, up_mens=0.05 + big, up_womens=0.05) == "mens"
    # someone who buys both has no natural match, so the argmax stands
    assert RB.choose_offer(True, True, up_mens=0.02, up_womens=0.01) == "mens"
    assert RB.choose_offer(True, True, up_mens=0.01, up_womens=0.02) == "womens"


def test_a_losing_campaign_is_never_recommended():
    """Beating a worse option is not the same as being worth doing. Targeting
    that loses $86 where blanket sending loses $3,484 is still a loss."""
    v, advice = RB.verdict_for(n_sel=30_897, n_base=64_000,
                               targeted=-85.67, blanket=-3484.0)
    assert v == "stop", "recommended a campaign that loses money"
    assert "loses money" in advice


def test_a_campaign_needs_an_audience_worth_the_name():
    """34 people out of 64,000 is a rounding error, not a direct-mail campaign,
    however good the per-head economics look."""
    v, _ = RB.verdict_for(n_sel=34, n_base=64_000, targeted=46.48, blanket=-38_684.0)
    assert v == "stop", "recommended a campaign to 0.05% of the base"


def test_a_genuinely_profitable_list_is_still_recommended():
    """The floor and the loss check must not swallow the real case."""
    assert RB.verdict_for(20_000, 64_000, targeted=9_000.0, blanket=1_000.0)[0] == "target"
    assert RB.verdict_for(63_000, 64_000, targeted=10_347.0, blanket=10_300.0)[0] == "blanket"
    assert RB.verdict_for(0, 64_000, targeted=0.0, blanket=-100.0)[0] == "stop"
