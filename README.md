<div align="center">

# NextBest

**Most marketing money is spent on people who were going to buy anyway.**
This finds the ones who buy *because* you contacted them.

### [→ Open the live app](https://jampana-jagadeesh.github.io/NextBest/)

[![tests](https://github.com/Jampana-Jagadeesh/NextBest/actions/workflows/ci.yml/badge.svg)](https://github.com/Jampana-Jagadeesh/NextBest/actions/workflows/ci.yml)
[![pages](https://github.com/Jampana-Jagadeesh/NextBest/actions/workflows/pages.yml/badge.svg)](https://github.com/Jampana-Jagadeesh/NextBest/actions/workflows/pages.yml)

`uplift modelling` · `causal inference` · `64,000 real customers` · `5 estimators` · `23 tests`

</div>

---

## The problem, in one line

A campaign report shows revenue. It does not show **extra** revenue.

A discount handed to a customer who had already decided to buy still lands in the
results as a win. So the report always looks good, and nobody can tell which part
of it the campaign actually caused.

NextBest measures the part you cannot see — how much being contacted **changes**
what each customer does — and turns it into four decisions:

| | Group | What it means | What you do |
|:--:|---|---|---|
| 🔵 | **Email them** | Contact genuinely changes their behaviour | Send |
| 🟡 | **No email needed** | They were coming back regardless | Send content, not a coupon |
| ⚪ | **Not worth emailing** | No response either way | Skip |
| 🔴 | **Leave them alone** | Contact actively puts them off | Suppress |

A propensity model cannot tell the last two apart from the first — all three look
like "likely buyer". Only a model of *change* returns a negative score.

---

## What the real data says

Measured against the **21,306 customers held back at random**:

| | |
|---|---|
| The campaign caused | **+6.1 points** of extra visits — 10.6% → 16.7% |
| An incremental visit is worth | **$4.41** — $7.16 revenue at **45%** gross margin |
| **So one contact is worth** | **$0.269** ← the break-even |

That single number decides everything:

| Channel | Cost | Blanket send | Best targeted list | Verdict |
|---|--:|--:|--:|:--|
| Plain email | $0.08 | **+$10,596** | +$10,347 | Contact everyone — a model adds nothing |
| Email + 10% off | $0.30 | −$3,484 | −$86 | **Do not run it** — even the best list loses |
| SMS | $0.45 | −$13,084 | −$363 | **Do not run it** |
| Direct mail | $0.85 | −$38,684 | +$46 | **Do not run it** — only 34 people qualify |

Below **$0.27** a contact, send to everyone and skip the model entirely. Above it,
targeting is the only thing keeping the campaign alive. **The app says so on
screen** — including when the honest answer is "don't bother".

---

## The honest result

> On this dataset the recommended model's advantage over the baseline is **not
> statistically distinguishable**. Its Qini interval **[7, 115]** overlaps the
> baseline's **[−3, 53]**, and the winning margin is **−$0.00134 per customer** —
> which is noise, not a result.

That paragraph is on the **front page of the app**, above the numbers it qualifies.

Hillstrom is almost certainly too small and too weak-signal to support
per-customer targeting. A tool that reports a null result honestly is worth more
than one that manufactures a win.

The same applies to the verdicts. Beating a worse option is not the same as being
worth doing — so a list that still loses money returns **stop**, and so does a
direct-mail list of 34 people, however good the per-head economics look.

---

## Why there is no accuracy score

A churn model predicts something you eventually observe, so you can score it.
An uplift model predicts **how much contact changed a person's behaviour** — and
you only ever see one branch. Customer #4510 was emailed and came back; whether
they would have come back anyway is unobservable, forever.

No label, so no accuracy, no AUC, no F1. Any tool showing you one for uplift is
scoring a different question than the one it claims.

**So models are ranked by what their decisions actually earned** on held-out
customers — offline policy evaluation — which is the number a business cares
about anyway:

| # | Estimator | Qini | Top decile | Net at $0.30 | |
|:--:|---|--:|--:|--:|:--|
| 1 | **Transformed Outcome** | **48.7** | +8.9pp | −$30 | Best of the five here |
| 2 | X-Learner | 41.9 | +8.4pp | −$129 | Works, ranks behind |
| 3 | S-Learner | 15.7 | +7.5pp | −$214 | Works, ranks behind |
| 4 | T-Learner | −13.2 | +6.2pp | −$862 | Failed on this data |
| 5 | Propensity *(baseline)* | 30.1 | +7.7pp | −$1,099 | **Predicts the wrong thing** |

Look at row 5. It ranks **3rd on Qini and last on money** — it sorts people
plausibly, then loses the most. That is the entire argument of this project in
one line.

**Where accuracy *is* measurable:** the simulator plants a known per-customer
effect, so correlation with truth can be checked. S-Learner **0.743**, X-Learner
**0.694**, Transformed **0.671**, T-Learner **0.533** — and propensity **0.248**,
which is an asserted test. If that ever rises, the argument collapses.

---

## Run it

```bash
pip install -r requirements.txt

python simple/app.py     # the officer's view   -> http://127.0.0.1:8050
python run.py            # the analyst console  -> http://127.0.0.1:8000
```

Either builds its own data on first run, then starts instantly. No Node, no build
step, no `causalml` — numpy, pandas, scikit-learn and FastAPI.

| | **simple/** — officer view | **api/** — analyst console |
|---|---|---|
| Data | 64,000 **real** customers (Hillstrom) | 120,000 simulated, six-arm |
| Audience | Whoever runs the campaign | Whoever runs the models |
| Tabs | The model · Products & offers · Customers · The ML | Six analyst screens |
| Answers | What do I do on Monday? | Which model, which offer, what budget? |

Both read `src/nextbest/`. After changing anything there:

```bash
python simple/realbuild.py    # officer view, real data
python -m nextbest.train      # analyst console, simulated
python -m pytest tests -q     # 23 tests
```

---

## The data is real. The simulator is a test fixture.

`simple/` runs on **Hillstrom MineThatData (2008)** — 64,000 real customers of a
real retailer, randomly split into men's email, women's email and no email. That
random split is what makes the difference between groups a **causal effect**
rather than a correlation.

`src/nextbest/simulate.py` plants a known effect per customer. Useless as a
product, ideal as a **test fixture**: the suite asserts the four uplift learners
recover the planted effect and that the propensity model does not. Neither check
is possible on real data.

Any dataset works provided it has a **randomly assigned** treatment flag. Without
a held-back group there is no counterfactual, and no modelling recovers one.

---

## Decisions worth defending

**The offer follows what the customer buys.** The men's arm lifted more overall
(7.66pp vs 4.52pp), so a per-customer argmax sent the men's campaign to **70% of
womenswear buyers** — defensible statistically, indefensible commercially. The
other arm must now win by more than two points to override merchandise.
**91.3%** of the list matches, and the app flags the split if it drops below 75%.

**The discount is charged against P(buy | contacted)**, not against uplift. A
coupon is redeemed by everyone who buys, not only by the extra sales it caused.
Charging it to incremental buyers alone hid the entire cost of discounting
someone who was going to purchase regardless. That was a real bug; there is now a
test for it.

**Colour is never decoration.** The interface is monochrome; hue appears only on
the four decisions. Every foreground/background pair clears WCAG AA 4.5:1 in both
themes, and the states stay separable under protanopia and deuteranopia — with
icons carrying the meaning too, because with four states where one must be grey
and one must be red, no hue assignment clears every gate.

---

## Layout

```
NextBest/
├── docs/                 the published static build (built by CI)
├── simple/               the officer view, on real data
│   ├── realbuild.py      trains, scores, writes the copy
│   ├── app.py            FastAPI: overview, customers, recompute, export, whatif
│   └── static/           hand-built UI — no framework, no chart library
├── src/nextbest/         shared core
│   ├── simulate.py       known-effect generator — test fixture, not the product
│   ├── learners.py       S / T / X / Transformed Outcome + propensity baseline
│   ├── evaluate.py       Qini, decile lift, profit curve, bootstrap CIs
│   └── optimize.py       offer choice + budget allocation
├── tests/                23 tests, run in CI on the simulator
├── build_static.py       -> docs/  for GitHub Pages
└── run.py                analyst console launcher
```

**Publishing.** `.github/workflows/pages.yml` rebuilds from the raw dataset on
every push to `main`, runs the tests, then deploys — so the published site can
never drift from the code. `docs/data.json` (5.5 MB, ~1.2 MB gzipped) is built in
CI rather than committed. Every endpoint moves to the browser for the hosted
build; the only feature not included is the single-customer scorer, which needs
the fitted models. It is hidden rather than faked, and the page says so.

---

## Known limitations

- **No randomised holdout at most retailers.** Organisational, not technical, and
  it gates the whole product. Without one, none of this runs.
- **The prize is small here.** −$0.00134 per customer, confidence intervals
  overlapping. See the banner on tab 1.
- **No per-customer uncertainty.** Needs a Causal Forest; shipping fake error bars
  would be worse than none.
- **No feature store, no activation, no feedback loop.** Point-in-time features,
  an ESP/CDP connector, and campaign → holdout → realised lift → retrain are
  designed in `NextBest_Project_Plan.md` and not built. That is most of a real
  deployment.
- **Attribution is not SHAP** — counterfactual median substitution, labelled as
  such in the code and on screen.

---

<div align="center">

**Jagadeesh Jampana** · Retail customer intelligence · uplift modelling

[Email](mailto:jagadeeshjampana5@gmail.com) · [GitHub](https://github.com/Jampana-Jagadeesh) · [LinkedIn](https://linkedin.com/in/jampana-jagadeesh-9704002a2/)

<sub>Data: Hillstrom MineThatData (2008) — 64,000 real customers</sub>

</div>
