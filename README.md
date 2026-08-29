# NextBest — Retail Customer Intelligence

Most retail marketing money is spent on people who were going to buy anyway. A
campaign report shows revenue, not **extra** revenue — a discount handed to a
customer who had already decided to buy still lands in the results as a win.

On this dataset the campaign did pay, but only because email is nearly free.
At **$0.30** a contact every one of the five models loses money, and the app
says so rather than dressing the least-bad list up as a profit.

NextBest measures the part you cannot see — how much being contacted actually
**changes** what each customer does — and turns that into four decisions:
contact them, no offer needed, not worth contacting, or leave them alone.

---

## Live demo

The app is published to GitHub Pages as a fully working static build:

**https://jampana-jagadeesh.github.io/NextBest/**

GitHub Pages serves files, not Python — so every endpoint moves to the browser.
The sliders still re-price against the real held-back group using the same
formula the server uses, the customer list still filters 64,000 real rows, and
CSV export is generated client-side. The only feature not in the hosted build is
the single-customer scorer, which needs the fitted models; it is hidden rather
than faked, and the page says so.

### Publishing it yourself

```bash
python simple/realbuild.py     # train on the real data
python build_static.py         # -> docs/
python -m http.server -d docs 8080    # check it at localhost:8080
```

Then in the repo: **Settings → Pages → Source: GitHub Actions**. The workflow in
`.github/workflows/pages.yml` rebuilds from the real dataset on every push to
`main`, runs the tests, and deploys — so the published site can never drift from
the code.

`docs/data.json` (5.5 MB, ~1.2 MB gzipped) is built in CI and gitignored rather
than committed.

---

## Run it locally

```bash
pip install -r requirements.txt
python simple/app.py     # the officer's view   -> http://127.0.0.1:8050
python run.py            # the analyst console  -> http://127.0.0.1:8000
```

Either builds its own data on first run and then starts instantly. No Node, no
build step, no `causalml` — numpy, pandas, scikit-learn and FastAPI.

| | **simple/** — officer view | **api/** — analyst console |
|---|---|---|
| Data | 64,000 **real** customers (Hillstrom) | 120,000 simulated, six-arm |
| Audience | Someone running the campaign | Someone running the models |
| Tabs | The model · Products & offers · Customers · The ML | Six analyst screens |
| Answers | What do I do on Monday? | Which model, which offer, what budget? |

Both read `src/nextbest/`. After changing anything there, rebuild both:

```bash
python simple/realbuild.py    # officer view, real data
python -m nextbest.train      # analyst console, simulated
python -m pytest tests -q     # 22 tests
```

---

## The four decisions

| Group | Meaning | Action |
|---|---|---|
| **Email them** | Contact genuinely changes what they do | Send |
| **No email needed** | They were coming back regardless | Send content, not a coupon |
| **Not worth emailing** | No response either way | Skip |
| **Leave them alone** | They would have come back; contact puts them off | Suppress |

A propensity model cannot tell the last two apart from the first — all three look
like "likely buyer". Only a model of *change* returns a negative score.

**Which of the two emails they get is constrained.** The men's arm lifted more
overall (7.66pp vs 4.52pp), so picking the per-customer argmax sent the men's
campaign to 70% of womenswear buyers. That is defensible statistically and
indefensible commercially. The offer now follows what the customer actually
buys unless the other arm beats it by more than two points — 91% of the list
matches its merchandise, and tab 2 flags the split if it drops below 75%.

---

## What the real data says

Measured against the 21,306 customers held back at random:

| | |
|---|---|
| The campaign caused | **+6.09 points** of extra visits |
| An incremental visit is worth | **$9.80** revenue → $4.41 margin |
| **So one contact is worth** | **$0.269** — the break-even |

Any way of reaching a customer costing more than **$0.27**
loses money if sent to everyone. At $0.08 an email this campaign already pays for
itself and **you do not need a model**. The app says so on screen. Targeting only
earns its keep once a contact costs more than the margin it creates.

### The honest result

On this dataset the recommended model's advantage over the baseline is **not statistically distinguishable**. Its Qini interval [7, 115] overlaps the baseline's [-3, 53], and the winning margin is $-0.00134 per customer — which is noise, not a result. Hillstrom is almost certainly too small and too weak-signal to support per-customer targeting. Reporting that is a stronger finding than claiming a win.

That is on the front page of the app, above the numbers it qualifies. A tool that
reports a null result honestly is worth more than one that manufactures a win.

The same applies to the verdicts. Beating a worse option is not the same as
being worth doing, so a list that still loses money returns **stop**, not
"target" — and so does a direct-mail list of 34 people, however good the
per-head economics look. `verdict_for()` and `choose_offer()` are pinned by
tests; `app.js` mirrors them for the hosted build.

---

## Economics, stated openly

- **45%** gross margin on an order
- **$0.08** to send one email — the campaign as it ran
- **$0.30** for the model comparison, a labelled what-if

Both prices are on screen with their labels. Two prices are fine; showing both
without saying which is which is not.

The discount is charged against **P(buy | contacted)**, not against uplift. A
coupon is redeemed by everyone who buys, not only by the extra sales it caused —
charging it to incremental buyers alone hid the entire cost of discounting
someone who was going to purchase regardless. That was a real bug, and there is
now a test for it.

---

## The data is real. The simulator is a test fixture.

`simple/` runs on **Hillstrom MineThatData (2008)** — 64,000 real customers of a
real retailer, randomly split into men's email, women's email and no email. The
random split is what makes the difference between groups a causal effect rather
than a correlation.

There is no accuracy column, and that is the point: on real customers you see
what someone did after being emailed, never what they would have done otherwise.
So models are ranked by what their decisions actually earned on held-out
customers — offline policy evaluation — which is the number a business cares
about anyway.

`src/nextbest/simulate.py` plants a known effect per customer. That makes it
useless as a product and ideal as a **test fixture**: `tests/` asserts that the
four uplift learners recover the planted effect and that the propensity model
does not. Neither check is possible on real data.

For Kaggle sets (X5 RetailHero, Criteo-UPLIFT) put `kaggle.json` in `~/.kaggle/`
and use `realdata.kaggle_download`. Any dataset works provided it has a
**randomly assigned** treatment flag. Without a held-back group there is no
counterfactual, and no modelling recovers one.

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
│   ├── simulate.py       known-effect generator — used by tests, not the product
│   ├── learners.py       S / T / X / Transformed Outcome + propensity baseline
│   ├── evaluate.py       Qini, decile lift, profit curve, bootstrap CIs
│   └── optimize.py       offer choice + budget allocation
├── tests/                22 tests, run in CI on the simulator
├── build_static.py       -> docs/  for GitHub Pages
└── run.py                analyst console launcher
```

---

## Known limitations

- **No randomized holdout at most retailers.** This is organizational, not
  technical, and it gates the whole product. Without one, none of this runs.
- **The prize is small here.** -0.00134 per customer, and the
  confidence intervals overlap. Hillstrom is likely too small and too weak-signal
  to support per-customer targeting — see the banner on tab 1.
- **No per-customer uncertainty.** Needs a Causal Forest; shipping fake error
  bars would be worse than none.
- **No feature store, no activation, no feedback loop.** Point-in-time features,
  an ESP/CDP connector, and campaign → holdout → realised lift → retrain are all
  designed in `NextBest_Project_Plan.md` and not built. That is most of a real
  deployment.
- **Attribution is not SHAP** — counterfactual median substitution, labelled as
  such in the code and on screen.
