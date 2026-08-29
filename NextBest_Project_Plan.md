# NextBest — Uplift-Based Next-Best-Offer Engine

**End-to-end ML project plan**

| | |
|---|---|
| **Duration** | 8–10 weeks |
| **Core technique** | Uplift modeling / CATE (Conditional Average Treatment Effect) |
| **Verticals** | Retail · Hotels · Airlines · Fashion |
| **Stack** | Python · dbt · FastAPI · React |
| **Headline metric** | Incremental profit |

---

## 1. Why this is not another propensity model

This distinction is the whole project. If you can explain it in two sentences, you have already beaten most portfolios.

### What everyone builds — Propensity model

```
score = P(buy | contacted)
```

Ranks customers by how likely they are to convert after a campaign. The top of that list is dominated by loyal customers who were going to buy anyway. You hand them a 20% discount and burn margin on revenue you already had.

### What you build — Uplift model

```
score = P(buy | contacted) − P(buy | not contacted)
```

Ranks customers by how much the contact *changes* their behavior. The top of this list is people whose decision genuinely hinges on the offer. Every conversion you buy here is incremental — it would not have happened otherwise.

> **The one-line pitch**
> Propensity answers *"who will buy?"* Uplift answers *"who will buy only if I ask?"* The second question is the one the marketing budget is actually trying to answer, and almost nobody models it.

---

## 2. The four customers hiding in your database

Every customer sits in one of four boxes, defined by what they do when treated versus untreated. You only ever observe one of those two worlds per person — that is the fundamental problem of causal inference, and it shapes every design decision downstream.

|  | **Would NOT buy on their own** | **Would buy on their own** |
|---|---|---|
| **Buys when contacted** | **PERSUADABLES**<br>Buy only when contacted. The offer tips them over. Pure incremental revenue.<br>→ *Target — this is the product* | **SURE THINGS**<br>Buy either way. Contacting them changes nothing except your margin.<br>→ *Skip — pure margin leak* |
| **Does not buy when contacted** | **LOST CAUSES**<br>Never buy, treated or not. Wasted contact cost and inbox fatigue.<br>→ *Skip — wasted spend* | **SLEEPING DOGS**<br>Would have bought, but the contact drives them away.<br>→ *Never contact — negative uplift* |

**Why sleeping dogs matter more than they sound.** A propensity model cannot see them at all — it scores a sleeping dog and a sure thing identically, because both look like "likely buyer." Only an uplift model produces a *negative* score and tells you to back off. In hotels and airlines this is very real: unsolicited discount mail teaches a high-value business traveler to wait for the discount.

---

## 3. Data strategy — solve this before you write model code

Uplift modeling has one hard prerequisite that trips up most attempts: **you need data where treatment was assigned randomly.** Without a control group there is no counterfactual, and no amount of modeling recovers it. Decide your data source in week one.

| Source | What it gives you | Size | Use it for |
|---|---|---|---|
| **Criteo-UPLIFT v2** | Genuine large-scale randomized ad experiment; treatment flag, visit + conversion labels, 12 anonymized features | ~13.9M rows | Primary real-RCT anchor. Proves the pipeline works at scale |
| **Hillstrom MineThatData** | The classic teaching set — men's email / women's email / no email, with visit, conversion and spend | 64,000 | Fast iteration, three-arm multi-treatment, readable features |
| **X5 RetailHero** | Real grocery retail RCT with full transaction history | ~200k | The retail vertical, rich RFM feature engineering |
| **Lenta** | Retail SMS campaign with control group, ~190 features | ~690k | Wide-feature stress test, feature selection practice |
| **Starbucks Portfolio** | Multiple offer types — BOGO, discount, informational — with view and completion events | ~300k events | The multi-treatment offer-selection layer |
| **Your own simulator** | Synthetic customers with *known ground-truth uplift* per person, for hotels / airlines / fashion | any | Validating the estimator itself, demoing the other three verticals |

### The senior move

Build the simulator **even though you have real data.** On real data you can never check whether your per-person uplift estimate is *correct* — you only ever see one outcome per customer. On simulated data you know the true individual treatment effect, so you can measure your estimator's actual error and compare T-Learner vs X-Learner vs Causal Forest against ground truth.

Then apply the winner to Criteo. That two-track approach — **simulate to validate, real data to prove** — is what separates this from a notebook that calls `causalml` and plots a curve.

### If you only have observational data

Sometimes there is no control group — marketing contacted whoever they felt like. You can still estimate uplift, but you must correct for the fact that treatment was not random: fit a propensity-to-be-treated model, then use inverse-propensity weighting or a doubly-robust learner (`DR-Learner`, `EconML`). State the unconfoundedness assumption explicitly and run a sensitivity analysis. Being upfront that this is weaker evidence than an RCT is itself a senior signal.

---

## 4. System architecture

A closed loop, not a one-way pipeline. Every campaign you send holds out a control group, which becomes the training data for the next model. **That feedback edge is the part most projects forget.**

```
[01 SOURCE]        transactions, customers, campaign_log
     |
[02 INGEST]        Python / Airbyte  ->  Parquet  ->  warehouse
     |
[03 MODEL]         dbt: staging -> intermediate -> marts  (+ tests, lineage)
     |
[04 FEATURES] *    point-in-time correct feature table, as-of joins
     |
[05 TRAIN]    *    causalml + Optuna + MLflow
     |
[06 OPTIMIZE]      argmax offer per customer, knapsack under budget
     |
[07 SERVE]         FastAPI: batch scores + realtime /score
     |
[08 CONSUME]       React console: audience builder + experiment ledger
     |
     +--------> FEEDBACK LOOP: campaign sent with 10% holdout
                -> realized lift measured
                -> appended to training set
                -> back to step 03
```

---

## 5. The build, phase by phase

Ordered by dependency, not by interest. **Resist jumping to Phase 4** — the reason Phase 3 exists is so you have a wrong-but-plausible baseline to beat, which is what makes the final result legible to a non-technical audience.

### Phase 0 — Frame the decision, not the model *(Week 1)*

Write the one-page problem statement first. Who gets contacted, with what offer, under what budget, and what does a win look like in money? Everything downstream is measured against this page.

- Define the unit economics: contact cost, discount cost, average margin per conversion
- Write the profit function you will optimize — this becomes your model selection metric
- Pick datasets, and run a statistical power calculation for the control group size
- Repo scaffold: `src/`, `dbt/`, `api/`, `ui/`, `notebooks/`, Makefile, pre-commit, Docker Compose

**Ship:** `PROBLEM.md` + running repo skeleton
**Skill:** Problem framing, unit economics

---

### Phase 1 — Data engineering foundation *(Week 1–2)*

Build the warehouse properly. This is the half of the project that most ML portfolios skip, and it is the half that gets a data engineer hired.

- Ingestion scripts landing raw files to Parquet with schema validation (`pandera`)
- Warehouse: DuckDB locally, Postgres or Redshift deployed — identical dbt code either way
- dbt layers: `staging` (typed, renamed) → `intermediate` (joined) → `marts` (`dim_customer`, `fct_transaction`, `fct_campaign_contact`)
- dbt tests on every mart: uniqueness, not-null, accepted values, plus a custom test asserting treatment/control balance
- Orchestration with Dagster or Airflow — one DAG, daily schedule, retries

**Ship:** dbt docs site + green test suite
**Skill:** dbt, dimensional modeling, orchestration, data contracts

---

### Phase 2 — Point-in-time correct features *(Week 2–3)*

The single highest-risk step for silent failure. Every feature must be computed strictly from data available *before* the campaign was sent, or your offline metrics will be beautiful and your online results garbage.

- **RFM block** — recency, frequency, monetary, tenure, inter-purchase gap and its variance
- **Trend block** — spend last 30/90/365d, ratio of recent to historical, category mix drift
- **Engagement block** — opens, clicks, sessions, app logins, prior offer redemption rate
- **Vertical block** — nights and lead time (hotel), routes and cabin mix (air), size returns and season affinity (fashion)
- **Sensitivity block** — prior discount depth used, share of purchases made on promotion. *These are the features that actually drive uplift*
- Enforce as-of joins with an explicit `feature_ts < campaign_ts` assertion in dbt

**Ship:** Versioned feature mart + leakage test
**Skill:** Feature engineering, temporal joins, leakage prevention

---

### Phase 3 — The deliberately wrong baseline *(Week 3)*

Build the propensity model everyone else builds. You need it to demonstrate the failure mode — and later, the improvement.

- LightGBM classifier on the treated group only, predicting conversion
- Rank customers, take the top decile, then measure what the *control* group in that same decile did
- Show the punchline: the highest-propensity decile has near-zero incremental lift, because it is full of sure things

**Ship:** The chart that justifies the entire project
**Skill:** Baselines, calibration, honest evaluation

---

### Phase 4 — Uplift models: build a ladder, not one model *(Week 4–5)*

Fit the family in increasing sophistication and compare them on the same holdout. Knowing *why* X-Learner beats T-Learner on imbalanced treatment groups is the depth an interviewer probes for.

| Model | How it works | When it wins |
|---|---|---|
| **T-Learner** | Separate models on treated and control, then subtract | Simple baseline; high variance |
| **S-Learner** | One model with treatment as a feature | Can wash out the effect entirely if the tree ignores it |
| **X-Learner** | Imputes the effect and cross-fits | Markedly better when treatment groups are very unequal in size |
| **R-Learner / DR-Learner** | Residualized and doubly-robust | The modern default (`EconML`) |
| **Uplift RF / Causal Forest** | Trees split directly on divergence in treatment response | Non-linear heterogeneity |
| **Class Transformation** (Lai / Kane) | Reframes as a single classification problem | Fast and surprisingly strong |

- Tune with Optuna against **Qini**, never accuracy
- Track every run in MLflow

**Ship:** Model bake-off table + registered champion
**Skill:** Meta-learners, causal ML, experiment tracking

---

### Phase 5 — Evaluation and the money question *(Week 5–6)*

Uplift cannot be scored with accuracy or AUC, because the per-person label does not exist. You evaluate by binning customers and comparing treated to control *within* each bin.

- Qini curve and Qini coefficient — the field-standard ranking metric
- Uplift curve and AUUC, plus an uplift-per-decile bar chart
- Bootstrap confidence intervals on Qini — a single point estimate on a small control group is close to meaningless
- Translate to money: `expected incremental profit = (uplift × margin) − contact cost − discount cost`
- Find the optimal cutoff: the point where marginal profit hits zero. **That number is the deliverable**

**Ship:** Evaluation module + profit curve with optimal k
**Skill:** Causal evaluation, bootstrap CIs, decision theory

---

### Phase 6 — Multi-offer selection and budget allocation *(Week 6–7)*

This is what makes it "next-best-offer" instead of "uplift score." One model per offer, then choose per customer, then allocate under a real constraint.

- Train an uplift model per offer arm: 10% off, free shipping, 2× points, room upgrade, free checked bag
- Per customer, pick `argmax` over offers of `(uplift × margin − offer cost)`
- Budget-constrained assignment: greedy by incremental profit per unit spend, then compare against an LP / knapsack optimum
- Guardrails: contact frequency caps, suppression of negative-uplift customers, fairness check across segments
- *Stretch* — Thompson sampling bandit for online offer exploration

**Ship:** Offer optimizer with budget slider
**Skill:** Multi-treatment CATE, constrained optimization

---

### Phase 7 — Serving layer *(Week 7–8)*

Two paths, because campaigns are batch but the site is realtime.

- **Batch:** nightly job scores the full base, writing `customer_id, offer, uplift, expected_profit, quadrant` to a serving table
- **Realtime:** FastAPI `POST /score` loading the MLflow model, Pydantic request/response schemas, Redis cache for feature lookups
- `GET /audience` returns an optimized target list for a given budget and offer set
- OpenAPI docs, health checks, structured logging, p95 latency under 100 ms
- Containerize; `docker compose` brings up warehouse, API and UI in one command

**Ship:** Dockerized API with OpenAPI spec
**Skill:** FastAPI, model serving, containerization

---

### Phase 8 — The console *(Week 8–9)*

Six screens. The Audience Builder is the one that sells the whole thing, so build it first and build it well.

- React + Vite + TypeScript, TanStack Query for data, Tailwind + shadcn/ui for components
- Recharts or visx for the Qini curve, decile bars, and offer heatmap
- Dark and light themes, keyboard-accessible tables, skeleton loading states
- *Fast path:* Streamlit gets 70% of this in 20% of the time — but a real React console is the differentiator, so budget for it

**Ship:** Deployed console
**Skill:** React, data visualization, product thinking

---

### Phase 9 — MLOps and the feedback loop *(Week 9–10)*

Close the loop. A model that cannot be retrained on its own results is a demo, not a system.

- Every campaign launched from the UI automatically reserves a randomized holdout — non-negotiable, and it is what generates next month's training data
- Experiment ledger table: predicted lift, realized lift, confidence interval, verdict
- Drift monitoring with Evidently: feature PSI, prediction distribution, realized-Qini decay
- CI in GitHub Actions: lint, unit tests, dbt build, model smoke test, container build
- Scheduled retraining with automatic champion/challenger promotion gated on Qini

**Ship:** Monitoring dashboard + CI pipeline
**Skill:** MLOps, drift detection, experiment design

---

## 6. How you know it works

The Qini curve is to uplift what ROC is to classification. Sort customers by predicted uplift, walk down the list, and plot cumulative incremental conversions against the fraction of the base targeted.

```
lift |          ,-*-.
     |        ,'  |  `-.                 * = optimal cutoff k
     |      ,'    |     `-._
     |    ,'      |         `-.___       --- uplift model
     |  ,'    _.-''''----....___  `--    ... propensity baseline
     |,'_.-'''                     ``    ___ random targeting (diagonal)
     +----------------------------------
       fraction of base targeted ->
```

The area between your model and the diagonal is the **Qini coefficient**. The peak is the cutoff where you should stop contacting — past it, you are spending budget on sure things and starting to wake sleeping dogs.

| Metric | What it measures | Report it to |
|---|---|---|
| **Qini coefficient** | Area between your uplift curve and random targeting | Data science |
| **AUUC** | Area under the uplift curve; cousin of Qini, different normalization | Data science |
| **Uplift @ top decile** | Incremental conversion rate among your top 10% vs the base rate | Campaign managers |
| **Incremental profit** | `(uplift × margin) − contact cost − discount cost`, summed over targets | Finance — this is the one that matters |
| **Optimal target %** | The cutoff maximizing profit. Often 30–60% of the base, not 100% | Everyone |
| **Sleeping dogs avoided** | Count of negative-uplift customers correctly suppressed | Retention team |

> **Do not report accuracy, F1, or AUC-ROC.**
> There is no per-customer uplift label to be accurate against — you observe treated *or* control, never both. Any project reporting classification accuracy on an uplift model has misunderstood the problem, and a reviewer will spot it instantly.

---

## 7. The console — six screens

Design it as an operating tool, not a report. A campaign manager should be able to go from "I have a $50,000 budget" to an exported target list in under a minute.

### Screen 01 — Campaign Command
The landing view. Answers "is this working?" in three seconds.
- KPI row: incremental revenue, persuadables reached, budget used, lift vs random
- Qini curve as the hero chart
- Live quadrant population breakdown
- Active campaigns with realized-vs-predicted lift

### Screen 02 — Audience Builder *(the centerpiece)*
The interaction that sells the project. Drag a budget slider and watch the audience, quadrant mix and projected profit recompute live.
- Budget slider → auto-selects top-k by incremental profit per dollar
- Segment, vertical and tier filters
- Quadrant composition bar updating in real time
- "Excluded: 4,182 sleeping dogs" callout
- Export to CSV or push to the campaign tool

### Screen 03 — Customer 360
Per-customer explanation, so the recommendation is auditable rather than a black box.
- Uplift score with confidence interval and quadrant badge
- Counterfactual bars: `P(buy | treated)` vs `P(buy | control)`
- SHAP driver bars for the **uplift**, not the propensity
- Recommended offer with expected incremental margin
- Transaction and contact timeline

### Screen 04 — Offer Matrix
Where multi-treatment modeling becomes visible at a glance.
- Heatmap of segment × offer, cells colored by mean uplift
- Diverging scale so negative uplift is unmistakable
- Click a cell to drill into that cohort
- Cost per incremental conversion, per offer

### Screen 05 — Experiment Ledger
The screen that proves you understand causal work rather than just modeling it.
- Every campaign with its holdout size and design
- Predicted lift vs realized lift with confidence intervals
- Statistical significance verdict per campaign
- Calibration scatter accumulating over time

### Screen 06 — Model Health
Operational truth about the deployed model.
- Qini trend across retraining runs
- Feature drift (PSI) with threshold alerts
- Prediction distribution shift
- Champion vs challenger comparison

---

## 8. Stack, by layer

Chosen for defensibility rather than novelty. Every row has a reason you can state out loud.

| Layer | Choice | Why | Alternative |
|---|---|---|---|
| Storage | **DuckDB** local, **Postgres / Redshift** deployed | Same SQL, zero-setup locally, credible in production | Snowflake, BigQuery |
| Transformation | **dbt-core** | Version-controlled SQL, built-in testing, auto docs and lineage | SQLMesh |
| Orchestration | **Dagster** | Asset-based model fits ML lineage better than task-based DAGs | Airflow, Prefect |
| Validation | **Pandera** + dbt tests | Schema contracts at both the Python and SQL boundary | Great Expectations |
| Features | dbt-built feature mart | A dedicated feature store is over-engineering at this scale — say so deliberately | Feast |
| Base learner | **LightGBM** | Fast, handles categoricals and missingness, standard tabular workhorse | XGBoost, CatBoost |
| Uplift | **causalml** + **scikit-uplift** | causalml for meta-learners and causal forests, sklift for Qini and plotting | EconML, DoWhy |
| Tuning | **Optuna** | Accepts a custom Qini objective; pruning saves real time | Hyperopt |
| Tracking | **MLflow** | Experiments, model registry and champion/challenger in one place | Weights & Biases |
| Explainability | **SHAP** | Per-customer reason codes — explain the uplift model, not the propensity model | Captum |
| API | **FastAPI** + Pydantic | Typed contracts, free OpenAPI docs, async performance | Flask, BentoML |
| Frontend | **React + Vite + TypeScript** | Type safety across the API boundary; the differentiating deliverable | Streamlit (fast path) |
| Charts | **Recharts** or **visx** | Recharts for speed; visx when the Qini curve needs custom treatment | ECharts, Plotly |
| Monitoring | **Evidently** | Purpose-built drift reports that drop straight into the health screen | WhyLabs, NannyML |
| Delivery | **Docker Compose** + GitHub Actions | One command to run everything; CI proves it stays working | Kubernetes |

---

## 9. What you will actually be able to claim

Mapped to how a hiring manager reads a CV. **The causal inference column is the scarce one** — very few candidates have it.

### Data engineering
- **Dimensional modeling** — fact and dimension design for customer, transaction and campaign events
- **dbt** — layered transformations, tests, macros, docs, lineage
- **Orchestration** — scheduled DAGs with retries, backfills, asset dependencies
- **Point-in-time correctness** — as-of joins and leakage prevention, the hardest DE skill in ML
- **Data contracts** — schema validation at ingestion and transformation boundaries
- **Warehouse SQL** — window functions, incremental models, partition pruning

### Machine learning & causal
- **Causal inference** — potential outcomes, CATE estimation, the fundamental problem
- **Meta-learners** — S, T, X, R and DR learners, and when each one wins
- **Causal forests** — divergence-based splitting and honest trees
- **Experiment design** — randomization, power analysis, holdout sizing
- **Non-standard evaluation** — Qini, AUUC, bootstrap CIs on ranking metrics
- **Constrained optimization** — budget allocation as a knapsack over expected profit
- **MLOps** — registry, drift detection, automated retraining with promotion gates

### Frontend & product
- **React + TypeScript** — typed components against a typed API contract
- **Data visualization** — Qini curves, diverging heatmaps, calibration scatters
- **Interactive analytics UX** — live recomputation under a budget constraint
- **Information design** — surfacing the decision before the detail
- **Accessibility & theming** — keyboard navigation, contrast, light and dark
- **Stakeholder translation** — turning Qini into dollars a CFO will act on

---

## 10. Where this goes wrong

Every one of these has sunk a real uplift project. Read them before Phase 1, not after Phase 8.

| Trap | What happens | Defense |
|---|---|---|
| **No control group** | No counterfactual, so no uplift to model. The project is dead before it starts | Confirm randomized treatment data exists in week 1. If not, use a public RCT or switch to doubly-robust estimation with stated assumptions |
| **Underpowered control** | A 2% holdout produces Qini estimates with CIs wide enough to contain zero | Run a power calculation up front; report bootstrap CIs on every uplift number |
| **Post-treatment leakage** | Features computed after the campaign leak the outcome. Offline Qini looks superb, online performance is random | Assert `feature_ts < campaign_ts` in dbt and fail the build on violation |
| **Scoring like a classifier** | Reporting accuracy or AUC-ROC — a signal you have not understood the problem | Qini, AUUC, uplift-per-decile and incremental profit only |
| **Trusting individual scores** | Uplift estimates are very noisy per person; "+3.7% for this customer" implies precision you do not have | Present deciles and bands in the UI; reserve point estimates for aggregates |
| **Ignoring cost asymmetry** | Waking a sleeping dog costs far more than missing a persuadable, but a symmetric metric treats them the same | Put real costs in the profit function and optimize that, not Qini alone |
| **Overfitting the Qini** | Model selection on a single small validation split picks noise | Repeated CV stratified on the treatment flag, with CIs across folds |
| **Treatment imbalance** | T-Learner variance explodes when control is a tenth the size of treatment | Use X-Learner or DR-Learner, built for exactly this case |

---

## 11. Definition of done

Ship when all of these are true. Not before, and — importantly — not after adding a fourth model nobody asked for.

| Deliverable | Acceptance test |
|---|---|
| **One-command startup** | `docker compose up` brings warehouse, API and UI online with seeded data |
| **Reproducible pipeline** | `make all` runs ingestion → dbt → features → training → evaluation from a clean clone |
| **Beats the baseline** | Uplift model Qini exceeds the propensity model's, with non-overlapping bootstrap CIs |
| **Stated in money** | A named dollar figure for incremental profit at the optimal cutoff versus targeting everyone |
| **Explainable** | Any customer's recommendation traces to features, uplift drivers and expected margin |
| **Loop closed** | Launching a campaign in the UI reserves a holdout and writes a ledger row automatically |
| **Tested** | CI green: linting, unit tests, dbt build, leakage assertion, container build |
| **Explained** | A README that opens with the persuadables-vs-sure-things distinction and the money number |

### If you are short on time

The minimum version that still lands: **Hillstrom dataset, T-Learner and X-Learner, Qini evaluation, propensity baseline for contrast, and the Audience Builder screen alone.** That is roughly three weeks and still demonstrates the causal-inference depth that makes this project worth doing.

---

*NextBest — uplift-based next-best-offer engine — build plan v1*
