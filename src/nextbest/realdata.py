"""Real randomised-experiment data.

The simulator exists to measure estimator ERROR (only possible when the true
effect is known). This module exists to prove the same pipeline survives contact
with a genuine RCT, where the signal is weak, the features are thin, and nothing
can be validated against ground truth.

Hillstrom MineThatData (2008) is the anchor: 64,000 customers randomised across
three arms -- men's email, women's email, no email -- with visit, conversion and
spend recorded over the following two weeks. Public, no credentials, no licence
friction.

    python -m nextbest.realdata

Kaggle sources (X5 RetailHero, Criteo-UPLIFT, Lenta) need an API token. Drop
kaggle.json in ~/.kaggle/ and use `kaggle_download` below; without it the CLI
cannot authenticate and this module falls back to Hillstrom.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import evaluate
from .config import DATA_DIR, MODEL_DIR, SEED
from .learners import LEARNERS

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_CSV = DATA_DIR / "hillstrom.csv"


def _log(msg: str) -> None:
    print(f"[realdata] {msg}", flush=True)


# --------------------------------------------------------------- acquisition
def download_hillstrom(force: bool = False) -> Path:
    if HILLSTROM_CSV.exists() and not force:
        return HILLSTROM_CSV
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"downloading Hillstrom from {HILLSTROM_URL}")
    urllib.request.urlretrieve(HILLSTROM_URL, HILLSTROM_CSV)
    return HILLSTROM_CSV


def kaggle_download(dataset: str, dest: Path | None = None) -> Path:
    """Fetch a Kaggle dataset, e.g. kaggle_download('davinwijaya/customer-retention').

    Requires the `kaggle` package and ~/.kaggle/kaggle.json. Raises with a clear
    message rather than a stack trace when either is missing, because that is
    the failure every reader of this repo will hit first.
    """
    dest = dest or (DATA_DIR / dataset.split("/")[-1])
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "  1. kaggle.com -> Settings -> API -> Create New Token\n"
            f"  2. save the downloaded kaggle.json to {token}\n"
            "  3. pip install kaggle\n"
            "Until then the pipeline uses Hillstrom, which needs no credentials."
        )
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    _log(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


# ------------------------------------------------------------------ loading
def load_hillstrom(outcome: str = "visit") -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, treatment, y, raw).

    Outcome defaults to `visit`. `conversion` is the commercially interesting
    label but fires on only ~0.9% of customers, which leaves too little signal
    to estimate heterogeneous effects on 64k rows -- a real and instructive
    constraint, not a shortcut.
    """
    path = download_hillstrom()
    raw = pd.read_csv(path)

    t = (raw["segment"] != "No E-Mail").astype(int).to_numpy()
    y = raw[outcome].astype(int).to_numpy()

    X = pd.DataFrame(index=raw.index)
    X["recency"] = raw["recency"].astype(float)
    X["history"] = np.log1p(raw["history"].astype(float))
    X["mens"] = raw["mens"].astype(float)
    X["womens"] = raw["womens"].astype(float)
    X["newbie"] = raw["newbie"].astype(float)
    # history_segment looks like "2) $100 - $200" -- the leading digit is ordinal
    X["history_segment"] = raw["history_segment"].str.extract(r"^(\d)").astype(float)
    for z in ("Surburban", "Rural", "Urban"):
        X[f"zip_{z.lower()}"] = (raw["zip_code"] == z).astype(float)
    for ch in ("Phone", "Web", "Multichannel"):
        X[f"channel_{ch.lower()}"] = (raw["channel"] == ch).astype(float)

    return X.astype(float), t, y, raw


# ---------------------------------------------------------------- bake-off
def run(outcome: str = "visit", seed: int = SEED) -> dict:
    t_start = time.time()
    X, t, y, raw = load_hillstrom(outcome)
    _log(f"{len(X):,} customers | treated {t.mean():.1%} | {outcome} rate {y.mean():.2%}")

    tr, te = train_test_split(np.arange(len(X)), test_size=0.30,
                              random_state=seed, stratify=t)

    # crude but honest economics for the profit curve on this dataset
    avg_margin, offer_cost = 30.0, 3.0

    results = []
    for key, cls in LEARNERS.items():
        t0 = time.time()
        m = cls().fit(X.iloc[tr], t[tr], y[tr])
        pred = m.predict(X.iloc[te])
        # NOTE: no ground truth is possible here -- that is the entire point
        res = evaluate.evaluate_all(pred, t[te], y[te], None, avg_margin, offer_cost)
        results.append({
            "key": key, "name": cls.name, "is_uplift": key != "propensity",
            "qini": res["qini"], "qini_ci": res["qini_ci"],
            "top_decile_uplift": res["top_decile_uplift"],
            "bottom_decile_uplift": res["bottom_decile_uplift"],
            "deciles": res["deciles"], "curve": res["curve"],
            "optimal_fraction": res["profit"]["optimal_fraction"],
            "fit_seconds": round(time.time() - t0, 2),
        })
        _log(f"  {cls.name:22s} qini={res['qini']:8.1f} "
             f"[{res['qini_ci']['lo']:.0f}, {res['qini_ci']['hi']:.0f}]  "
             f"top={res['top_decile_uplift']:+.2f}pp bot={res['bottom_decile_uplift']:+.2f}pp")

    uplift_only = [r for r in results if r["is_uplift"]]
    champion = max(uplift_only, key=lambda r: r["qini"])

    # observed arm effect -- the sanity check that the experiment worked at all
    arms = []
    for seg in raw["segment"].unique():
        sub = raw[raw["segment"] == seg]
        arms.append({"arm": seg, "n": int(len(sub)),
                     "rate": round(float(sub[outcome].mean()) * 100, 3)})
    base = next(a["rate"] for a in arms if a["arm"] == "No E-Mail")
    for a in arms:
        a["lift_pp"] = round(a["rate"] - base, 3)

    out = {
        "source": "Hillstrom MineThatData 2008 (real randomised email experiment)",
        "url": HILLSTROM_URL,
        "outcome": outcome,
        "n": int(len(X)),
        "treated_share": round(float(t.mean()) * 100, 2),
        "outcome_rate": round(float(y.mean()) * 100, 3),
        "features": list(X.columns),
        "arms": arms,
        "results": results,
        "champion": {"key": champion["key"], "name": champion["name"]},
        "has_ground_truth": False,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t_start, 1),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "hillstrom_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _log(f"champion on real data: {champion['name']} -> models/hillstrom_metrics.json")
    return out


if __name__ == "__main__":
    run()
