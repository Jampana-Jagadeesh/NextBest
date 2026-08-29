#!/usr/bin/env python
"""Build a static, hostable copy of the app into docs/ for GitHub Pages.

GitHub Pages serves files, not Python. Everything the app does turns out to be
either a lookup or arithmetic, so all of it moves to the browser:

    /api/overview   -> a JSON bundle
    /api/customers  -> filter and sort an array
    /api/customer   -> find in that array
    /api/recompute  -> lift x margin - cost, the same formula the server used
    /api/export     -> build the CSV client-side and hand it to the browser

The one exception is /api/whatif, which needs the fitted gradient-boosting
models. That feature is hidden in the static build rather than faked, with a
note pointing at the local app.

    python build_static.py
    -> docs/  (set Pages to deploy from the docs/ folder on main)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
SIMPLE = ROOT / "simple"
OUT = ROOT / "docs"

# Only the columns the interface actually renders. Shipping the rest would
# double the payload for data nothing reads.
COLS = ["customer_id", "months_since_purchase", "spend_12m", "spend_band",
        "buys_mens", "buys_womens", "new_customer", "area", "channel",
        "category", "extra_sales_pp", "buys_alone_pct", "buys_if_contacted_pct",
        "value_of_contact", "cost_of_contact", "action", "reward",
        "arm", "visited", "spent"]


def main() -> None:
    for f in ("metrics.json", "customers.pkl", "artifacts.joblib"):
        if not (SIMPLE / f).exists():
            raise SystemExit(f"missing simple/{f} -- run: python simple/realbuild.py")

    metrics = json.loads((SIMPLE / "metrics.json").read_text(encoding="utf-8"))
    cust = pd.read_pickle(SIMPLE / "customers.pkl")
    art = joblib.load(SIMPLE / "artifacts.joblib")

    OUT.mkdir(exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(SIMPLE / "static" / name, OUT / name)

    # Only the static build has a data.json. app.js checks for this marker so the
    # server build does not fetch a file that is not there and log a 404.
    html = (OUT / "index.html").read_text(encoding="utf-8")
    marker = "<script>window.NB_STATIC=1</script>"
    assert marker not in html
    html = html.replace("</head>", f"{marker}\n</head>", 1)

    # FastAPI mounts these at /static/, but here they sit beside index.html and
    # the site is served from a subpath (/NextBest/), so an absolute /static/...
    # resolves to the domain root and 404s -- taking the CSS and the JS with it.
    before = html
    html = html.replace(chr(34) + "/static/", chr(34)).replace("=/static/", "=")
    assert html != before, "expected the /static/ asset paths to rewrite"
    assert "/static/" not in html, "an absolute /static/ path survived"

    # Assets are served with max-age=600 too, so a browser can hold a stale
    # stylesheet after a deploy and render the new HTML against old CSS.
    # Stamping the content hash into the URL makes every deploy self-healing.
    import hashlib
    def _v(name):
        return hashlib.sha1((OUT / name).read_bytes()).hexdigest()[:8]
    html = html.replace('href="styles.css"', f'href="styles.css?v={_v("styles.css")}"')
    html = html.replace('src="app.js"', f'src="app.js?v={_v("app.js")}"')
    (OUT / "index.html").write_text(html, encoding="utf-8")

    slim = cust[COLS].copy()
    for c in ("spend_12m", "extra_sales_pp", "buys_alone_pct",
              "buys_if_contacted_pct", "spent"):
        slim[c] = slim[c].round(2)
    slim["value_of_contact"] = slim["value_of_contact"].round(4)

    # Seven string columns repeated 64,000 times were 60% of the payload.
    # Dictionary-encode them: the browser decodes on read.
    STRINGS = ["spend_band", "area", "channel", "category", "action", "reward", "arm"]
    dictionary: dict[str, list[str]] = {}
    for c in STRINGS:
        vals = sorted(slim[c].astype(str).unique())
        dictionary[c] = vals
        lookup = {v: i for i, v in enumerate(vals)}
        slim[c] = slim[c].astype(str).map(lookup).astype("int16")

    E = art["eval"]
    bundle = {
        "metrics": metrics,
        "columns": COLS,
        "dict": dictionary,
        # rows as arrays, not objects: the same data at a third of the bytes
        "rows": json.loads(slim.to_json(orient="values")),
        "eval": {
            "uplift": [round(float(x), 5) for x in E["uplift"]],
            "t": [int(x) for x in E["t"]],
            "y": [int(x) for x in E["y"]],
            "rev_per_incremental_visit": float(E["rev_per_incremental_visit"]),
            "lift_all": float(E["lift_all"]),
            "n_base": int(E["n_base"]),
        },
        "base": {
            "uplift": [round(float(x), 4) for x in art["base_uplift"]],
            "p_control": [round(float(x), 4) for x in art["base_pctrl"]],
        },
        "static": True,
    }

    data = OUT / "data.json"
    data.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")

    # Pages would otherwise run the output through Jekyll and drop some files.
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    mb = data.stat().st_size / 1024 / 1024
    print(f"[static] docs/data.json  {mb:.1f} MB  ({len(bundle['rows']):,} customers)")
    print(f"[static] docs/           index.html, styles.css, app.js, .nojekyll")
    print(f"[static] serves gzipped at roughly {mb * 0.22:.1f} MB")
    print("[static] test locally:  python -m http.server -d docs 8080")


if __name__ == "__main__":
    main()
