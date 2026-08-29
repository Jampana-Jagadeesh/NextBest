#!/usr/bin/env python
"""NextBest launcher.

    python run.py                 train if needed, then serve on :8000
    python run.py --retrain       force a fresh simulation + training run
    python run.py --real          also run the Hillstrom RCT bake-off
    python run.py --port 8100     serve elsewhere
    python run.py --train-only    build artifacts and exit
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

ARTIFACTS = [ROOT / "data" / "scored.pkl",
             ROOT / "models" / "artifacts.joblib",
             ROOT / "models" / "metrics.json"]


def main() -> int:
    ap = argparse.ArgumentParser(description="NextBest uplift engine")
    ap.add_argument("--retrain", action="store_true", help="force retraining")
    ap.add_argument("--real", action="store_true", help="also run the Hillstrom RCT bake-off")
    ap.add_argument("--train-only", action="store_true", help="build artifacts, do not serve")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n", type=int, default=None, help="population size override")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    missing = [p for p in ARTIFACTS if not p.exists()]
    if a.retrain or missing:
        if missing and not a.retrain:
            print(f"[run] artifacts missing ({', '.join(p.name for p in missing)}) -- training")
        from nextbest import train
        train.run(**({"n": a.n} if a.n else {}))
    else:
        print("[run] artifacts present -- skipping training (use --retrain to rebuild)")

    if a.real:
        from nextbest import realdata
        realdata.run()

    if a.train_only:
        return 0

    url = f"http://{a.host}:{a.port}/"
    print(f"\n  NextBest console -> {url}\n  Ctrl+C to stop\n")
    if not a.no_browser:
        Timer(1.4, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run("api.main:app", host=a.host, port=a.port, app_dir=str(ROOT), log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
