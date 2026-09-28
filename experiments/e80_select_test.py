"""E80 test stage: for each information set (own stream; + perpetual and ETH), the epoch with the best day-5 likelihood
of the val run, then train on days 1-5 for that many epochs and read days 6-7 once (likelihood and the edge audit with
the policy chosen on day 5)."""
import glob
import json
import os
import subprocess
import sys

RES = os.path.join(os.path.dirname(__file__), "results", "e80"); rc = 0
for p in sorted(glob.glob(os.path.join(RES, "tv_market_val_*.json"))):
    r = json.load(open(p)); a = r["args"]
    print(json.dumps({"val": os.path.basename(p), "best_epoch": r["best_epoch"], "best_day5": r["best_day5"], "policy": r["policy"]}), flush=True)
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "e80_tv_market.py"), "--mode", "test", "--epochs", str(r["best_epoch"])]
    for k in ("L", "fine", "cross", "mag", "d", "M", "lr", "batch", "fees"):
        cmd += [f"--{k}", str(a[k])]
    rc |= subprocess.call(cmd)
sys.exit(rc)
