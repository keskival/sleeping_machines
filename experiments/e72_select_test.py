"""E72 test stage: within each hazard family, pick the CDE point-process configuration and epoch count with the best
day-5 validation score (selection on validation only, as in E52), then train on days 1-5 and score days 6-7 once."""
import glob
import json
import os
import subprocess
import sys

RES = os.path.join(os.path.dirname(__file__), "results", "e72")
rc = 0
for fine in (0, 1):
    best = None
    for p in glob.glob(os.path.join(RES, "cde_thp_val_*.json")):
        r = json.load(open(p))
        if r["args"]["epochs"] < 5 or r["args"]["fine"] != fine:
            continue
        e = max(r["epochs"], key=lambda x: x["day5"])
        if best is None or e["day5"] > best[0]:
            best = (e["day5"], r["args"], e["epoch"])
    if best is None:
        continue
    score, args, ep = best
    print(json.dumps({"family": "fine" if fine else "coarse", "selected": args, "epoch": ep, "val_day5": score}), flush=True)
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "e72_cde_thp.py"), "--mode", "test"]
    for k in ("d", "N", "layers", "sel", "L", "fine", "lr"):
        cmd += [f"--{k}", str(args[k])]
    rc |= subprocess.call(cmd + ["--epochs", str(ep)])
sys.exit(rc)
