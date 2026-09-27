"""E52 test stage: pick the THP configuration and epoch count with the best day-5 validation score (selection on
validation only, as preregistered), then train on days 1-5 and score days 6-7 once."""
import glob
import json
import os
import subprocess
import sys

RES = os.path.join(os.path.dirname(__file__), "results", "e52")
rc = 0
for fine in (0, 1):                                        # compare within a hazard family (same window bank)
    best = None
    for p in glob.glob(os.path.join(RES, "thp_val_*.json")):
        r = json.load(open(p))
        if r["args"]["epochs"] < 5 or r["args"]["fine"] != fine:
            continue
        e = max(r["epochs"], key=lambda x: x["day5"])
        if best is None or e["day5"] > best[0]:
            best = (e["day5"], r["args"], e["epoch"])
    score, args, ep = best
    print(json.dumps({"family": "fine" if fine else "coarse", "selected": args, "epoch": ep, "val_day5": score}), flush=True)
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "e52_thp.py"), "--mode", "test", "--d", str(args["d"]),
           "--L", str(args["L"]), "--fine", str(args["fine"]), "--lr", str(args["lr"]), "--epochs", str(ep)]
    rc |= subprocess.call(cmd)
sys.exit(rc)
