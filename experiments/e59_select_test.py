"""E59 test stage: for absolute and for relative band coding, pick the configuration with the best held-out-speaker
validation accuracy (selection on validation only), then score the test set once."""
import glob
import json
import os
import subprocess
import sys

RES = os.path.join(os.path.dirname(__file__), "results", "e51")
rc = 0
for rel in (0, 1):
    runs = []
    for p in glob.glob(os.path.join(RES, "*_spk.json")):
        r = json.load(open(p))
        if int(r["args"].get("rel", 0)) == rel:
            runs.append((r["spk_acc"], r["args"]))
    acc, args = max(runs, key=lambda x: x[0])
    print(json.dumps({"rel": rel, "selected": {k: args[k] for k in ("B", "O", "a", "timing")}, "spk_acc": acc}), flush=True)
    rc |= subprocess.call([sys.executable, os.path.join(os.path.dirname(__file__), "e51_shd_world.py"), "--B", str(args["B"]),
                           "--O", str(args["O"]), "--a", str(args["a"]), "--rel", str(rel), "--eval", "test"])
sys.exit(rc)
