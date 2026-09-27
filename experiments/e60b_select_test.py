"""E60b test stage: best held-out-user validation config among the depth runs (Winnow classifiers), then the test once."""
import glob, json, os, subprocess, sys
RES = os.path.join(os.path.dirname(__file__), "results", "e60")
best = None
for p in glob.glob(os.path.join(RES, "depth_*_val.json")):
    r = json.load(open(p))
    for k, v in r.items():
        if k.startswith("winnow_") and not k.endswith("mistakes") and (best is None or v > best[0]):
            best = (v, r["args"], k)
acc, a, key = best
print(json.dumps({"selected": a, "classifier": key, "val_acc": acc}), flush=True)
sys.exit(subprocess.call([sys.executable, os.path.join(os.path.dirname(__file__), "e60b_dvs_depth.py"), "--reg", str(a["reg"]),
                          "--T", str(a["T"]), "--eta", str(a["eta"]), "--tri", str(a.get("tri", 0)), "--eval", "test"]))
