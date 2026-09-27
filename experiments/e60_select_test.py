"""E60 test stage: pick the configuration (and classifier) with the best held-out-user validation accuracy, run the test once."""
import glob, json, os, subprocess, sys
RES = os.path.join(os.path.dirname(__file__), "results", "e60")
best = None
for p in glob.glob(os.path.join(RES, "*_val.json")):
    r = json.load(open(p)); acc = max(r["val_bag_acc"], r["val_seq_acc"])
    if best is None or acc > best[0]:
        best = (acc, r["args"])
acc, a = best
print(json.dumps({"selected": a, "val_acc": acc}), flush=True)
sys.exit(subprocess.call([sys.executable, os.path.join(os.path.dirname(__file__), "e60_dvs.py"), "--G", str(a["G"]), "--R", str(a["R"]),
                          "--S", str(a["S"]), "--lo", str(a["lo"]), "--hi", str(a["hi"]), "--rel", str(a["rel"]), "--O", str(a["O"]),
                          "--eval", "test"]))
