"""Compare the completed matched SHD continuations on paired utterances."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import h5py
import numpy as np
import e51_shd_world as S


def paired(a, b, y):
    ac, bc = np.array(a) == y, np.array(b) == y
    gained, lost = int((bc & ~ac).sum()), int((ac & ~bc).sum())
    n = gained+lost
    exact = min(1., 2*sum(math.comb(n,j) for j in range(min(gained,lost)+1))/2**n) if n else 1.
    return {"a_correct":int(ac.sum()), "b_correct":int(bc.sum()), "n":len(y),
            "gain_pp":100*float((bc.astype(float)-ac).mean()),
            "gained":gained,"lost":lost,"exact_discordance_p":exact,
            "note":"Descriptive paired test; utterances share speakers; no correction for earlier model exploration"}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--tag", required=True); a=p.parse_args()
    out=Path("experiments/results/e122")/(a.tag+".json")
    if out.exists(): raise FileExistsError(out)
    paths=[Path("experiments/results/e122")/f"d8_n2048_{arm}_s6.json" for arm in ("control","invariance")]
    control, aug=[json.loads(p.read_text()) for p in paths]
    assert control["status"] == aug["status"] == "completed"
    for key in ("checkpoint_sha256","fit_ids","dev_original_ids","dev_additional_ids","initial"):
        assert control[key] == aug[key], key
    for key in ("limit","epochs","bs","lr","seed"):
        assert control["args"][key] == aug["args"][key]
    with h5py.File(Path(S.ROOT)/"shd_train.h5", "r") as f:
        selected=np.isin(np.array(f["extra"]["speaker"]), S.VAL_SPEAKERS)
        labels=np.array(f["labels"])[selected]
        speakers=np.array(f["extra"]["speaker"])[selected]
    comparisons={}
    for split in ("dev_original","dev_additional","pooled"):
        parts=[split] if split != "pooled" else ["dev_original","dev_additional"]
        ids=np.concatenate([np.array(control[s+"_ids"]) for s in parts])
        y=labels[ids]
        pred=lambda r,endpoint: sum((r[endpoint][s]["predictions"] for s in parts), [])
        old,plain,new=pred(control,"initial"),pred(control,"final"),pred(aug,"final")
        comparisons[split]={"starting_vs_control":paired(old,plain,y),
                            "control_vs_invariance":paired(plain,new,y),
                            "starting_vs_invariance":paired(old,new,y)}
        if split == "pooled":
            comparisons[split]["by_speaker"]={str(spk):paired(np.array(plain)[speakers[ids]==spk],
                np.array(new)[speakers[ids]==spk], y[speakers[ids]==spk]) for spk in S.VAL_SPEAKERS}
    result={"status":"completed","matched_start_order_budget":True,"comparisons":comparisons,
            "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),*paths]},
            "scope":"One seed; clean train-file utterances from held-out speakers 3/6; official test untouched"}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__": main()
