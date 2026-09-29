"""Matched causal-context continuations, class errors and fixed-state check."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from e122_summary import paired
from e125_class_audit import class_scores
import e51_shd_world as S


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True)
    args=parser.parse_args()
    out=Path("experiments/results/e126")/(args.tag+".json")
    if out.exists():raise FileExistsError(out)
    root=Path("experiments/results/e122")
    paths={"mean":root/"d8_n4096_pool_control_s6.json",
           "full_context":root/"d8_n4096_bridge_s6.json",
           "new_columns":root/"d8_n4096_bridge_frozen_s6.json"}
    models={n:json.loads(p.read_text()) for n,p in paths.items()}
    baseline=models["mean"]
    for model in models.values():
        assert model["status"]=="completed"
        for key in ("checkpoint_sha256","fit_ids","dev_original_ids","dev_additional_ids","initial"):
            assert model[key]==baseline[key],key
        for key in ("limit","epochs","bs","lr","augment","seed"):
            assert model["args"][key]==baseline["args"][key]
    with h5py.File(Path(S.ROOT)/"shd_train.h5","r") as data:
        held=np.isin(np.array(data["extra"]["speaker"]),S.VAL_SPEAKERS)
        labels=np.array(data["labels"]);speakers=np.array(data["extra"]["speaker"])
    parts=("dev_original","dev_additional")
    ids=np.concatenate([np.array(baseline[s+"_ids"]) for s in parts])
    y=labels[held][ids];spk=speakers[held][ids]
    predictions=lambda model,endpoint:np.array(sum((model[endpoint][s]["predictions"] for s in parts),[]))
    initial=predictions(baseline,"initial");plain=predictions(baseline,"final")
    comparisons={};classes={}
    fit_ids=np.array(baseline["fit_ids"])
    for name,model in models.items():
        pred=predictions(model,"final")
        comparisons[name]={"starting_vs_final":paired(initial,pred,y),
                           "mean_vs_final":paired(plain,pred,y),
                           "by_speaker":{str(s):paired(plain[spk==s],pred[spk==s],y[spk==s])
                                         for s in S.VAL_SPEAKERS}}
        classes[name]={"fit":class_scores(model["final"]["fit"]["predictions"],labels[~held][fit_ids]),
                       "held":class_scores(pred,y)}
    parent_path=Path(baseline["args"]["checkpoint"])
    parent=torch.load(parent_path,weights_only=False,map_location="cpu")["state_dict"]
    frozen=torch.load(paths["new_columns"].with_suffix(".pt"),weights_only=False,map_location="cpu")["state_dict"]
    assert all(torch.equal(v,frozen[k]) for k,v in parent.items())
    result={"status":"completed","matched_initial_order_budget":True,"old_state_exactly_fixed":True,
            "comparisons":comparisons,"class_errors":classes,
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]},
            "scope":"One seed, paired held-out training-file speakers 3/6; official test untouched; descriptive comparisons"}
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
