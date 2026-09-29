"""Same-query breadth references with explicitly scoped arithmetic ledgers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e120_shared_bench import BUILDERS, inputs
from e124_work_audit import common_work, dense_work, read


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True)
    args=parser.parse_args()
    out=Path("experiments/results/e124")/(args.tag+".json")
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);rows=[]
    for name in ("language","market","temporal","mnist","dvs"):
        common_path=Path("experiments/results/e120")/f"{name}_d8_20260929.json"
        reference_path=Path("experiments/results/e123")/f"{name}_tf_d32_l2_s6.json"
        common,reference=read(common_path),read(reference_path)
        task=BUILDERS[name](common["args"]["fit"],common["args"]["dev"],6)
        assert common["fit_ids"]==reference["fit_ids"]==[r.identity for r in task.fit]
        assert common["dev_ids"]==reference["dev_ids"]==[r.identity for r in task.dev]
        saved=torch.load(common_path.with_suffix(".pt"),weights_only=False,map_location="cpu")
        model=SharedEventModel(**saved["config"]);model.load_state_dict(saved["state_dict"]);model.eval()
        ledgers=[]
        with torch.no_grad():
            for sample in task.dev:
                _,_,stats,_=model(**inputs([sample]))
                ledgers.append(common_work(saved["config"],len(sample.prefix.channels),stats))
        C,d,L=saved["config"]["classes"],32,8
        F,K,J=2*d+1,3,saved["config"].get("evidence_count",0)
        # Executed training-forward contractions, including all losing values.
        # Backward, optimizer and memory fitting are deliberately separate.
        training_forward_macs=0
        for epoch in common["curve"]:
            E=epoch["training_prefix_packets"];V=epoch["training_value_evaluations"]
            S=epoch["training_scan_compositions"]
            training_forward_macs+=L*E*K*F+V*d*F+K*(d+1)*S
            training_forward_macs+=len(task.fit)*((d+1)**2+(d+1)*C+J*(d+1+C))
        dense=[dense_work("transformer",32,2,len(r.prefix.channels),C) for r in task.dev]
        dense_training_macs=sum(dense_work("transformer",32,2,len(r.prefix.channels),C)["dense_macs"]
                                for r in task.fit)*reference["args"]["epochs"]
        rows.append({"task":name,"common_result":str(common_path),"reference_result":str(reference_path),
            "common_metric":{k:v for k,v in common["final"]["dev"].items() if k in ("accuracy","nll","correct","n")},
            "reference_metric":{k:v for k,v in reference["final"]["dev"].items() if k in ("accuracy","nll","correct","n")},
            "common_forward_map_scan_flops":float(np.mean([2*(r["dense_macs"]+r["memory_scan_macs"]) for r in ledgers])),
            "reference_forward_map_attention_flops":float(np.mean([2*r["dense_macs"] for r in dense])),
            "common_training_forward_map_scan_flops":int(2*training_forward_macs),
            "reference_training_forward_map_attention_flops":int(2*dense_training_macs),
            "common_neural_presentations":len(task.fit)*8,"reference_presentations":len(task.fit)*8,
            "epochs":8,"common_depth":8,"reference_depth":2,"width":32,
            "memory_fitting":common["protocol"],"same_examples":True})
    phase=read(Path("experiments/results/e124/modular_phase_only_s6_200.json"))
    teachers={"phase":{"fitting_presentations":phase["fitting_presentations"],
                        "mistaken_updates":phase["local_updates"],
                        "learned_scalar_update_visits":5*phase["local_updates"],
                        "note":"Three position-tagged symbol offsets, one global offset and one class clock per mistaken update; integer counter excluded"}}
    for name in ("lstm","tf"):
        ref=read(Path("experiments/results/e123")/f"modular_{name}_d32_l2_s6.json")
        steps=ref["args"]["epochs"]*math.ceil(len(ref["fit_ids"])/ref["args"]["bs"])
        teachers[name]={"fitting_presentations":ref["fitting_presentations"],"optimizer_steps":steps,
                        "parameters":ref["parameters"],"learned_scalar_update_visits":steps*ref["parameters"],
                        "note":"Dense Adam parameter visits; excludes momentum/variance state updates and backward arithmetic"}
    result={"status":"completed","rows":rows,"arithmetic_teacher_work":teachers,
        "scope":"FLOPs count 2 per dense contraction/scan MAC; per-prefix inference without padding. Training-forward uses recorded carrier packets, all winning/losing values and scan combines. These are estimates for these contractions, not total runtime FLOPs or joules. Nonlinearities, sorting, normalization arithmetic, evidence fitting/lookup, backward and optimizer excluded. Reference uses same neural examples/schedule but no common model's separately fitted evidence bank.",
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__),Path("experiments/e124_work_audit.py")]}}
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
