"""Logical work ledger for completed common-model and small dense controls.

This is an analytic operation estimate, not a hardware energy meter. A MAC
costs two arithmetic units. Other arithmetic, nonlinear functions and estimated
sort comparisons cost one unit each; memory accesses are reported separately.
Evaluate each prefix separately so batch padding cannot hide per-query work.
"""
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
from e120_shared_tasks import modular, recall
from e120_shared_bench import inputs


def common_work(config, events, stats, pointer=False):
    d,C,L=config.get("dim",32),config["classes"],config["depth"]
    if config.get("phase_only"):
        # Grouped affine reduction, including symbol signs/indices and payload.
        scalar=10*events+3+2*C+4
        sorting=C*math.ceil(math.log2(C))+2*(C-1)
        return {"dense_macs":0,"memory_scan_macs":0,"scalar_ops":scalar,
                "sort_and_race_comparisons":sorting,"logical_state_reads":events+C,
                "pointer_candidates":0,"estimated_operations":scalar+sorting}
    K=3;F=2*d+1;J=config.get("evidence_count",0)
    # Value maps use only the selected payload; routers evaluate all 3 choices.
    dense=L*events*(K*F+d*F)+(d+1)**2+(d+1)*C+J*(d+1+C)
    scans=sum(r["scan_compositions"] for r in stats["layers"])
    scan_macs=K*(d+1)*scans
    scalar=events*(d+7)+events*(2*d+1)+d+1
    scalar+=L*(3*K*d*F+K*d)  # abs, row sums, division and clamp of value weights
    scalar+=L*events*(K*(d+10)+4*d+4)+K*scans
    sorting=L*(4*events*math.ceil(math.log2(max(events,2)))+events*(K*2+K-1))
    reads=events*(d//4)+L*events*(K*F+d*F)+2*scan_macs+(d+1)**2+(d+1)*C
    candidates=0
    if config.get("phase_period") is not None:
        scalar+=10*events+3+2*C+3*C
        sorting+=C*math.ceil(math.log2(C))+2*(C-1)
        reads+=events+C
    if pointer:
        n=events-1;candidates=4*n-6
        scalar+=4*n*3+candidates+4*C
        sorting+=max(candidates-1,0)
        reads+=candidates+1+C
    return {"dense_macs":dense,"memory_scan_macs":scan_macs,"scalar_ops":scalar,
            "sort_and_race_comparisons":sorting,"logical_state_reads":reads,
            "pointer_candidates":candidates,
            "estimated_operations":2*(dense+scan_macs)+scalar+sorting}


def dense_work(model, dim, depth, events, classes):
    if model=="lstm":
        macs=events*(4*dim+depth*4*dim*(dim+dim))+dim*classes
        scalar=events*(7+dim+depth*22*dim)+classes
    else:
        macs=depth*(12*events*dim*dim+2*events*events*dim)+dim*classes
        scalar=events*(2*dim+1)+depth*(22*events*dim+10*events*events)+events*dim+dim+classes
    return {"dense_macs":macs,"memory_scan_macs":0,"scalar_ops":scalar,
            "sort_and_race_comparisons":0,"logical_state_reads":macs,
            "pointer_candidates":0,"estimated_operations":2*macs+scalar}


def read(path):
    value=json.loads(path.read_text())
    if value.get("status")!="completed":raise ValueError(path)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=Path("experiments/results/e124")/(a.tag+".json")
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    tasks={"modular":modular(1473,3440,6),"recall":recall(512,256,6)}
    models=[("shared_phase_only",Path("experiments/results/e124/modular_phase_only_s6_200.json"),"modular"),
            ("shared_d2_phase",Path("experiments/results/e121/shared_d2_guard_s6_200.json"),"modular"),
            ("shared_d2_recall",Path("experiments/results/e120/recall_d2_20260929.json"),"recall")]
    rows=[]
    for name,path,task_name in models:
        result=read(path);task=tasks[task_name]
        saved=torch.load(path.with_suffix(".pt"),weights_only=False,map_location="cpu")
        model=SharedEventModel(**saved["config"]);model.load_state_dict(saved["state_dict"],strict=True);model.eval()
        splits={"unseen_all":task.dev} if task_name=="modular" else {"dev":task.dev,"context4x":task.extra["context4x"]}
        for split,examples in splits.items():
            ledgers=[];pred=[]
            with torch.no_grad():
                for sample in examples:
                    logits,_,stats,_=model(**inputs([sample]))
                    pred.append(int(logits.argmax(-1)[0]))
                    ledger=common_work(saved["config"],len(sample.prefix.channels),stats,pointer=task_name=="recall")
                    ledgers.append(ledger)
            expected=(result["extra"][split] if split=="context4x" else result["final"]["dev"] if
                      name=="shared_phase_only" or task_name=="recall" else result["final"]["unseen_all"])
            assert pred==expected["predictions"]
            rows.append({"model":name,"task":task_name,"split":split,"n":len(examples),
                "accuracy":sum(p==r.label for p,r in zip(pred,examples))/len(examples),
                "work":{k:float(np.mean([r[k] for r in ledgers])) for k in ledgers[0]},
                "result_path":str(path),"fitting_presentations":result.get("fitting_presentations"),
                "count_scope":"Full configured common forward; pointer lookup charged including precomputed evidence"})
    for task_name in tasks:
        for model_name in ("lstm","transformer"):
            tag="tf" if model_name=="transformer" else model_name
            path=Path("experiments/results/e123")/f"{task_name}_{tag}_d32_l2_s6.json"
            result=read(path);task=tasks[task_name]
            assert result["dev_ids"]==[r.identity for r in task.dev]
            for split,examples in ({"unseen_all":task.dev} if task_name=="modular" else
                                  {"dev":task.dev,"context4x":task.extra["context4x"]}).items():
                metric=result["extra"][split] if split=="context4x" else result["final"]["dev"]
                ledgers=[dense_work(model_name,32,2,len(r.prefix.channels),task.config["classes"]) for r in examples]
                rows.append({"model":model_name,"task":task_name,"split":split,"n":len(examples),
                    "accuracy":metric["accuracy"],"work":{k:float(np.mean([r[k] for r in ledgers])) for k in ledgers[0]},
                    "result_path":str(path),"fitting_presentations":result["fitting_presentations"],
                    "count_scope":"No padding, per-query recurrent/attention maps and scalar-operation estimate"})
    result={"status":"completed","rows":rows,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "units":"Estimated logical operations: 2 per MAC; 1 per other scalar operation, nonlinear function, or estimated sort comparison",
            "limitations":"Not measured joules, executed CPU instructions or a latency prediction. Memory reads reported separately; transfers, allocations, kernel launch and instrumentation not charged. Transcendental/division costs are unit-weighted. Shared weight normalization included. Backward/optimizer work is not inferred from forward counts.",
            "protocol":"Exact common-model development examples; modular 1473 fit ×200; recall pointer4000 + neural512×8 versus dense4512×8; one seed, width32/depth2; no dense hyperparameter search"}
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
