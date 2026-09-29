"""Train the common model's sparse periodic path without unused carrier layers."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e120_shared_tasks import modular
from e120_shared_bench import inputs


@torch.no_grad()
def evaluate(model,rows):
    scores,_,st,_=model(**inputs(rows))
    predictions=scores.argmax(-1).tolist()
    correct=sum(p==r.label for p,r in zip(predictions,rows))
    return {"correct":correct,"n":len(rows),"accuracy":correct/len(rows),
            "predictions":predictions,"phase_certified":st["phase_certified"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True)
    p.add_argument("--epochs",type=int,default=200);p.add_argument("--seed",type=int,default=6)
    a=p.parse_args()
    out=Path("experiments/results/e124")/(a.tag+".json");out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    task=modular(1473,3440,a.seed)
    config={**task.config,"depth":0,"phase_period":17,"phase_seed":a.seed,"phase_only":True}
    model=SharedEventModel(**config)
    order_rng=np.random.default_rng(a.seed+10);phase_rng=np.random.default_rng(a.seed+121)
    started=time.perf_counter()
    result={"status":"running","args":vars(a),"config":config,"curve":[],
            "fit_ids":[r.identity for r in task.fit],"dev_ids":[r.identity for r in task.dev],
            "protocol":{**task.protocol,"note":"Only the common class's phase primitive executes; supplied period; local timing teaching"},
            "parameters":69,"neural_parameters":0,"fitting_presentations":len(task.fit)*a.epochs,
            "hardware":{"platform":platform.platform(),"threads":1,"torch":torch.__version__},
            "energy_joules":None,"source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),Path("sleeping_machines/shared_event.py"),Path("sleeping_machines/phase_memory.py")]}}
    for epoch in range(1,a.epochs+1):
        mistakes=0
        for i in order_rng.permutation(len(task.fit)):
            row=task.fit[i];mistakes+=model.phase_memory.teach(row.prefix.channels,row.label,phase_rng)
        if epoch in (1,8) or epoch%10==0 or epoch==a.epochs:
            row={"epoch":epoch,"mistakes":mistakes,"fit":evaluate(model,task.fit),"dev":evaluate(model,task.dev)}
            result["curve"].append(row)
            print(json.dumps({"epoch":epoch,"dev":row["dev"]["accuracy"]}),flush=True)
    result.update(status="completed",final=result["curve"][-1],wall_s=time.perf_counter()-started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  local_updates=int(model.phase_memory.updates))
    # This extraction should preserve the independently taught phase state exactly.
    parent=Path("experiments/results/e121/shared_d2_guard_s6_200.pt")
    if parent.exists() and a.seed==6 and a.epochs==200:
        full=torch.load(parent,weights_only=False,map_location="cpu")["state_dict"]
        result["phase_state_matches_coupled"] = all(torch.equal(v,full[k]) for k,v in model.state_dict().items())
        assert result["phase_state_matches_coupled"]
    out.write_text(json.dumps(result,indent=2)+"\n")
    torch.save({"state_dict":model.state_dict(),"config":config,"args":vars(a)},out.with_suffix(".pt"))


if __name__=="__main__":main()
