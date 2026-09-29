"""Matched shallow shared-core arithmetic runs, with/without periodic state.

Same split, optimizer, example order and 200-epoch schedule. The phase arm adds
local phase teaching plus a bounded neural correction to its class-clock scores.
All shared hidden layers execute; only winning hidden values are emitted.
"""
import argparse
import hashlib
import json
import math
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
from e120_shared_bench import inputs,loss_for,calibrate,evaluate


@torch.no_grad()
def phase_metrics(model,rows,bs):
    if model.phase_memory is None: return {}
    correct,certified,changed=0,0,0
    for i in range(0,len(rows),bs):
        part=rows[i:i+bs]
        logits,_,st,_=model(**inputs(part))
        winner=np.array(st["phase_winners"])
        correct+=int((winner==np.array([r.label for r in part])).sum())
        certified+=st["phase_certified"]
        change=logits.argmax(-1).numpy()!=winner
        changed+=int(change.sum())
        margin=np.array(st["phase_margin"])
        bounds=np.array(st["phase_effective_bound"])
        assert not np.any(change & (margin>2*bounds))
        if model.phase_margin_guard:
            assert not np.any(change)
    return {"phase_correct":correct,"n":len(rows),"phase_accuracy":correct/len(rows),
            "certified_queries":certified,"neural_winner_changes":changed}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tag",required=True)
    p.add_argument("--phase",type=int,choices=(0,1),required=True)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--margin-guard",type=int,choices=(0,1),default=0)
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--bs",type=int,default=64)
    p.add_argument("--seed",type=int,default=6)
    a=p.parse_args()
    out=Path("experiments/results/e121")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    # All 3,440 unseen tuples are reported; the first 256 are the old E120 dev.
    task=modular(1473,3440,a.seed)
    task.protocol["note"] = ("Position-tagged operand symbols; all 3440 unseen tuples evaluated; "
        "200-epoch schedule; learned periodic state with supplied period 17" if a.phase else
        "Position-tagged operand symbols; all 3440 unseen tuples evaluated; no periodic state")
    config={**task.config,"depth":a.depth,"phase_period":17 if a.phase else None,"phase_seed":a.seed,
            "phase_margin_guard":bool(a.margin_guard)}
    model=SharedEventModel(**config)
    calibration=calibrate(model,task.fit,a.bs)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    order_rng=np.random.default_rng(a.seed+10)
    phase_rng=np.random.default_rng(a.seed+121)
    started=time.perf_counter()
    def evaluate_all():
        return {"fit":evaluate(model,task.fit,a.bs),"dev":evaluate(model,task.dev[:256],a.bs),
                "unseen_all":evaluate(model,task.dev,a.bs),
                "phase":phase_metrics(model,task.dev,a.bs)}
    source=[Path(__file__),Path("sleeping_machines/shared_event.py"),Path("sleeping_machines/phase_memory.py"),
            Path("experiments/e120_shared_tasks.py"),Path("sleeping_machines/readout_calibration.py")]
    result={"status":"running","args":vars(a),"config":config,"calibration":calibration,
            "curve":[],"initial":evaluate_all(),"fit_ids":[r.identity for r in task.fit],
            "unseen_ids":[r.identity for r in task.dev],"protocol":task.protocol,
            "parameters":sum(p.numel() for p in model.parameters()),
            "phase_parameters":69 if a.phase else 0,
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
            "hardware":{"platform":platform.platform(),"threads":1,"torch":torch.__version__},
            "energy_joules":None}
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix(".json.tmp");tmp.write_text(json.dumps(result,indent=2)+"\n");tmp.replace(out)
    persist()
    for epoch in range(1,a.epochs+1):
        lr=.003*(.1+.9*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1,1)))/2)
        for group in optimizer.param_groups:group["lr"]=lr
        order=order_rng.permutation(len(task.fit));total=0.;phase_mistakes=0
        model.train()
        for start in range(0,len(order),a.bs):
            rows=[task.fit[j] for j in order[start:start+a.bs]]
            scores=model(**inputs(rows))[0]
            loss=loss_for(scores,rows)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite arithmetic loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(rows)
            if model.phase_memory is not None:
                for row in rows:
                    phase_mistakes+=model.phase_memory.teach(row.prefix.channels,row.label,phase_rng)
        if epoch in (1,8) or epoch%10==0 or epoch==a.epochs:
            row={"epoch":epoch,"online_nll":total/len(task.fit),"phase_mistakes":phase_mistakes,**evaluate_all()}
            result["curve"].append(row);persist()
            print(json.dumps({"epoch":epoch,"fit":row["fit"]["accuracy"],"dev":row["dev"]["accuracy"],
                  "all_unseen":row["unseen_all"]["accuracy"],"wall_s":result["wall_s"]}),flush=True)
    result.update(status="completed",final=result["curve"][-1])
    if model.phase_memory is not None:result["phase_local_updates"]=int(model.phase_memory.updates)
    torch.save({"state_dict":model.state_dict(),"config":config,"args":vars(a),
                "optimizer":optimizer.state_dict(),"order_rng":order_rng.bit_generator.state,
                "phase_rng":phase_rng.bit_generator.state,"source_sha256":result["source_sha256"]},out.with_suffix(".pt"))
    persist()


if __name__=="__main__":main()
