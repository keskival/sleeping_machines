"""Read-only reset diagnostics of a completed E143 temporal residual learner.

Keep its trained correction head and frozen parent. Reset one hidden subsystem
to its original initialization; measure coordinated representation dependence.
This does not replace a matched frozen-feature training comparison and does not
prove that no other learner could obtain the same score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
import torch
from e143_event_state_shd import evaluate
from e139_fine_packet_model import load_marked
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_state import CoalescedEventStateEncoder


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True)
    p.add_argument("--run",default="experiments/results/e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json")
    a=p.parse_args();out=Path("experiments/results/e145")/(a.tag+".json");out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError("Unique output required")
    run=json.loads(Path(a.run).read_text())
    if run["status"]!="completed":raise ValueError("Completed training required")
    cfg=run["args"];epoch=max(run["curve"],key=lambda r:r["dev"]["correct"])["epoch"]
    base=Path(a.run).with_suffix(".pt")
    checkpoint=base.with_name(base.stem+f".epoch{epoch}.pt")
    if not checkpoint.exists():checkpoint=base
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["result"]["final"]["epoch"]!=epoch:
        raise ValueError("Best-development checkpoint not preserved")
    for name,digest in run["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError(f"Changed source: {name}")
    torch.set_num_threads(1);torch.manual_seed(cfg["seed"]);started=time.perf_counter()
    parent=torch.load(cfg["checkpoint"],map_location="cpu",weights_only=False)
    core=SharedEventModel(depth=8,memory_backend="linear")
    core.load_state_dict(parent["state_dict"]);core.requires_grad_(False);core.eval()
    encoder=CoalescedEventStateEncoder(sources=720,width=cfg["width"],modes=cfg["modes"],depth=cfg["depth"])
    initial={k:v.clone() for k,v in encoder.state_dict().items()}
    trained=saved["encoder_state_dict"]
    held=load_marked(512,"val_spk",cfg["seed"]+1)
    if [r[8] for r in held]!=run["dev_absolute_ids"]:raise ValueError("Changed held IDs")
    fit=load_marked(128,"fit_spk",cfg["seed"])
    resets={"trained":lambda name:False,
        "reset_state_stack":lambda name:name.startswith("layers."),
        "reset_source_embedding":lambda name:name.startswith("embedding."),
        "reset_clocks":lambda name:".clock." in name,
        "reset_modal_dynamics":lambda name:name.endswith(("raw_rate","frequency")),
        "reset_all_hidden":lambda name:not name.startswith("head.")}
    rows={}
    for mode,predicate in resets.items():
        state={k:initial[k] if predicate(k) else v for k,v in trained.items()}
        encoder.load_state_dict(state)
        row=dict(held=evaluate(core,encoder,held,cfg["bs"],cfg["window"]),
            fit128=evaluate(core,encoder,fit,cfg["bs"],cfg["window"]),
            reset_parameter_names=[k for k in trained if predicate(k)])
        rows[mode]=row
        print(json.dumps({"mode":mode,"held_correct":row["held"]["correct"],"held_nll":row["held"]["nll"],"fit128_correct":row["fit128"]["correct"]}),flush=True)
    expected=next(r for r in run["curve"] if r["epoch"]==epoch)["dev"]
    if rows["trained"]["held"]["predictions"]!=expected["predictions"]:
        raise ValueError("Restored checkpoint predictions differ")
    base_predictions=rows["trained"]["held"]["predictions"]
    labels=rows["trained"]["held"]["labels"]
    for mode,row in rows.items():
        pred=row["held"]["predictions"]
        row["paired_vs_trained"]=dict(
            trained_only_correct=sum(p==y and q!=y for p,q,y in zip(base_predictions,pred,labels)),
            reset_only_correct=sum(p!=y and q==y for p,q,y in zip(base_predictions,pred,labels)),
            changed_predictions=sum(p!=q for p,q in zip(base_predictions,pred)))
    result=dict(status="completed",run=a.run,selected_epoch=epoch,
        selection="Best of the three completed private-development epochs; exploratory selection, no official test",
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        rows=rows,dev_absolute_ids=run["dev_absolute_ids"],fit128_absolute_ids=[r[8] for r in fit],
        scope="Uncoordinated subsystem resets with trained head fixed. Tests dependence of this fitted solution, not a matched training intervention or architectural necessity. No optimizer updates; no official test",
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1),
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('experiments/e143_event_state_shd.py'),Path('sleeping_machines/event_state.py')]})
    out.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
