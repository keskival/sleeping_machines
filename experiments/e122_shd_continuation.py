"""Matched SHD continuations: more fitting data, with/without input invariance.

Same frozen starting checkpoint, optimizer state, example order and schedule.
Only the augmented arm changes fitting inputs. Evaluate clean utterances;
preserve the original 256 development examples and separately score 256 new
examples from the same held-out speakers. No official test access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from e117_serial_event_shd import batch, load_items
from e118_race_carrier_shd import RaceNet, evaluate
from sleeping_machines.shared_event import SharedEventModel


def augment(item, rng):
    b, t, c, y, identity = item
    shift = int(rng.integers(-2, 3))
    scale = math.exp(float(rng.uniform(-.15, .15)))
    shifted = b+shift
    keep = (shifted >= 0) & (shifted < 40)
    if not keep.any():
        shifted, keep = b, np.ones(len(b), dtype=bool)
    return (shifted[keep], (t[keep]*scale).astype(np.float32), c[keep], y, identity)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--augment", type=int, choices=(0,1), required=True)
    p.add_argument("--limit", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--readout", choices=("mean","weighted"), default="mean")
    p.add_argument("--value-backward", choices=("full", "winner"), default="full")
    p.add_argument("--separate-keys", action="store_true",
                   help="Compute immutable key routing from each query; values learn separately")
    p.add_argument("--train-value-only", action="store_true",
                   help="Teach only added context value maps, holding routes and old parameters fixed")
    p.add_argument("--global-context-layers", default="",
                   help="Comma-separated zero-initialized causal context layers, e.g. 3,7")
    p.add_argument("--train-new-only", action="store_true",
                   help="Hold checkpoint parameters fixed; teach only added context/key columns")
    p.add_argument("--checkpoint", default="experiments/results/e119/race_d8_linear_n1024_e8_s6.pt")
    p.add_argument("--initial-result", help="Reuse verified identical initial predictions from an interrupted run")
    p.add_argument("--checkpoint-every", type=int, default=128)
    p.add_argument("--resume-progress", action="store_true")
    a = p.parse_args()
    if Path(a.tag).name != a.tag or min(a.limit,a.epochs,a.bs,a.checkpoint_every)<1:
        raise ValueError("Invalid run settings")
    out = Path("experiments/results/e122")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    progress_path=out.with_suffix(".progress.pt")
    if out.exists() and not a.resume_progress:raise FileExistsError(out)
    if a.resume_progress and (not out.exists() or not progress_path.exists()):
        raise ValueError("Resume requires this tag's result and progress checkpoint")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    saved = torch.load(a.checkpoint,weights_only=False,map_location="cpu")
    context_layers = tuple(int(j) for j in a.global_context_layers.split(",") if j)
    net = SharedEventModel(depth=8,memory_backend="linear",readout=a.readout,
                           global_context_layers=context_layers, value_backward=a.value_backward)
    missing,unexpected = net.load_state_dict(saved["state_dict"],strict=False)
    named = dict(net.named_parameters())
    allowed_new = {name for name in named if name == "readout_gain.weight" or
                   ".bridge_value" in name or ".bridge_route" in name}
    if unexpected or not set(missing).issubset(allowed_new):
        raise ValueError(f"Checkpoint mismatch: {missing}, {unexpected}")
    if a.separate_keys:
        net.freeze_keys_from_values()
        named = dict(net.named_parameters())
    key_names={n for n in named if n.startswith("key_")}
    groups = saved.get("optimizer_parameter_groups")
    if groups is None:
        old_names = [n for n in named if n not in missing and n not in key_names]
        if "readout_gain.weight" in saved["state_dict"]:
            groups = [[n for n in old_names if n != "readout_gain.weight"], ["readout_gain.weight"]]
        else:
            groups = [old_names]
    assert set(sum(groups, [])) == set(named) - set(missing) - key_names
    opt = torch.optim.Adam([named[n] for n in groups[0]], lr=a.lr)
    for group in groups[1:]:
        opt.add_param_group({"params": [named[n] for n in group], "lr": a.lr})
    opt.load_state_dict(saved["optimizer"])
    if missing:
        groups = groups + [missing]
        opt.add_param_group({"params": [named[n] for n in missing], "lr": a.lr})
    if a.train_new_only or a.train_value_only:
        if not allowed_new:raise ValueError("There are no new components to train")
        train_names = {n for n in allowed_new if ".bridge_value" in n} if a.train_value_only else allowed_new
        if not train_names:raise ValueError("No added context values to train")
        for name,param in named.items():param.requires_grad_(name in train_names and name not in key_names)
    order_rng = np.random.default_rng(a.seed+2)
    order_rng.bit_generator.state = saved.get("numpy_rng", saved.get("order_rng"))
    augmentation_rng = np.random.default_rng(a.seed+122)
    if "augmentation_rng" in saved:
        augmentation_rng.bit_generator.state = saved["augmentation_rng"]
    fit = load_items(40,.01,a.limit,"fit_spk",a.seed)
    held = load_items(40,.01,512,"val_spk",a.seed+1)
    if len(held) != 512: raise ValueError("Missing held-out examples")
    started = time.perf_counter()
    def eval_splits():
        return {"fit":evaluate(net,fit,a.bs), "dev_original":evaluate(net,held[:256],a.bs),
                "dev_additional":evaluate(net,held[256:],a.bs)}
    sources = [Path(__file__),Path("experiments/e117_serial_event_shd.py"),
               Path("experiments/e118_race_carrier_shd.py"),
               Path("sleeping_machines/shared_event.py"),Path("sleeping_machines/event_memory.py")]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cached=None
    if a.initial_result and not a.resume_progress:
        cached=json.loads(Path(a.initial_result).read_text())
        for key in ("augment","limit","epochs","bs","lr","seed","readout","value_backward",
                    "separate_keys","train_value_only","global_context_layers","train_new_only","checkpoint"):
            if cached["args"].get(key)!=vars(a)[key]:raise ValueError(f"Initial-cache setting differs: {key}")
        if cached["checkpoint_sha256"]!=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest():
            raise ValueError("Initial-cache checkpoint differs")
        for path in sources[1:]:
            if cached["source_sha256"][str(path)]!=source_hashes[str(path)]:
                raise ValueError(f"Initial-cache computation source differs: {path}")
        for split,rows in (("fit",fit),("dev_original",held[:256]),("dev_additional",held[256:])):
            if cached[split+"_ids"]!=[r[4] for r in rows]:raise ValueError("Initial-cache IDs differ")
    progress=torch.load(progress_path,weights_only=False,map_location="cpu") if a.resume_progress else None
    if progress is not None:
        result=progress["result"]
        if result["status"]!="running" or result["source_sha256"]!=source_hashes:
            raise ValueError("Resume requires running status and identical computation sources")
        for key,value in vars(a).items():
            if key not in ("resume_progress","initial_result") and result["args"].get(key)!=value:
                raise ValueError(f"Resume setting differs: {key}")
        net.load_state_dict(progress["state_dict"]);opt.load_state_dict(progress["optimizer"])
        order_rng.bit_generator.state=progress["order_rng"]
        augmentation_rng.bit_generator.state=progress["augmentation_rng"]
        torch.set_rng_state(progress["torch_rng"])
    else:
        result = {"status":"running","args":vars(a),"curve":[],"initial":cached["initial"] if cached else eval_splits(),
              "checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              "source_sha256":source_hashes,
              "fit_ids":[r[4] for r in fit],"dev_original_ids":[r[4] for r in held[:256]],
              "dev_additional_ids":[r[4] for r in held[256:]],
              "protocol":"SHD train only, speakers 3/6 held out; original 256 and disjoint additional 256; clean evaluation; no test access",
              "augmentation":"fit only: integer band shift Uniform{-2,...,2}, crop outside 0..39; time scale exp(Uniform[-.15,.15]); count retained on surviving packets" if a.augment else "none",
              "normalization":"frozen original fit-only readout calibration, unchanged in both arms",
              "new_parameters": missing, "global_context_layers": context_layers,
              "hardware":{"platform":platform.platform(),"torch":torch.__version__,"threads":1},
              "energy_joules":None}
        if cached:result["initial_evaluation_reused_from"]={"path":a.initial_result,
                "sha256":hashlib.sha256(Path(a.initial_result).read_bytes()).hexdigest(),
                "scope":"Same starting state, computation sources and query IDs; initial evaluation runtime reused"}
    elapsed_before=progress["elapsed_s"] if progress is not None else 0.
    def persist():
        result.update(wall_s=elapsed_before+time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result,indent=2)+"\n"); tmp.replace(out)
    persist()
    print(json.dumps({"initial":{k:v["accuracy"] for k,v in result["initial"].items()}}),flush=True)
    for epoch in range(progress["epoch"] if progress is not None else 1,a.epochs+1):
        lr=a.lr*(.1+.9*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1,1)))/2)
        for group in opt.param_groups: group["lr"]=lr
        continuing=progress is not None and progress["epoch"]==epoch
        order=progress["order"] if continuing else order_rng.permutation(len(fit))
        net.train()
        nll,steps,packets,values=0.,0,0,0
        grads=np.zeros(8); global_scans=0; key_scans=0; key_values=0
        if continuing:
            nll,steps,packets,values,grads,global_scans,key_scans,key_values=progress["accumulators"]
        first_start=progress["next_start"] if continuing else 0
        for start in range(first_start,len(fit),a.bs):
            rows=[fit[j] for j in order[start:start+a.bs]]
            if a.augment: rows=[augment(r,augmentation_rng) for r in rows]
            x=batch(rows)
            opt.zero_grad(set_to_none=True)
            scores,_,stats,_=net(*x[:4],len(rows))
            loss=F.cross_entropy(scores,x[-1])
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite SHD loss")
            loss.backward()
            grads += [float(layer.route.grad.norm()) if layer.route.grad is not None else 0.
                      for layer in net.layers]
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            nll+=float(loss.detach())*len(rows); steps+=1; packets+=stats["packets"]
            values+=sum(s["value_evaluations"] for s in stats["layers"])
            global_scans+=sum(s.get("global_scan_compositions",0) for s in stats["layers"])
            key_scans+=sum(s.get("key_scan_compositions",0) for s in stats["layers"])
            key_values+=sum(s.get("key_value_evaluations",0) for s in stats["layers"])
            next_start=min(start+a.bs,len(fit))
            if steps%a.checkpoint_every==0 or next_start==len(fit):
                result["progress"]={"epoch":epoch,"completed_examples":next_start,
                    "optimizer_steps":steps,"online_nll_so_far":nll/next_start}
                persist()
                state={"state_dict":net.state_dict(),"optimizer":opt.state_dict(),
                    "result":result,"epoch":epoch,"next_start":next_start,"order":order,
                    "order_rng":order_rng.bit_generator.state,"augmentation_rng":augmentation_rng.bit_generator.state,
                    "torch_rng":torch.get_rng_state(),"elapsed_s":result["wall_s"],
                    "accumulators":(nll,steps,packets,values,grads,global_scans,key_scans,key_values)}
                temporary=progress_path.with_suffix(".pt.tmp");torch.save(state,temporary);temporary.replace(progress_path)
                print(json.dumps(result["progress"]),flush=True)
        row={"epoch":epoch,"lr":lr,"online_nll":nll/len(fit),"route_grad_norm":(grads/steps).tolist(),
             "training_input_packets":packets,"training_value_evaluations":values,**eval_splits()}
        row["training_global_scan_compositions"] = global_scans
        if a.separate_keys:
            row["training_key_scan_compositions"] = key_scans
            row["training_key_value_evaluations"] = key_values
        if context_layers:
            row["bridge_parameter_norm"] = {n:float(named[n].detach().norm()) for n in named
                                             if ".bridge_" in n}
        if net.readout_gain is not None:
            row["readout_gain_norm"] = float(net.readout_gain.weight.detach().norm())
        result["curve"].append(row)
        progress=None
        checkpoint={"args":vars(a),"state_dict":net.state_dict(),"optimizer":opt.state_dict(),
                    "config":{"depth":8,"memory_backend":"linear","readout":a.readout,
                              "global_context_layers":context_layers,"value_backward":a.value_backward,
                              "separate_keys":a.separate_keys},
                    "epoch":epoch,"order_rng":order_rng.bit_generator.state,
                    "augmentation_rng":augmentation_rng.bit_generator.state,"curve":result["curve"],
                    "optimizer_parameter_groups":groups,
                    "source_sha256":result["source_sha256"]}
        tmp=out.with_suffix(".pt.tmp"); torch.save(checkpoint,tmp); tmp.replace(out.with_suffix(".pt"))
        persist()
        print(json.dumps({"epoch":epoch,"online_nll":row["online_nll"],"fit":row["fit"]["accuracy"],
              "original":row["dev_original"]["accuracy"],"additional":row["dev_additional"]["accuracy"],
              "wall_s":result["wall_s"]}),flush=True)
    result.update(status="completed",final=result["curve"][-1]); persist()


if __name__=="__main__": main()
