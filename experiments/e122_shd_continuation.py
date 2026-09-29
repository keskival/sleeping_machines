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
    p.add_argument("--checkpoint", default="experiments/results/e119/race_d8_linear_n1024_e8_s6.pt")
    a = p.parse_args()
    if Path(a.tag).name != a.tag or min(a.limit,a.epochs,a.bs)<1:
        raise ValueError("Invalid run settings")
    out = Path("experiments/results/e122")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    saved = torch.load(a.checkpoint,weights_only=False,map_location="cpu")
    net = RaceNet(depth=8,memory_backend="linear")
    net.load_state_dict(saved["state_dict"],strict=True)
    opt = torch.optim.Adam(net.parameters(),lr=a.lr)
    opt.load_state_dict(saved["optimizer"])
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
    result = {"status":"running","args":vars(a),"curve":[],"initial":eval_splits(),
              "checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "fit_ids":[r[4] for r in fit],"dev_original_ids":[r[4] for r in held[:256]],
              "dev_additional_ids":[r[4] for r in held[256:]],
              "protocol":"SHD train only, speakers 3/6 held out; original 256 and disjoint additional 256; clean evaluation; no test access",
              "augmentation":"fit only: integer band shift Uniform{-2,...,2}, crop outside 0..39; time scale exp(Uniform[-.15,.15]); count retained on surviving packets" if a.augment else "none",
              "normalization":"frozen original fit-only readout calibration, unchanged in both arms",
              "hardware":{"platform":platform.platform(),"torch":torch.__version__,"threads":1},
              "energy_joules":None}
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result,indent=2)+"\n"); tmp.replace(out)
    persist()
    print(json.dumps({"initial":{k:v["accuracy"] for k,v in result["initial"].items()}}),flush=True)
    for epoch in range(1,a.epochs+1):
        lr=a.lr*(.1+.9*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1,1)))/2)
        for group in opt.param_groups: group["lr"]=lr
        order=order_rng.permutation(len(fit))
        net.train()
        nll,steps,packets,values=0.,0,0,0
        grads=np.zeros(8)
        for start in range(0,len(fit),a.bs):
            rows=[fit[j] for j in order[start:start+a.bs]]
            if a.augment: rows=[augment(r,augmentation_rng) for r in rows]
            x=batch(rows)
            opt.zero_grad(set_to_none=True)
            scores,_,stats,_=net(*x[:4],len(rows))
            loss=F.cross_entropy(scores,x[-1])
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite SHD loss")
            loss.backward()
            grads += [float(layer.route.grad.norm()) for layer in net.layers]
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            nll+=float(loss.detach())*len(rows); steps+=1; packets+=stats["packets"]
            values+=sum(s["value_evaluations"] for s in stats["layers"])
        row={"epoch":epoch,"lr":lr,"online_nll":nll/len(fit),"route_grad_norm":(grads/steps).tolist(),
             "training_input_packets":packets,"training_value_evaluations":values,**eval_splits()}
        result["curve"].append(row)
        checkpoint={"args":vars(a),"state_dict":net.state_dict(),"optimizer":opt.state_dict(),
                    "epoch":epoch,"order_rng":order_rng.bit_generator.state,
                    "augmentation_rng":augmentation_rng.bit_generator.state,"curve":result["curve"],
                    "source_sha256":result["source_sha256"]}
        tmp=out.with_suffix(".pt.tmp"); torch.save(checkpoint,tmp); tmp.replace(out.with_suffix(".pt"))
        persist()
        print(json.dumps({"epoch":epoch,"online_nll":row["online_nll"],"fit":row["fit"]["accuracy"],
              "original":row["dev_original"]["accuracy"],"additional":row["dev_additional"]["accuracy"],
              "wall_s":result["wall_s"]}),flush=True)
    result.update(status="completed",final=result["curve"][-1]); persist()


if __name__=="__main__": main()
