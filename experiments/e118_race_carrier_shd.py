"""E118: winner-only delayed residual carriers with local loser credit.

All alternatives are scored, but only the earliest emits. During training a
zero-forward correction credits race scores from detached losing payloads
and delays. It is a local first-order surrogate, not exact discrete credit.
Fixed fit-only readout whitening isolates the conditioning failure diagnosed
by E117's layer probes. Run only with the guarded queue runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from e117_serial_event_shd import batch, load_items, segmented_memory

torch.set_num_threads(1)
OUT = Path(__file__).parent / "results" / "e118"


# Keep historical experiment entry points/checkpoint schemas compatible.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import RaceLayer, SharedEventModel


class RaceNet(SharedEventModel):
    def __init__(self, bands=40, dim=32, depth=8, groups=5, beta=1., cf_credit=True,
                 memory_backend="doubling"):
        super().__init__(bands, dim, depth, groups, beta, cf_credit, memory_backend)


@torch.no_grad()
def calibrate(net, items, bs):
    net.eval()
    rows = []
    for start in range(0, len(items), bs):
        inputs = batch(items[start:start+bs])
        rows.append(net(*inputs[:4], len(inputs[-1]))[1])
    x = torch.cat(rows).double()
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-4)
    z = (x-center)/scale
    cov = z.T@z/len(z)
    eigenvalues, vectors = torch.linalg.eigh(cov)
    whitener = (vectors * (eigenvalues.clamp_min(0)+.1).rsqrt()[None, :]) @ vectors.T
    net.center.copy_(center)
    net.scale.copy_(scale)
    net.whitener.copy_(whitener)
    return {"fit_only": True, "shrinkage": .1, "median_std": float(scale.median()),
            "covariance_eigenvalues": eigenvalues.tolist(),
            "effective_rank": float(eigenvalues.sum().square()/eigenvalues.square().sum())}


@torch.no_grad()
def evaluate(net, items, bs):
    net.eval()
    correct, nll, packets = 0, 0., 0
    max_payload, delay_sum = 0., 0.
    winners = np.zeros((len(net.layers), 3), dtype=np.int64)
    margins = np.zeros(len(net.layers))
    predictions = []
    scan_work = 0
    for start in range(0, len(items), bs):
        inputs = batch(items[start:start+bs])
        logits, _, stats, _ = net(*inputs[:4], len(inputs[-1]))
        pred, y = logits.argmax(-1), inputs[-1]
        correct += int((pred==y).sum())
        nll += float(F.cross_entropy(logits, y, reduction="sum"))
        predictions.extend(pred.tolist())
        packets += stats["packets"]
        delay_sum += stats["mean_added_delay_ms"]*stats["packets"]
        max_payload = max(max_payload, stats["max_payload"])
        for j, st in enumerate(stats["layers"]):
            winners[j] += st["winner_counts"]
            margins[j] += st["mean_margin_ms"]*stats["packets"]
            scan_work += st["scan_compositions"]
    return {"correct": correct, "n": len(items), "accuracy": correct/len(items),
            "nll": nll/len(items), "predictions": predictions, "packets": packets,
            "carrier_emissions": packets*len(net.layers), "race_candidates": packets*len(net.layers)*3,
            "winner_counts": winners.tolist(), "mean_margin_ms": (margins/packets).tolist(),
            "mean_added_delay_ms": delay_sum/packets, "max_payload": max_payload,
            "scan_compositions": scan_work}


def contracts(net):
    b = torch.tensor([0, 1, 5, 7, 2, 10, 1, 4, 0])
    t = torch.tensor([.01, .01, .02, .04, .06, .08, .02, .04, .06])
    c = torch.tensor([1., 3., 2., 1., 4., 1., 2., 1., 3.])
    ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    net.eval()
    inference = net(b,t,c,ids,2)[0]
    serial = net(b,t,c,ids,2,sequential=True)[0]
    net.train()
    training, _, _, trace = net(b,t,c,ids,2,trace=True)
    difference = float((inference-training).detach().abs().max())
    serial_difference = float((inference-serial).detach().abs().max())
    assert difference < 1e-7 and serial_difference < 1e-7
    # A losing payload is absent from forward output and ordinary value
    # credit; its detached contrast can nevertheless alter router credit.
    torch.manual_seed(118)
    layer = RaceLayer(4, 8, 1., True)
    layer.train()
    x = torch.tensor([[.2,.1,-.1,.3]], requires_grad=True)
    out, times, _, tr = layer(x, torch.tensor([.01]), torch.ones(1), torch.zeros(1,dtype=torch.long), trace=True)
    winner = int(tr["winner"][0])
    out.sum().backward()
    loser_value_norm = sum(float(layer.value.grad[k].abs().sum()) for k in range(3) if k != winner)
    route_grad_norm = float(layer.route.grad.norm())
    assert loser_value_norm == 0 and route_grad_norm > 0
    forward = out.detach().clone()
    with torch.no_grad():
        for k in range(3):
            if k != winner:
                layer.value[k].add_(10.)
    out2 = layer(x, torch.tensor([.01]), torch.ones(1), torch.zeros(1,dtype=torch.long))[0]
    assert torch.equal(forward, out2.detach())
    return {"winner_only_training_inference_error": difference,
            "scan_serial_error": serial_difference,
            "loser_value_gradient_l1": loser_value_norm,
            "counterfactual_router_gradient_l2": route_grad_norm,
            "changing_loser_values_changes_forward": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--credit", type=int, choices=(0,1), default=1)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--eval-limit", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=.003)
    ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args()
    if Path(a.tag).name != a.tag:
        raise ValueError("tag must be a filename component")
    OUT.mkdir(exist_ok=True)
    output = OUT/(a.tag+'.json')
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    torch.manual_seed(a.seed)
    net = RaceNet(depth=a.depth, cf_credit=bool(a.credit))
    checks = contracts(net)
    fit = load_items(40,.01,a.limit,"fit_spk",a.seed)
    dev = load_items(40,.01,a.eval_limit,"val_spk",a.seed+1)
    normalization = calibrate(net,fit,a.bs)
    optimizer = torch.optim.Adam(net.parameters(),lr=a.lr)
    rng = np.random.default_rng(a.seed+2)
    result = {"args": vars(a), "status": "running", "contracts": checks,
              "readout_calibration": normalization, "parameters": sum(p.numel() for p in net.parameters()),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "shared_source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] /
                  "sleeping_machines" / name).read_bytes()).hexdigest()
                  for name in ("shared_event.py", "event_memory.py")},
              "data_protocol": "SHD train only; speakers 3/6 held out; no augmentation; no test access",
              "fit_ids": [x[4] for x in fit], "dev_ids": [x[4] for x in dev],
              "fit_labels": [x[3] for x in fit], "dev_labels": [x[3] for x in dev], "curve": []}
    result["initial"] = {"fit": evaluate(net,fit,a.bs), "dev": evaluate(net,dev,a.bs)}
    print(json.dumps({"initial_fit": result["initial"]["fit"]["nll"],
                      "contracts": checks,"calibration_rank": normalization["effective_rank"]}),flush=True)
    for epoch in range(1,a.epochs+1):
        net.train()
        order = rng.permutation(len(fit))
        grads, route_grads = np.zeros(a.depth), np.zeros(a.depth)
        train_loss, n_batches, value_evaluations = 0., 0, 0
        for start in range(0,len(fit),a.bs):
            items = [fit[j] for j in order[start:start+a.bs]]
            inputs = batch(items)
            optimizer.zero_grad(set_to_none=True)
            logits,_,stats,_ = net(*inputs[:4],len(items))
            loss = F.cross_entropy(logits,inputs[-1])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite race loss")
            loss.backward()
            for j,layer in enumerate(net.layers):
                grads[j] += math.sqrt(sum(float(p.grad.square().sum()) for p in layer.parameters()
                                          if p.grad is not None))
                route_grads[j] += float(layer.route.grad.norm()) if layer.route.grad is not None else 0.
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
            train_loss += float(loss.detach())*len(items)
            value_evaluations += sum(s["value_evaluations"] for s in stats["layers"])
            n_batches += 1
        row = {"epoch": epoch,"training_online_nll": train_loss/len(fit),
               "fit": evaluate(net,fit,a.bs),"dev": evaluate(net,dev,a.bs),
               "layer_grad_norm": (grads/n_batches).tolist(),
               "route_grad_norm": (route_grads/n_batches).tolist(),
               "training_value_evaluations": value_evaluations,"wall_s": time.time()-started}
        result["curve"].append(row)
        result["wall_s"] = time.time()-started
        result["max_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({"epoch":epoch,"fit_acc":row["fit"]["accuracy"],"dev_acc":row["dev"]["accuracy"],
                          "fit_nll":row["fit"]["nll"],"dev_nll":row["dev"]["nll"],
                          "route_grad":row["route_grad_norm"],"wall_s":row["wall_s"]}),flush=True)
    result["status"] = "completed"
    result["wall_s"] = time.time()-started
    output.write_text(json.dumps(result,indent=2)+'\n')
    torch.save({"args":vars(a),"state_dict":net.state_dict()},OUT/(a.tag+'.pt'))
    print("completed",output,flush=True)


if __name__ == "__main__":
    main()
