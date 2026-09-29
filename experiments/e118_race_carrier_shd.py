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


class RaceLayer(nn.Module):
    def __init__(self, dim, depth, beta, cf_credit, options=3):
        super().__init__()
        self.alpha, self.cf_credit = beta/depth, cf_credit
        self.options, self.dim = options, dim
        self.value = nn.Parameter(torch.randn(options, dim, 2*dim+1)*.03)
        self.bias = nn.Parameter(torch.zeros(options, dim))
        self.route = nn.Parameter(torch.randn(options, 2*dim+1)*.1)
        self.route_bias = nn.Parameter(torch.zeros(options))
        self.log_tau = nn.Parameter(torch.logspace(math.log10(.02), math.log10(.8), options).log())

    def forward(self, x, t, count, keys, sequential=False, override=None, trace=False):
        # Learned delays can reorder carriers. Memory sees actual arrival
        # order within each receiver, with the original index breaking ties.
        time_order = torch.argsort(t, stable=True)
        inverse = torch.argsort(time_order)
        mem, mass, work = segmented_memory(
            x[time_order], t[time_order], count[time_order], keys[time_order],
            self.log_tau.clamp(math.log(.002), math.log(4.)).exp(), sequential)
        mem, mass = mem[inverse], mass[inverse]
        features = torch.cat((x[:, None, :].expand(-1, self.options, -1), mem,
                              (mass/(1+mass))[:, :, None]), -1)
        scores = (features*self.route[None, :, :]).sum(-1) + self.route_bias
        delay = .001 + .010*torch.sigmoid(-scores)
        winner = delay.argmin(-1)
        if override is not None:
            event, choice = override
            winner = winner.clone()
            winner[event] = choice
        w = self.value / self.value.abs().sum(-1, keepdim=True).clamp_min(1)
        rows = torch.arange(len(x))
        if self.training and self.cf_credit or trace:
            alternatives = torch.tanh(torch.einsum("ekf,kdf->ekd", features, w) + self.bias)
            correction = alternatives[rows, winner]
            value_evaluations = len(x)*self.options
        else:
            # Inference and pathwise control evaluate only selected values.
            correction = torch.zeros_like(x)
            for k in range(self.options):
                selected = torch.nonzero(winner == k, as_tuple=True)[0]
                correction = correction.index_copy(0, selected,
                    torch.tanh(F.linear(features[selected, k], w[k], self.bias[k])))
            alternatives = None
            value_evaluations = len(x)
        winning_delay = delay[rows, winner]
        out = x + self.alpha*correction
        tout = t + winning_delay
        if self.training and self.cf_credit:
            p = torch.softmax(-delay/.002, -1)
            zero_forward = p-p.detach()
            # Loser values have no ordinary value-path gradient. They supply
            # only a score direction comparing alternative and winner.
            out = out + self.alpha * (zero_forward[:, :, None] *
                (alternatives-correction[:, None, :]).detach()).sum(1)
            tout = tout + (zero_forward * (delay-winning_delay[:, None]).detach()).sum(1)
        sorted_delay = delay.detach().sort(-1).values
        stats = {"winner_counts": torch.bincount(winner, minlength=self.options).tolist(),
                 "mean_margin_ms": float((sorted_delay[:, 1]-sorted_delay[:, 0]).mean()*1000),
                 "mean_delay_ms": float(winning_delay.detach().mean()*1000),
                 "scan_compositions": work, "value_evaluations": value_evaluations}
        details = None
        if trace:
            details = {"winner": winner, "alternatives": alternatives,
                       "delays": delay, "out": out, "times": tout}
        return out, tout, stats, details


class RaceNet(nn.Module):
    def __init__(self, bands=40, dim=32, depth=8, groups=5, beta=1., cf_credit=True):
        super().__init__()
        if depth < 1 or dim % 4 or bands % groups or not 0 < beta/depth <= 1:
            raise ValueError("Invalid dimensions or residual bound")
        # The depth-1 control uses alpha=1 under the same beta/depth rule.
        # Its conditional lower bound is zero; the strict invertibility
        # certificate applies only when alpha<1, as in the depth-8 model.
        self.bands, self.dim, self.groups = bands, dim, groups
        self.embedding = nn.Embedding(bands, dim//4)
        nn.init.normal_(self.embedding.weight, std=.5)
        self.head = nn.Linear(dim+1, 20)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        self.layers = nn.ModuleList([RaceLayer(dim, depth, beta, cf_credit) for _ in range(depth)])
        self.register_buffer("time_constants", torch.tensor([.05, .2, .8]))
        self.register_buffer("center", torch.zeros(dim+1))
        self.register_buffer("scale", torch.ones(dim+1))
        self.register_buffer("whitener", torch.eye(dim+1))

    def forward(self, b, t, c, ids, size, sequential=False, overrides=None, trace=False):
        phi = torch.cat((t.new_ones((len(t), 1)), torch.exp(-t[:, None]/self.time_constants)), -1)
        x = (self.embedding(b)[:, :, None]*phi[:, None, :]).flatten(1)
        original_t = t
        width = self.bands//self.groups
        stats, traces = [], []
        for j, layer in enumerate(self.layers):
            keys = ids*(self.groups+1) + (b + (width//2 if j%2 else 0))//width
            x, t, st, tr = layer(x, t, c, keys, sequential,
                                 (overrides or {}).get(j), trace)
            stats.append(st)
            if trace:
                traces.append(tr)
        mass = x.new_zeros(size).index_add(0, ids, c)
        mean = x.new_zeros((size, self.dim)).index_add(0, ids, x*c[:, None])/mass[:, None]
        summary = torch.cat((mean, torch.log1p(mass[:, None])/10), -1)
        features = ((summary-self.center)/self.scale) @ self.whitener
        logits = self.head(features)
        return logits, summary, {"layers": stats, "packets": len(t),
                                 "max_payload": float(x.detach().abs().max()),
                                 "mean_added_delay_ms": float((t-original_t).detach().mean()*1000)}, traces


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
