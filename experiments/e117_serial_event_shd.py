"""E117: bounded serial event carriers, local lazy memory, deepest-only SHD loss.

One continuation per input packet per layer; no firing threshold or clock grid.
Training uses a segmented associative scan over actual arrivals (O(E log E)
work), while the same memory recurrence has O(E) online work. Fixed alternating
tonotopic partitions are a capacity diagnostic, not learned sparse routing.
See THEORY §§166–168. All reported scores are training-speaker/development;
this script never opens the SHD test split.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import e51_shd_world as S

torch.set_num_threads(1)
OUT = Path(__file__).parent / "results" / "e117"


def packets(t, u, window):
    """Fixed, half-open bins; emit nonempty band packets at their closing time.

    A count is never attached to an earlier spike. Origin is the recorded
    utterance start, independent of its future duration. No empty-bin work.
    """
    bins = np.floor(t / window).astype(np.int64)
    pairs, counts = np.unique(np.stack((bins, u), axis=1), axis=0, return_counts=True)
    return (pairs[:, 1].astype(np.int64),
            ((pairs[:, 0] + 1) * window).astype(np.float32), counts.astype(np.float32))


def load_items(bands, window, limit, part, seed):
    with h5py.File(os.path.join(S.ROOT, "shd_train.h5"), "r") as f:
        held_out = np.isin(np.array(f["extra"]["speaker"]), S.VAL_SPEAKERS)
        size = int((held_out if part == "val_spk" else ~held_out).sum())
    order = np.random.default_rng(seed).permutation(size)[:limit]
    selected = set(order.tolist())
    items = {}
    for index, (t, u, y) in enumerate(S.utterances("train", bands, part)):
        if index not in selected:
            continue
        if not len(t):
            raise ValueError("Empty utterance requires an explicit silent example policy")
        b, ts, c = packets(t, u, window)
        items[index] = (b, ts, c, y, int(index))
        if len(items) == len(selected):
            break
    return [items[int(index)] for index in order]


def batch(items):
    bands = torch.from_numpy(np.concatenate([x[0] for x in items]))
    times = torch.from_numpy(np.concatenate([x[1] for x in items]))
    counts = torch.from_numpy(np.concatenate([x[2] for x in items]))
    ids = torch.repeat_interleave(torch.arange(len(items)),
                                  torch.tensor([len(x[0]) for x in items]))
    labels = torch.tensor([x[3] for x in items])
    return bands, times, counts, ids, labels


def segmented_memory(x, times, counts, keys, taus, sequential=False):
    """Exact exponential numerator/mass recurrence, reset at each receiver.

    Stable key sort retains deterministic (time, band) order at tied times.
    Online execution must apply that same tie policy. A packet depends only
    on arrivals already processed in its receiver, including itself.
    """
    order = torch.argsort(keys, stable=True)
    k, t, c, h = keys[order], times[order], counts[order], x[order]
    first = torch.cat((torch.ones(1, dtype=torch.bool), k[1:] != k[:-1]))
    dt = torch.diff(t, prepend=t[:1]).clamp_min(0)
    decay = torch.exp(-dt[:, None] / taus[None, :]) * (~first[:, None])
    values = torch.cat((h, h.new_ones((len(h), 1))), -1) * c[:, None]
    z = values[:, None, :].expand(-1, len(taus), -1)
    work = 0
    if sequential:
        state = z.new_zeros(z.shape[1:])
        rows = []
        for j in range(len(z)):
            state = decay[j, :, None] * state + z[j]
            rows.append(state)
        z = torch.stack(rows)
        work = len(z)
    else:
        # A segmented affine monoid: a zero at a receiver boundary prevents
        # any contribution from a different receiver at every scan scale.
        a = decay
        step = 1
        while step < len(z):
            z = torch.cat((z[:step], z[step:] + a[step:, :, None] * z[:-step]), 0)
            a = torch.cat((a[:step], a[step:] * a[:-step]), 0)
            work += len(z) - step
            step *= 2
    mass = z[:, :, -1]
    memory = z[:, :, :-1] / (mass[:, :, None] + 1e-4)
    inverse = torch.argsort(order)
    return memory[inverse], mass[inverse], work


class CarrierLayer(nn.Module):
    def __init__(self, dim, depth, beta, banks=3):
        super().__init__()
        self.alpha = beta / max(depth, 1)
        self.mix = nn.Linear(dim * (banks + 1) + banks, dim)
        nn.init.normal_(self.mix.weight, std=0.03)
        nn.init.zeros_(self.mix.bias)
        tau = torch.logspace(math.log10(.02), math.log10(.8), banks)
        self.log_tau = nn.Parameter(tau.log())

    def weight(self):
        # Induced infinity norm <= 1, including all memory banks together.
        w = self.mix.weight
        return w / w.abs().sum(-1, keepdim=True).clamp_min(1)

    def forward(self, x, t, c, keys, sequential=False):
        mem, mass, scan_work = segmented_memory(
            x, t, c, keys, self.log_tau.clamp(math.log(.002), math.log(4.)).exp(), sequential)
        features = torch.cat((x, mem.flatten(1), mass / (1 + mass)), -1)
        branch = torch.tanh(F.linear(features, self.weight(), self.mix.bias))
        return x + self.alpha * branch, scan_work


class SerialEventNet(nn.Module):
    def __init__(self, bands, dim, depth, groups, beta):
        super().__init__()
        if dim % 4 or bands % groups:
            raise ValueError("dim must divide by four, bands by groups")
        if depth and not 0 < beta / depth < 1:
            raise ValueError("Residual increment must be in (0,1) for the bound")
        self.bands, self.groups, self.dim = bands, groups, dim
        self.embedding = nn.Embedding(bands, dim // 4)
        nn.init.normal_(self.embedding.weight, std=.5)
        self.layers = nn.ModuleList([CarrierLayer(dim, depth, beta) for _ in range(depth)])
        self.head = nn.Linear(dim + 1, 20)
        self.register_buffer("time_constants", torch.tensor([.05, .2, .8]))

    def forward(self, b, t, c, ids, size, bypass=False, return_features=False, sequential=False):
        # An explicit time×address input representation preserves coarse
        # temporal frequency structure in the shallow control too.
        phi = torch.cat((t.new_ones((len(t), 1)), torch.exp(-t[:, None] / self.time_constants)), -1)
        x = (self.embedding(b)[:, :, None] * phi[:, None, :]).flatten(1)
        width = self.bands // self.groups
        key_sets = []
        for offset in (0, width // 2):
            group = (b + offset) // width
            key_sets.append(ids * (self.groups + 1) + group)
        scan_work = 0
        features = []
        mass = x.new_zeros(size).index_add(0, ids, c)

        def pooled(h):
            mean = h.new_zeros((size, self.dim)).index_add(0, ids, h * c[:, None]) / mass[:, None]
            return torch.cat((mean, torch.log1p(mass[:, None]) / 10), -1)

        if return_features:
            features.append(pooled(x))
        if not bypass:
            for j, layer in enumerate(self.layers):
                x, work = layer(x, t, c, key_sets[j % 2], sequential)
                scan_work += work
                if return_features:
                    features.append(pooled(x))
        summary = pooled(x)
        stats = {"input_packets": len(t), "carrier_updates": len(t) * (0 if bypass else len(self.layers)),
                 "scan_compositions": scan_work, "max_payload": float(x.detach().abs().max())}
        return self.head(summary), stats, features


@torch.no_grad()
def evaluate(net, items, bs, bypass=False):
    correct = 0
    loss = 0.
    predictions = []
    totals = {"input_packets": 0, "carrier_updates": 0, "scan_compositions": 0, "max_payload": 0.}
    for start in range(0, len(items), bs):
        inputs = batch(items[start:start + bs])
        logits, stats, _ = net(*inputs[:4], len(inputs[-1]), bypass=bypass)
        y = inputs[-1]
        loss += float(F.cross_entropy(logits, y, reduction="sum"))
        pred = logits.argmax(-1)
        correct += int((pred == y).sum())
        predictions.extend(pred.tolist())
        for key, value in stats.items():
            totals[key] = max(totals[key], value) if key == "max_payload" else totals[key] + value
    return {"correct": correct, "n": len(items), "accuracy": correct / len(items),
            "nll": loss / len(items), "predictions": predictions, **totals}


def contracts(net):
    """Small analytical witnesses before the scientific diagnostic."""
    gen = torch.Generator().manual_seed(117)
    b = torch.tensor([0, 1, 5, 7, 2, 10, 1, 4, 0])
    t = torch.tensor([.01, .01, .02, .04, .06, .08, .02, .04, .06])
    c = torch.tensor([1., 3., 2., 1., 4., 1., 2., 1., 3.])
    ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    # Independent serial recurrence, not a second copy of the scan.
    a = net(b, t, c, ids, 2)[0]
    s = net(b, t, c, ids, 2, sequential=True)[0]
    scan_error = float((a - s).detach().abs().max())
    if scan_error > 2e-6:
        raise AssertionError(f"Scan/serial mismatch {scan_error}")
    layer = CarrierLayer(4, 8, 1.).double()
    x = torch.randn(9, 4, dtype=torch.float64, generator=gen, requires_grad=True)
    keys = ids * 3 + b // 5
    td, cd = t.double(), c.double()
    y, _ = layer(x, td, cd, keys)
    memory = segmented_memory(x, td, cd, keys, layer.log_tau.exp())[0]
    mass_bound_slack = float(x.detach().abs().max() - memory.detach().abs().max())
    v = torch.randn(x.shape, dtype=x.dtype, generator=gen)
    jv = torch.autograd.functional.jvp(lambda h: layer(h, td, cd, keys)[0], x, v)[1]
    eps = 1e-5
    fd = (layer(x + eps*v, td, cd, keys)[0] - layer(x - eps*v, td, cd, keys)[0]) / (2*eps)
    derivative_error = float((jv-fd).detach().abs().max())
    ratio = float(jv.abs().max() / v.abs().max())
    assert mass_bound_slack >= -1e-10 and derivative_error < 1e-7
    assert 1-layer.alpha-1e-9 <= ratio <= 1+layer.alpha+1e-9
    prefix = 4
    prefix_error = float((layer(x[:prefix], td[:prefix], cd[:prefix], keys[:prefix])[0]
                          - y[:prefix]).detach().abs().max())
    assert prefix_error < 1e-9
    pb, pt, pc = packets(np.array([.001, .009, .011]), np.array([0, 0, 0]), .01)
    assert np.allclose(pt, [.01, .02]) and np.array_equal(pc, [2, 1])
    return {"scan_serial_error": scan_error, "memory_bound_slack": mass_bound_slack,
            "payload_jvp_finite_difference_error": derivative_error,
            "payload_jvp_supnorm_ratio": ratio, "prefix_error": prefix_error,
            "packet_close_times": pt.tolist(), "packet_counts": pc.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--bands", type=int, default=40)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--window", type=float, default=.01)
    ap.add_argument("--beta", type=float, default=1.)
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
    output = OUT / (a.tag + ".json")
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    torch.manual_seed(a.seed)
    net = SerialEventNet(a.bands, a.dim, a.depth, a.groups, a.beta)
    contract = contracts(net)
    fit = load_items(a.bands, a.window, a.limit, "fit_spk", a.seed)
    dev = load_items(a.bands, a.window, a.eval_limit, "val_spk", a.seed + 1)
    rng = np.random.default_rng(a.seed + 2)
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    result = {"args": vars(a), "status": "running", "contracts": contract,
              "parameters": sum(p.numel() for p in net.parameters()),
              "data_protocol": "SHD train only; speakers 3/6 held out; no augmentation",
              "fit_ids": [x[4] for x in fit], "dev_ids": [x[4] for x in dev],
              "fit_labels": [x[3] for x in fit], "dev_labels": [x[3] for x in dev],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "curve": []}
    lower = (1 - a.beta / a.depth)**a.depth if a.depth else 1.
    upper = (1 + a.beta / a.depth)**a.depth if a.depth else 1.
    result["fixed_address_payload_supnorm_gain_bounds"] = [lower, upper]
    result["initial"] = {"fit": evaluate(net, fit, a.bs), "dev": evaluate(net, dev, a.bs)}
    print(json.dumps({"initial_fit_nll": result["initial"]["fit"]["nll"],
                      "initial_dev_accuracy": result["initial"]["dev"]["accuracy"],
                      "parameters": result["parameters"], "contracts": contract}), flush=True)
    for epoch in range(1, a.epochs + 1):
        grad_sums = np.zeros(a.depth)
        train_loss = 0.
        batches = 0
        for start in range(0, len(fit), a.bs):
            if start == 0:
                order = rng.permutation(len(fit))
            items = [fit[j] for j in order[start:start+a.bs]]
            inputs = batch(items)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = net(*inputs[:4], len(items))
            loss = F.cross_entropy(logits, inputs[-1])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite terminal loss")
            loss.backward()
            for j, layer in enumerate(net.layers):
                grad_sums[j] += math.sqrt(sum(float(p.grad.square().sum()) for p in layer.parameters()
                                               if p.grad is not None))
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            train_loss += float(loss.detach()) * len(items)
            batches += 1
        row = {"epoch": epoch, "training_online_nll": train_loss/len(fit),
               "fit": evaluate(net, fit, a.bs), "dev": evaluate(net, dev, a.bs),
               "layer_grad_norm": (grad_sums/batches).tolist(),
               "wall_s": time.time()-started}
        result["curve"].append(row)
        result["wall_s"] = time.time()-started
        result["max_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "fit_acc": row["fit"]["accuracy"],
                          "fit_nll": row["fit"]["nll"], "dev_acc": row["dev"]["accuracy"],
                          "dev_nll": row["dev"]["nll"], "layer_grad": row["layer_grad_norm"],
                          "max_payload": row["dev"]["max_payload"], "wall_s": row["wall_s"]}), flush=True)
    result["bypass_dev"] = evaluate(net, dev, a.bs, bypass=True)
    result["bypass_fit"] = evaluate(net, fit, a.bs, bypass=True)
    result["status"] = "completed"
    result["wall_s"] = time.time()-started
    output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"args": vars(a), "state_dict": net.state_dict()}, OUT / (a.tag + ".pt"))
    print("completed", output, flush=True)


if __name__ == "__main__":
    main()
