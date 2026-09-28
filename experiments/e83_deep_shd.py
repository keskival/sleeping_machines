"""E83: depth and gradient-flow study for Sleeping Machines on spoken digits.

All layers are event layers.  Layer 1 uses local tonotopic wiring; each later
layer receives sparse messages from every earlier hidden layer plus a local
skip from the raw cochlear events.  The readout can inspect every layer's
emissions, so early features keep a direct path to the loss as depth grows.

This isolates the fixed-topology depth question from the separate speaker
equivariance question in E75.  It reports per-layer gradient norms, active
messages, and spikes; the small pilot is not a benchmark claim.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
from e71_event_cde import events  # noqa: E402
from e74_time_vector_net import TVLayer, to_events  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e83")


class DeepSHD(nn.Module):
    def __init__(self, bands, d, n, M1, M, depth, window, fan2, dmax, w_sd, seed=0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(bands, d)
        self.widths = [M1] + [M] * (depth - 1)
        self.layers = nn.ModuleList()
        centers = (torch.arange(M1) + 0.5) * bands / M1
        local = (torch.arange(bands)[:, None] - centers[None]).abs() <= window / 2
        self.layers.append(TVLayer(bands, M1, d, d, n, dmax, local, True, w_sd[0]))
        for i in range(1, depth):
            previous_width = sum(self.widths[:i])
            mask = torch.zeros(previous_width + bands, M, dtype=torch.bool)
            mask[:previous_width] = torch.rand(previous_width, M, generator=gen) < fan2
            centers = (torch.arange(M) + 0.5) * bands / M
            raw_local = (torch.arange(bands)[:, None] - centers[None]).abs() <= window / 2
            mask[previous_width:] = raw_local
            self.layers.append(TVLayer(previous_width + bands, M, d, d, n, dmax, mask, True, w_sd[1]))
        self.ro = TVLayer(sum(self.widths), 20, d, d, n, dmax, None, False, 0.3,
                          gate_bias=1.0, normalize=True)

    def forward(self, eb, ei, et, B, G):
        raw_v = self.emb(ei)
        emitted = []
        messages, spikes = [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, ei, et, raw_v
            else:
                bs, js, ts, vs = [], [], [], []
                offset = 0
                for (ob, oj, ot, ov), width in zip(emitted, self.widths[:i]):
                    bs.append(ob); js.append(oj + offset); ts.append(ot); vs.append(ov)
                    offset += width
                bs.append(eb); js.append(ei + offset); ts.append(et); vs.append(raw_v)
                ib, ij, it, iv = (torch.cat(v) for v in (bs, js, ts, vs))
            out, msg = layer(ib, ij, it, iv, B, G)
            emitted.append(out); messages.append(msg); spikes.append(len(out[2]) / B)

        rb, rj, rt, rv = [], [], [], []
        offset = 0
        for (ob, oj, ot, ov), width in zip(emitted, self.widths):
            rb.append(ob); rj.append(oj + offset); rt.append(ot); rv.append(ov)
            offset += width
        V, msg = self.ro(torch.cat(rb), torch.cat(rj), torch.cat(rt), torch.cat(rv), B, G)
        messages.append(msg)
        p = torch.softmax(V[::4], -1).mean(0)
        return p, {"msgs": messages, "spikes": spikes}


def grad_norm(module):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                         if p.grad is not None))


def stratified_limit(data, limit, rng):
    if not limit or limit >= len(data):
        return data
    by_class = {}
    for i, item in enumerate(data):
        by_class.setdefault(int(item[-1]), []).append(i)
    each, extra = divmod(limit, len(by_class))
    chosen = []
    leftovers = []
    for key in sorted(by_class):
        ids = np.asarray(by_class[key])
        rng.shuffle(ids)
        chosen.extend(ids[:each].tolist())
        leftovers.extend(ids[each:].tolist())
    if extra:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:extra])
    rng.shuffle(chosen)
    return [data[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=140)
    ap.add_argument("--merge", type=float, default=0.002)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M1", type=int, default=64)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--dmax", type=float, default=50.0)
    ap.add_argument("--w_sd", default="0.05,0.05")
    ap.add_argument("--shift", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--eval_limit", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()

    def load(split, part):
        return [(*events(t, u, a.merge), y) for t, u, y in S.utterances(split, a.bands, part) if len(t) > 1]

    tr = stratified_limit(load("train", "fit_spk"), a.limit, rng)
    ev = stratified_limit(load("train", "val_spk"), a.eval_limit, rng)
    widths_sd = [float(v) for v in a.w_sd.split(",")]
    if len(widths_sd) != 2:
        raise ValueError("--w_sd requires two comma-separated values")
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2, a.dmax,
                  widths_sd, a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    nb = math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()),
           "layer_params": [sum(p.numel() for p in l.parameters()) for l in net.layers],
           "train": len(tr), "eval": len(ev), "curve": []}
    print(json.dumps({"train": len(tr), "eval": len(ev), "params": res["params"],
                      "layer_params": res["layer_params"], "load_s": round(time.time() - t0)}), flush=True)
    for ep in range(a.epochs):
        net.train(); tl = 0.0; perm = rng.permutation(len(tr)); gsum = np.zeros(a.depth)
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, y, tmax = to_events(items, a.bands, a.shift, rng, a.drop)
            p, _ = net(eb, ei, et, len(items), tmax + 2 * int(a.dmax) + 60)
            loss = -torch.log(p[torch.arange(len(y)), y] + 1e-8).mean()
            opt.zero_grad(); loss.backward()
            gsum += np.asarray([grad_norm(l) for l in net.layers])
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss.detach())

        net.eval(); ok = 0; st = {"msgs": np.zeros(a.depth + 1), "spikes": np.zeros(a.depth)}
        for layer in net.layers:
            layer.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, y, tmax = to_events(items, a.bands, 0, rng, 0.0)
                p, info = net(eb, ei, et, len(items), tmax + 2 * int(a.dmax) + 60)
                ok += int((p.argmax(1) == y).sum())
                for key in st:
                    st[key] += np.asarray(info[key]) * len(items)
        send = [round(float(l.sent[l.mask].float().mean()), 3) for l in net.layers]
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4),
               "spk_acc": round(ok / len(ev), 4),
               "msgs_per_utt": (st["msgs"] / len(ev)).round(0).tolist(),
               "spikes_per_utt": (st["spikes"] / len(ev)).round(0).tolist(),
               "synapses_sending": send,
               "layer_grad_norms": (gsum / nb).round(5).tolist(),
               "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    path = os.path.join(OUT, f"deep_d{a.d}_n{a.n}_M{a.M1}-{a.M}_depth{a.depth}_spk_s{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
