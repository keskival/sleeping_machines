"""E84: depth and gradient-flow study for event-native market world models.

The core is E80's time-vector event layer, now stacked at configurable depth.
Every stage receives sparse messages from all earlier hidden event streams and
a direct route from the raw market events.  The readout sees every stage.  The
pilot selects on day 5 only and reports predictive log likelihood; it makes no
trading-edge claim.
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
import e52_thp as T52  # noqa: E402
import e80_tv_market as A  # noqa: E402
from e44_tpp import NT  # noqa: E402
from e17_market import PILOT, load_day  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e84")


class DeepTVPP(nn.Module):
    ll = T52.THP.ll

    def __init__(self, nf, nb, d, n, M, Ms, depth, fan2, dmax, seed=0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(NT, d)
        self.tf = nn.Linear(nf, d)
        self.M, self.depth = M, depth
        self.layers = nn.ModuleList()
        self.layers.append(A.TVLayerW(NT, M, d, d, n, dmax, None, True, 0.05))
        for i in range(1, depth):
            previous_width = i * M
            mask = torch.zeros(previous_width + NT, M, dtype=torch.bool)
            mask[:previous_width] = torch.rand(previous_width, M, generator=gen) < fan2
            mask[previous_width:] = True
            self.layers.append(A.TVLayerW(previous_width + NT, M, d, d, n, dmax,
                                          mask, True, 0.05))
        self.st = A.TVLayerW(depth * M, Ms, d, d, n, dmax, None, False, 0.3, gate_bias=1.0)
        self.norm = nn.LayerNorm(Ms)
        self.head = nn.Linear(Ms + d, nb * NT)
        self.nb = nb

    def forward(self, y, x):
        B, L = y.shape; G = L + 1
        gap = (torch.exp(5 * x[..., 0]) - 1e-4).clamp(min=0)
        w = torch.zeros(G, B); w[1:] = gap.T
        raw_v = self.emb(y) + self.tf(x)
        eb = torch.arange(B).repeat_interleave(L)
        raw_i = y.reshape(-1)
        raw_t = torch.arange(L).float().repeat(B) + 1 - 1e-3
        raw_vf = raw_v.reshape(B * L, -1)
        outputs, messages, spikes = [], [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, raw_i, raw_t, raw_vf
            else:
                bs, js, ts, vs = [], [], [], []
                offset = 0
                for (ob, oj, ot, ov) in outputs:
                    bs.append(ob); js.append(oj + offset); ts.append(ot); vs.append(ov)
                    offset += self.M
                bs.append(eb); js.append(raw_i + offset); ts.append(raw_t); vs.append(raw_vf)
                ib, ij, it, iv = (torch.cat(v) for v in (bs, js, ts, vs))
            out, msg = layer(ib, ij, it, iv, B, G, w)
            outputs.append(out); messages.append(msg); spikes.append(len(out[2]) / (B * L))

        rb, rj, rt, rv = [], [], [], []
        offset = 0
        for (ob, oj, ot, ov) in outputs:
            rb.append(ob); rj.append(oj + offset); rt.append(ot); rv.append(ov)
            offset += self.M
        V, msg = self.st(torch.cat(rb), torch.cat(rj), torch.cat(rt), torch.cat(rv), B, G, w)
        messages.append(msg)
        h = torch.cat([self.norm(V[1:].permute(1, 0, 2)), raw_v], -1)
        self.work = {"msgs_per_event": [m / L for m in messages],
                     "spikes_per_event": spikes}
        return nn.functional.softplus(self.head(h)).view(B, L, self.nb, NT) + 1e-6


def grad_norm(module):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                         if p.grad is not None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--fine", type=int, default=1)
    ap.add_argument("--cross", type=int, default=0)
    ap.add_argument("--mag", type=int, default=0)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--Ms", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--dmax", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunks_per_day", type=int, default=128,
                    help="random training windows per day; 0 uses every window")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.L < 2 or a.batch < 1:
        raise ValueError("--L must be >= 2 and --batch must be positive")
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t0 = time.time()
    qs = np.concatenate([load_day(d)[2][::50] for d in PILOT])
    big_q = float(np.quantile(qs, 0.99)); gaps = T52.bank(a.fine)
    train_days = list(PILOT[:4]); eval_days = [PILOT[4]]
    tr = [A.encode_day(d, big_q, gaps, a.cross, a.mag) for d in train_days]
    ev = [A.encode_day(d, big_q, gaps, a.cross, a.mag) for d in eval_days]
    chunks = []
    for di, day in enumerate(tr):
        day_chunks = [(di, s) for s in T52.windows(len(day[2][0]) - 1, a.L, a.L)]
        if a.chunks_per_day and len(day_chunks) > a.chunks_per_day:
            take = rng.choice(len(day_chunks), a.chunks_per_day, replace=False)
            day_chunks = [day_chunks[k] for k in sorted(take)]
        chunks.extend(day_chunks)
    net = DeepTVPP(tr[0][2][1].shape[1], len(gaps) + 1, a.d, a.n, a.M,
                   a.Ms, a.depth, a.fan2, a.dmax, a.seed)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    tag = f"val_L{a.L}_f{a.fine}_x{a.cross}_m{a.mag}_d{a.d}_M{a.M}_e{a.epochs}_depth{a.depth}_s{a.seed}"
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()),
           "layer_params": [sum(p.numel() for p in l.parameters()) for l in net.layers],
           "train_days": train_days, "validation_day": eval_days[0],
           "training_windows": len(chunks), "epochs": []}
    print(json.dumps({"params": res["params"], "layer_params": res["layer_params"],
                      "training_windows": len(chunks), "load_s": round(time.time() - t0, 1)}), flush=True)
    best = (-float("inf"), None, 0)
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(chunks)); loss_sum = 0.0
        gsum = np.zeros(a.depth); n_updates = 0
        for i0 in range(0, len(order), a.batch):
            parts = [[e[s:s + a.L] for e in tr[di][2]]
                     for di, s in (chunks[k] for k in order[i0:i0 + a.batch])]
            y, x, b, sp, yn = (torch.stack([p[j] for p in parts]) for j in range(5))
            loss = -net.ll(y, x, b, sp, yn).mean()
            opt.zero_grad(); loss.backward()
            gsum += np.asarray([grad_norm(l) for l in net.layers]); n_updates += 1
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            loss_sum += float(loss.detach())
        net.eval()
        score = T52.score_day(net, ev[0][2], a.L)[0]
        row = {"epoch": ep + 1, "train_nll": round(loss_sum / max(n_updates, 1), 5),
               "val_day5_nats_per_event": float(score),
               "layer_grad_norms": (gsum / max(n_updates, 1)).round(5).tolist(),
               "work": net.work, "wall_s": round(time.time() - t0, 1)}
        res["epochs"].append(row); print(json.dumps(row), flush=True)
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
    res["best_epoch"] = best[2]; res["best_day5_nats_per_event"] = best[0]
    path = os.path.join(OUT, f"tv_market_{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
