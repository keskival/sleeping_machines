"""E84: depth and gradient-flow study for event-native market world models.

The core is E80's time-vector event layer, now stacked at configurable depth.
Each hidden stage receives sparse messages only from the previous stage.
Intermediate stages get auxiliary point-process losses through a shared
readout; the deployed prediction uses only the deepest stage. The pilot selects
on day 5 and reports predictive log likelihood; it makes no trading-edge claim.
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
    def __init__(self, nf, nb, d, n, M, Ms, depth, fan1, fan2, readout_fan, dmax, seed=0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(NT, d)
        self.tf = nn.Linear(nf, d)
        self.M, self.depth = M, depth
        self.layers = nn.ModuleList()
        mask0 = torch.rand(NT, M, generator=gen) < fan1
        for j in range(M):
            if not mask0[:, j].any():
                mask0[torch.randint(NT, (), generator=gen), j] = True
        for i0 in range(NT):
            if not mask0[i0].any():
                mask0[i0, torch.randint(M, (), generator=gen)] = True
        self.layers.append(A.TVLayerW(NT, M, d, d, n, dmax, mask0, True, 0.05))
        for i in range(1, depth):
            mask = torch.rand(M, M, generator=gen) < fan2
            for j in range(M):
                if not mask[:, j].any():
                    mask[torch.randint(M, (), generator=gen), j] = True
            for i0 in range(M):
                if not mask[i0].any():
                    mask[i0, torch.randint(M, (), generator=gen)] = True
            self.layers.append(A.TVLayerW(M, M, d, d, n, dmax, mask, True, 0.05))
        readout_mask = torch.rand(M, Ms, generator=gen) < readout_fan
        for j in range(Ms):
            if not readout_mask[:, j].any():
                readout_mask[torch.randint(M, (), generator=gen), j] = True
        for i0 in range(M):
            if not readout_mask[i0].any():
                readout_mask[i0, torch.randint(Ms, (), generator=gen)] = True
        self.st = A.TVLayerW(M, Ms, d, d, n, dmax, readout_mask, False, 0.3, gate_bias=1.0)
        self.norm = nn.LayerNorm(Ms)
        self.head = nn.Linear(Ms, nb * NT)
        self.nb = nb

    @staticmethod
    def scored_pairs(layer, indices):
        if layer.mask is None:
            return len(indices) * layer.M
        return int(layer.mask[indices].sum())

    def forward(self, y, x, return_taps=False):
        B, L = y.shape; G = L + 1
        gap = (torch.exp(5 * x[..., 0]) - 1e-4).clamp(min=0)
        w = torch.zeros(G, B); w[1:] = gap.T
        raw_v = self.emb(y) + self.tf(x)
        eb = torch.arange(B).repeat_interleave(L)
        raw_i = y.reshape(-1)
        raw_t = torch.arange(L).float().repeat(B) + 1 - 1e-3
        raw_vf = raw_v.reshape(B * L, -1)
        outputs, messages, spikes, candidates = [], [], [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, raw_i, raw_t, raw_vf
            else:
                ib, ij, it, iv = outputs[-1]
            candidates.append(self.scored_pairs(layer, ij if i else raw_i))
            out, msg = layer(ib, ij, it, iv, B, G, w)
            outputs.append(out); messages.append(msg); spikes.append(len(out[2]) / (B * L))

        taps = []
        readout_inputs = outputs if return_taps else outputs[-1:]
        for out in readout_inputs:
            candidates.append(self.scored_pairs(self.st, out[1]))
            V, msg = self.st(out[0], out[1], out[2], out[3], B, G, w)
            messages.append(msg)
            h = self.norm(V[1:].permute(1, 0, 2))
            taps.append(nn.functional.softplus(self.head(h)).view(B, L, self.nb, NT) + 1e-6)
        deep_messages = sum(messages[:self.depth]) + messages[-1]
        deep_candidates = sum(candidates[:self.depth]) + candidates[-1]
        self.work = {"msgs_per_event": [m / L for m in messages],
                     "candidate_scores_per_event": [c / L for c in candidates],
                     "deep_msgs_per_event": deep_messages / L,
                     "deep_candidate_scores_per_event": deep_candidates / L,
                     "spikes_per_event": spikes,
                     "state_vector_updates_per_event": G * self.layers[0].n *
                         (self.depth * self.M + len(readout_inputs) * self.st.M) / L,
                     "deep_state_vector_updates_per_event": G * self.layers[0].n *
                         (self.depth * self.M + self.st.M) / L}
        return (taps[-1], taps) if return_taps else taps[-1]

    @staticmethod
    def ll_rates(rates, b, span, yn):
        at = rates.gather(2, b[..., None, None].expand(*b.shape, 1, NT)).squeeze(2)
        return torch.log(at.gather(2, yn[..., None]).squeeze(2)) - (rates.sum(3) * span).sum(2)

    def ll(self, y, x, b, span, yn):
        return self.ll_rates(self(y, x), b, span, yn)


def grad_norm(module):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                         if p.grad is not None))


def route_gradient_probe(final_loss, aux_loss, layers):
    groups = [list(layer.parameters()) for layer in layers]
    params = [p for group in groups for p in group]
    final_grads = torch.autograd.grad(final_loss, params, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)
    out = {"final_norms": [], "aux_norms": [], "cosines": []}
    start = 0
    for group in groups:
        stop = start + len(group)
        gf, ga = final_grads[start:stop], aux_grads[start:stop]
        final_sq = sum(float(g.detach().square().sum()) for g in gf if g is not None)
        aux_sq = sum(float(g.detach().square().sum()) for g in ga if g is not None)
        dot = sum(float((f.detach() * a.detach()).sum()) for f, a in zip(gf, ga)
                  if f is not None and a is not None)
        fn, an = math.sqrt(final_sq), math.sqrt(aux_sq)
        out["final_norms"].append(round(fn, 6))
        out["aux_norms"].append(round(an, 6))
        out["cosines"].append(round(dot / (fn * an), 6) if fn and an else None)
        start = stop
    return out


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
    ap.add_argument("--fan1", type=float, default=0.5)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--readout_fan", type=float, default=0.5)
    ap.add_argument("--aux_weight", type=float, default=0.2,
                    help="weight per intermediate supervised readout; zero is the no-auxiliary control")
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
                   a.Ms, a.depth, a.fan1, a.fan2, a.readout_fan, a.dmax, a.seed)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    tag = f"val_L{a.L}_f{a.fine}_x{a.cross}_m{a.mag}_d{a.d}_M{a.M}_e{a.epochs}_depth{a.depth}_aux{a.aux_weight:g}_s{a.seed}"
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()),
           "layer_params": [sum(p.numel() for p in l.parameters()) for l in net.layers],
           "train_days": train_days, "validation_day": eval_days[0],
           "training_windows": len(chunks), "epochs": []}
    print(json.dumps({"params": res["params"], "layer_params": res["layer_params"],
                      "training_windows": len(chunks), "load_s": round(time.time() - t0, 1)}), flush=True)
    best = (-float("inf"), None, 0)
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(len(chunks)); loss_sum = 0.0; aux_sum = 0.0
        route_probe = None
        gsum = np.zeros(a.depth); n_updates = 0
        for i0 in range(0, len(order), a.batch):
            parts = [[e[s:s + a.L] for e in tr[di][2]]
                     for di, s in (chunks[k] for k in order[i0:i0 + a.batch])]
            y, x, b, sp, yn = (torch.stack([p[j] for p in parts]) for j in range(5))
            rates, tap_rates = net(y, x, return_taps=True)
            main_loss = -net.ll_rates(rates, b, sp, yn).mean()
            aux_losses = [-net.ll_rates(tap, b, sp, yn).mean() for tap in tap_rates[:-1]]
            aux_loss = sum(aux_losses) if aux_losses else main_loss.new_zeros(())
            if i0 == 0:
                route_probe = route_gradient_probe(main_loss, aux_loss, net.layers)
            loss = main_loss + a.aux_weight * aux_loss
            opt.zero_grad(); loss.backward()
            gsum += np.asarray([grad_norm(l) for l in net.layers]); n_updates += 1
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            loss_sum += float(main_loss.detach()); aux_sum += float(aux_loss.detach())
        training_work = net.work
        net.eval()
        score = T52.score_day(net, ev[0][2], a.L)[0]
        row = {"epoch": ep + 1, "train_nll": round(loss_sum / max(n_updates, 1), 5),
               "aux_train_nll": round(aux_sum / max(n_updates, 1), 5),
               "first_batch_gradient_routes": route_probe,
               "val_day5_nats_per_event": float(score),
               "layer_grad_norms": (gsum / max(n_updates, 1)).round(5).tolist(),
               "training_work": training_work, "inference_work": net.work,
               "wall_s": round(time.time() - t0, 1)}
        res["epochs"].append(row); print(json.dumps(row), flush=True)
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
    res["best_epoch"] = best[2]; res["best_day5_nats_per_event"] = best[0]
    path = os.path.join(OUT, f"tv_market_{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
