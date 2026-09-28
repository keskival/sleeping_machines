"""E83: depth and gradient-flow study for Sleeping Machines on spoken digits.

All layers are event layers. Layer 1 uses local tonotopic wiring; each later
layer receives sparse messages only from the immediately preceding layer. A
shared auxiliary readout is trained at each depth, while inference uses only
the deepest readout. This supplies local learning signals without allowing the
prediction to bypass the event hierarchy.

This isolates the fixed-topology depth question from the separate speaker
equivariance question in E75. It reports per-layer gradient norms, candidate
score pairs, accepted messages, state scans, and spikes; the small pilot is not
a benchmark claim.
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
    def __init__(self, bands, d, n, M1, M, depth, window, fan2, readout_fan,
                 dmax, w_sd, seed=0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        self.depth = int(depth)
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(bands, d)
        self.widths = [M1] + [M] * (depth - 1)
        self.layers = nn.ModuleList()
        centers = (torch.arange(M1) + 0.5) * bands / M1
        local = (torch.arange(bands)[:, None] - centers[None]).abs() <= window / 2
        self.layers.append(TVLayer(bands, M1, d, d, n, dmax, local, True, w_sd[0]))
        for i in range(1, depth):
            previous_width = self.widths[i - 1]
            mask = torch.rand(previous_width, M, generator=gen) < fan2
            for j in range(M):
                if not mask[:, j].any():
                    mask[torch.randint(previous_width, (), generator=gen), j] = True
            for i0 in range(previous_width):
                if not mask[i0].any():
                    mask[i0, torch.randint(M, (), generator=gen)] = True
            self.layers.append(TVLayer(previous_width, M, d, d, n, dmax, mask, True, w_sd[1]))
        readout_width = max(self.widths)
        readout_mask = torch.rand(readout_width, 20, generator=gen) < readout_fan
        for j in range(20):
            if not readout_mask[:, j].any():
                readout_mask[torch.randint(readout_width, (), generator=gen), j] = True
        for i0 in range(readout_width):
            if not readout_mask[i0].any():
                readout_mask[i0, torch.randint(20, (), generator=gen)] = True
        self.ro = TVLayer(readout_width, 20, d, d, n, dmax, readout_mask, False, 0.3,
                          gate_bias=1.0, normalize=True)

    @staticmethod
    def scored_pairs(layer, indices):
        if layer.mask is None:
            return len(indices) * layer.M
        return int(layer.mask[indices].sum())

    def forward(self, eb, ei, et, B, G, return_taps=False):
        raw_v = self.emb(ei)
        emitted = []
        messages, spikes, candidates = [], [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, ei, et, raw_v
            else:
                ib, ij, it, iv = emitted[-1]
            candidates.append(self.scored_pairs(layer, ij))
            out, msg = layer(ib, ij, it, iv, B, G)
            emitted.append(out); messages.append(msg); spikes.append(len(out[2]) / B)

        taps = []
        readout_inputs = emitted if return_taps else emitted[-1:]
        for out in readout_inputs:
            candidates.append(self.scored_pairs(self.ro, out[1]))
            V, msg = self.ro(out[0], out[1], out[2], out[3], B, G)
            messages.append(msg)
            taps.append(torch.softmax(V[::4], -1).mean(0))
        scan_updates = G * self.layers[0].n * (sum(self.widths) + len(readout_inputs) * self.ro.M)
        deep_scan_updates = G * self.layers[0].n * (sum(self.widths) + self.ro.M)
        deep_candidates = sum(candidates[:self.depth]) + candidates[-1]
        deep_messages = sum(messages[:self.depth]) + messages[-1]
        return taps[-1], {"msgs": messages, "candidates": candidates,
                          "spikes": spikes, "tap_probs": taps,
                          "state_vector_updates_per_utt": scan_updates,
                          "deep_msgs_per_utt": deep_messages,
                          "deep_candidate_scores_per_utt": deep_candidates,
                          "deep_state_vector_updates_per_utt": deep_scan_updates}


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
    ap.add_argument("--readout_fan", type=float, default=0.5)
    ap.add_argument("--aux_weight", type=float, default=0.2,
                    help="weight per intermediate supervised readout; zero is the no-auxiliary control")
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
    net = DeepSHD(a.bands, a.d, a.n, a.M1, a.M, a.depth, a.window, a.fan2,
                  a.readout_fan, a.dmax, widths_sd, a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    nb = math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()),
           "layer_params": [sum(p.numel() for p in l.parameters()) for l in net.layers],
           "train": len(tr), "eval": len(ev), "curve": []}
    print(json.dumps({"train": len(tr), "eval": len(ev), "params": res["params"],
                      "layer_params": res["layer_params"], "load_s": round(time.time() - t0)}), flush=True)
    for ep in range(a.epochs):
        net.train(); tl = 0.0; aux_tl = 0.0; route_probe = None
        perm = rng.permutation(len(tr)); gsum = np.zeros(a.depth)
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, y, tmax = to_events(items, a.bands, a.shift, rng, a.drop)
            grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
            p, info = net(eb, ei, et, len(items), grid, return_taps=True)
            main_loss = -torch.log(p[torch.arange(len(y)), y] + 1e-8).mean()
            aux_losses = [-torch.log(tap[torch.arange(len(y)), y] + 1e-8).mean()
                          for tap in info["tap_probs"][:-1]]
            aux_loss = sum(aux_losses) if aux_losses else main_loss.new_zeros(())
            if i0 == 0:
                route_probe = route_gradient_probe(main_loss, aux_loss, net.layers)
            loss = main_loss + a.aux_weight * aux_loss
            opt.zero_grad(); loss.backward()
            gsum += np.asarray([grad_norm(l) for l in net.layers])
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(main_loss.detach()); aux_tl += float(aux_loss.detach())

        net.eval(); ok = 0; tap_ok = np.zeros(a.depth)
        st = {"msgs": np.zeros(a.depth * 2), "candidates": np.zeros(a.depth * 2),
              "spikes": np.zeros(a.depth), "state_vector_updates_per_utt": np.zeros(1),
              "deep_msgs_per_utt": np.zeros(1), "deep_candidate_scores_per_utt": np.zeros(1),
              "deep_state_vector_updates_per_utt": np.zeros(1)}
        for layer in net.layers:
            layer.sent.zero_()
        net.ro.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, y, tmax = to_events(items, a.bands, 0, rng, 0.0)
                grid = tmax + (a.depth + 1) * math.ceil(a.dmax) + 60
                p, info = net(eb, ei, et, len(items), grid, return_taps=True)
                ok += int((p.argmax(1) == y).sum())
                for k, tap in enumerate(info["tap_probs"]):
                    tap_ok[k] += int((tap.argmax(1) == y).sum())
                for key in st:
                    st[key] += np.asarray(info[key]) * len(items)
        send = [round(float(l.sent[l.mask].float().mean()), 3) for l in net.layers]
        row = {"epoch": ep + 1, "train_loss": round(tl / nb, 4),
               "spk_acc": round(ok / len(ev), 4),
               "tap_acc": (tap_ok / len(ev)).round(4).tolist(),
               "aux_train_loss": round(aux_tl / nb, 4),
               "first_batch_gradient_routes": route_probe,
               "msgs_per_utt": (st["msgs"] / len(ev)).round(0).tolist(),
               "candidate_scores_per_utt": (st["candidates"] / len(ev)).round(0).tolist(),
               "state_vector_updates_per_utt": int(st["state_vector_updates_per_utt"][0] / len(ev)),
               "deep_msgs_per_utt": round(float(st["deep_msgs_per_utt"][0] / len(ev)), 1),
               "deep_candidate_scores_per_utt": int(st["deep_candidate_scores_per_utt"][0] / len(ev)),
               "deep_state_vector_updates_per_utt": int(st["deep_state_vector_updates_per_utt"][0] / len(ev)),
               "spikes_per_utt": (st["spikes"] / len(ev)).round(0).tolist(),
               "synapses_sending": send,
               "layer_grad_norms": (gsum / nb).round(5).tolist(),
               "wall_s": round(time.time() - t0)}
        res["curve"].append(row); print(json.dumps(row), flush=True)
    path = os.path.join(OUT, f"deep_d{a.d}_n{a.n}_M{a.M1}-{a.M}_depth{a.depth}_aux{a.aux_weight:g}_spk_s{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
