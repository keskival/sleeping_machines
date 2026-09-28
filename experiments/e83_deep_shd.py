"""E83: depth and gradient-flow study for Sleeping Machines on spoken digits.

All layers are event layers. Layer 1 uses local tonotopic wiring; each later
layer receives sparse messages only from the immediately preceding layer. A
shared auxiliary readout is trained at each depth, while inference uses only
the deepest readout. This supplies local learning signals without allowing the
prediction to bypass the event hierarchy.

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
    def __init__(self, bands, d, n, M1, M, depth, window, fan2, readout_fan,
                 dmax, w_sd, seed=0):
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
            previous_width = self.widths[i - 1]
            mask = torch.rand(previous_width, M, generator=gen) < fan2
            for j in range(M):
                if not mask[:, j].any():
                    mask[torch.randint(previous_width, (), generator=gen), j] = True
            self.layers.append(TVLayer(previous_width, M, d, d, n, dmax, mask, True, w_sd[1]))
        readout_width = max(self.widths)
        readout_mask = torch.rand(readout_width, 20, generator=gen) < readout_fan
        for j in range(20):
            if not readout_mask[:, j].any():
                readout_mask[torch.randint(readout_width, (), generator=gen), j] = True
        self.ro = TVLayer(readout_width, 20, d, d, n, dmax, readout_mask, False, 0.3,
                          gate_bias=1.0, normalize=True)

    def forward(self, eb, ei, et, B, G):
        raw_v = self.emb(ei)
        emitted = []
        messages, spikes = [], []
        for i, layer in enumerate(self.layers):
            if i == 0:
                ib, ij, it, iv = eb, ei, et, raw_v
            else:
                ib, ij, it, iv = emitted[-1]
            out, msg = layer(ib, ij, it, iv, B, G)
            emitted.append(out); messages.append(msg); spikes.append(len(out[2]) / B)

        taps = []
        for out in emitted:
            V, msg = self.ro(*out, B, G)
            messages.append(msg)
            taps.append(torch.softmax(V[::4], -1).mean(0))
        return taps[-1], {"msgs": messages, "spikes": spikes, "tap_probs": taps}


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
        net.train(); tl = 0.0; aux_tl = 0.0
        perm = rng.permutation(len(tr)); gsum = np.zeros(a.depth)
        for i0 in range(0, len(tr), a.bs):
            items = [tr[j] for j in perm[i0:i0 + a.bs]]
            eb, ei, et, y, tmax = to_events(items, a.bands, a.shift, rng, a.drop)
            p, info = net(eb, ei, et, len(items), tmax + 2 * int(a.dmax) + 60)
            main_loss = -torch.log(p[torch.arange(len(y)), y] + 1e-8).mean()
            aux_losses = [-torch.log(tap[torch.arange(len(y)), y] + 1e-8).mean()
                          for tap in info["tap_probs"][:-1]]
            aux_loss = sum(aux_losses) if aux_losses else main_loss.new_zeros(())
            loss = main_loss + a.aux_weight * aux_loss
            opt.zero_grad(); loss.backward()
            gsum += np.asarray([grad_norm(l) for l in net.layers])
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += float(main_loss.detach()); aux_tl += float(aux_loss.detach())

        net.eval(); ok = 0; tap_ok = np.zeros(a.depth)
        st = {"msgs": np.zeros(a.depth * 2), "spikes": np.zeros(a.depth)}
        for layer in net.layers:
            layer.sent.zero_()
        net.ro.sent.zero_()
        with torch.no_grad():
            for i0 in range(0, len(ev), a.bs):
                items = ev[i0:i0 + a.bs]
                eb, ei, et, y, tmax = to_events(items, a.bands, 0, rng, 0.0)
                p, info = net(eb, ei, et, len(items), tmax + 2 * int(a.dmax) + 60)
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
               "msgs_per_utt": (st["msgs"] / len(ev)).round(0).tolist(),
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
