"""E82: why do the time-vector networks learn slowly on SHD? Learning curves of E74 variants on 512 training utterances
(held-out speakers: 256 validation utterances), 240 steps each, all else equal:
  norm        E74 as queued: class units read normalized by their count channel (§105), spike-time sensitivity floor 0.02
  norm_v02    the same with the floor at 0.2 (bounded 1/V' for grazing spikes)
  ln          non-spiking state units + LayerNorm + linear head per time (the E77 / E80 readout)
  ln_lr1e-2   the same at a higher learning rate
  readonly    no spiking layers: input events straight into the readout (what the spiking layers add)
Reported every 40 steps: training loss, held-out accuracy, spikes per utterance.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import e51_shd_world as S  # noqa: E402
import e74_time_vector_net as A  # noqa: E402
from e71_event_cde import events  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e82")


class Variant(nn.Module):
    def __init__(self, mode, M=64):
        super().__init__()
        self.mode = mode
        self.net = A.Net(140, 16, 8, M, M, 30, 0.25, 50.0, [0.05, 0.05])
        if mode == "norm_v02":
            for l in (self.net.l1, self.net.l2):
                l.vdot_min = 0.2
        if mode.startswith("ln"):
            self.st = A.TVLayer(M, M, 16, 16, 8, 50.0, None, False, 0.3, gate_bias=1.0)
            self.ln = nn.LayerNorm(M); self.head = nn.Linear(M, 20)
        if mode == "readonly":
            self.ro = A.TVLayer(140, 20, 16, 16, 8, 50.0, None, False, 0.3, gate_bias=1.0, normalize=True)

    def forward(self, eb, ei, et, B, G):
        n = self.net; ev = n.emb(ei)
        if self.mode == "readonly":
            V, _ = self.ro(eb, ei, et, ev, B, G); return torch.log(torch.softmax(V[::4], -1).mean(0) + 1e-8), [0, 0]
        (b1, j1, t1, y1), _ = n.l1(eb, ei, et, ev, B, G); (b2, j2, t2, y2), _ = n.l2(b1, j1, t1, y1, B, G)
        if self.mode.startswith("ln"):
            V, _ = self.st(b2, j2, t2, y2, B, G); logits = self.head(self.ln(V[::4]))
            return torch.log(torch.softmax(logits, -1).mean(0) + 1e-8), [len(t1) / B, len(t2) / B]
        V, _ = n.ro(b2, j2, t2, y2, B, G)
        return torch.log(torch.softmax(V[::4], -1).mean(0) + 1e-8), [len(t1) / B, len(t2) / B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--modes", default="norm,norm_v02,ln,ln_lr1e-2,readonly")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tr = [(*events(t, u, 0.002), y) for t, u, y in itertools.islice(S.utterances("train", 140, "fit_spk"), 0, 5120, 10)]
    ev = [(*events(t, u, 0.002), y) for t, u, y in itertools.islice(S.utterances("train", 140, "val_spk"), 0, 1024, 4)]
    res = {"args": vars(a), "train": len(tr), "eval": len(ev)}
    for mode in a.modes.split(","):
        torch.manual_seed(0); rng = np.random.default_rng(0); t0 = time.time()
        m = Variant(mode.replace("_lr1e-2", "") if mode == "ln_lr1e-2" else mode)
        m.mode = "ln" if mode == "ln_lr1e-2" else mode
        opt = torch.optim.Adam(m.parameters(), lr=1e-2 if mode == "ln_lr1e-2" else 3e-3); curve = []
        for step in range(1, a.steps + 1):
            items = [tr[j] for j in rng.choice(len(tr), 32, replace=False)]
            eb, ei, et, y, tmax = A.to_events(items, 140, 4, rng, 0.1)
            logp, sp = m(eb, ei, et, len(items), tmax + 160)
            loss = -logp[torch.arange(len(y)), y].mean(); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if step % 40 == 0:
                ok = 0
                with torch.no_grad():
                    for i0 in range(0, len(ev), 32):
                        items = ev[i0:i0 + 32]; eb, ei, et, yv, tmax = A.to_events(items, 140, 0, rng, 0.0)
                        lp, _ = m(eb, ei, et, len(items), tmax + 160); ok += int((lp.argmax(1) == yv).sum())
                row = {"step": step, "train_loss": round(float(loss), 3), "heldout_acc": round(ok / len(ev), 3),
                       "spikes": [round(s) for s in sp], "wall_s": round(time.time() - t0)}
                curve.append(row); print(json.dumps({"mode": mode, **row}), flush=True)
        res[mode] = curve
    json.dump(res, open(os.path.join(OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
