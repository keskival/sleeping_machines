"""E77: a time-vector character language model (THEORY §105-§107) on text8.

Characters are events at half-integer times (character j at j + 0.5, so the state read at integer time k + 1 has seen
characters 0..k and nothing later). Architecture:
  layer 1   spiking time-vector units (§105): content-gated, content-delayed messages from the characters; threshold
            firing (never earlier than the detecting step: causal); snapshot payloads
  layer 2   spiking time-vector units fed by a random quarter of layer 1
  state     non-spiking time-vector units read at every character boundary: h_k (the recurrent memory of §107(a))
  retrieval delay-coded attention (§107(c)): query from (h_k, x_k); keys and values from earlier positions j < k, values
            carry the character that followed (x_{j+1}); exact softmax over the keys within Delta of the best (the race opens
            a window of length kappa * Delta); simulated as that masked softmax, which the lemma makes identical; work = keys
            inside the window
  readout   softmax over 27 characters from (h_k, e(x_k), retrieval output)
Trained by gradients through spike times, payloads and delays (as E74); early stopping on 200k validation characters;
test on the same 1M characters as E62-E66 and E64 (windows of L with L/2 context). Reports bits per character, and messages,
spikes and retrieved keys per character.
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
import e62_charlm as S1  # noqa: E402
from e74_time_vector_net import TVLayer  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e77")
A = 27


class TVLM(nn.Module):
    def __init__(self, d, n, M1, M2, Mr, fan2, dmax, w_sd, retrieval, heads, dh, delta, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(A, d)
        kw = dict(causal=True, tau_range=(1.0, 200.0), td0=1.0)
        self.l1 = TVLayer(A, M1, d, d, n, dmax, None, True, w_sd[0], **kw)
        self.l2 = TVLayer(M1, M2, d, d, n, dmax, torch.rand(M1, M2, generator=gen) < fan2, True, w_sd[1], **kw)
        self.st = TVLayer(M2, Mr, d, d, n, dmax, None, False, 0.3, gate_bias=1.0, tau_range=(1.0, 200.0), td0=1.0)
        self.norm = nn.LayerNorm(Mr)
        self.retrieval, self.heads, self.dh, self.delta = retrieval, heads, dh, delta
        f = Mr + d
        if retrieval:
            self.Wq = nn.Linear(f, heads * dh); self.Wk = nn.Linear(f, heads * dh); self.Wv = nn.Linear(f, heads * dh)
            self.Ev = nn.Embedding(A, heads * dh); self.null = nn.Parameter(torch.zeros(heads))
        self.head = nn.Sequential(nn.Linear(f + (heads * dh if retrieval else 0), 256), nn.GELU(), nn.Linear(256, A))

    def forward(self, x):
        """x: (B, L) characters -> logits (B, L, A) for x[:, 1:] (position k predicts character k + 1), and work."""
        B, L = x.shape; G = L + 1
        eb = torch.arange(B).repeat_interleave(L); ei = x.reshape(-1); et = torch.arange(L).float().repeat(B) + 0.5
        ev = self.emb(ei)
        (b1, j1, t1, y1), m1 = self.l1(eb, ei, et, ev, B, G)
        (b2, j2, t2, y2), m2 = self.l2(b1, j1, t1, y1, B, G)
        V, m3 = self.st(b2, j2, t2, y2, B, G)                                   # (G, B, Mr)
        h = self.norm(V[1:L + 1].permute(1, 0, 2))                             # state after characters 0..k
        f = torch.cat([h, self.emb(x)], -1)
        work = {"msgs_per_char": [m1 / L, m2 / L, m3 / L], "spikes_per_char": [len(t1) / (B * L), len(t2) / (B * L)]}
        if self.retrieval:
            H, dh = self.heads, self.dh
            q = self.Wq(f).view(B, L, H, dh).transpose(1, 2); k = self.Wk(f).view(B, L, H, dh).transpose(1, 2)
            nxt = torch.cat([x[:, 1:], x[:, -1:]], 1)                          # value j carries x_{j+1}
            v = (self.Wv(f) + self.Ev(nxt)).view(B, L, H, dh).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(dh)                         # (B, H, L, L)
            allowed = torch.tril(torch.ones(L, L, dtype=torch.bool), -1)       # j < k: x_{j+1} is known at k
            s = s.masked_fill(~allowed, float("-inf"))
            s0 = self.null.view(1, H, 1, 1).expand(B, H, L, 1)                 # a null key (reply "nothing found")
            s = torch.cat([s, s0], -1)
            top = s.max(-1, keepdim=True).values.detach()
            inwin = s >= top - self.delta                                      # keys that reply inside the race window
            p = torch.softmax(s.masked_fill(~inwin, float("-inf")), -1)
            o = (p[..., :L] @ v).transpose(1, 2).reshape(B, L, H * dh)
            f = torch.cat([f, o], -1)
            work["keys_per_query"] = float(inwin[..., :L].sum(-1).float().mean())
            work["keys_visible"] = float(allowed.sum(-1).float().mean())
        return self.head(f), work


def score(net, data, L, bs, dump=False):
    """bits per character: windows of L, each scoring its second half (first window scores all), as E64. dump: also the
    probability of each true character, indexed by its position in `data` (for mixing with the native experts, E78)."""
    net.eval(); tot = 0.0; n = 0; half = L // 2; starts = list(range(0, len(data) - L - 1, half)); wk = []
    ptrue = np.full(len(data), np.nan, np.float32) if dump else None
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            ss = starts[i:i + bs]
            X = torch.stack([data[s:s + L] for s in ss]); Y = torch.stack([data[s + 1:s + L + 1] for s in ss])
            logits, w = net(X); wk.append(w)
            ce = nn.functional.cross_entropy(logits.reshape(-1, A), Y.reshape(-1), reduction="none").view(len(ss), L)
            for r, s0 in enumerate(ss):
                lo = 0 if s0 == 0 else half
                tot += float(ce[r, lo:].sum()); n += L - lo
                if dump:
                    ptrue[s0 + 1 + lo:s0 + 1 + L] = torch.exp(-ce[r, lo:]).numpy()
    net.train()
    agg = {k: (np.mean([w[k] for w in wk], 0).round(3).tolist() if isinstance(wk[0][k], list) else round(float(np.mean([w[k] for w in wk])), 2))
           for k in wk[0]}
    return (tot / n / math.log(2), agg, ptrue) if dump else (tot / n / math.log(2), agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1_000_000)
    ap.add_argument("--passes", type=float, default=5.0)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M1", type=int, default=128)
    ap.add_argument("--M2", type=int, default=128)
    ap.add_argument("--Mr", type=int, default=64)
    ap.add_argument("--fan2", type=float, default=0.25)
    ap.add_argument("--dmax", type=float, default=8.0)
    ap.add_argument("--w_sd", default="0.05,0.05")
    ap.add_argument("--retrieval", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--delta", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--valid", type=int, default=200_000)
    ap.add_argument("--test", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    x = S1.load(); train = torch.tensor(x[:a.D]).long()
    valid = torch.tensor(x[90_000_000:90_000_000 + a.valid]).long(); test = torch.tensor(x[95_000_000:95_000_000 + a.test]).long()
    net = TVLM(a.d, a.n, a.M1, a.M2, a.Mr, a.fan2, a.dmax, [float(v) for v in a.w_sd.split(",")], a.retrieval, a.heads, a.dh,
               a.delta, a.seed)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps = int(a.passes * a.D / (a.bs * a.L)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    res = {"args": vars(a), "params": sum(p.numel() for p in net.parameters()), "steps": steps, "valid_curve": []}
    print(json.dumps({"params": res["params"], "steps": steps}), flush=True)
    best = (float("inf"), None, -1)
    for step in range(steps):
        idx = rng.integers(0, a.D - a.L - 1, a.bs)
        X = torch.stack([train[i:i + a.L] for i in idx]); Y = torch.stack([train[i + 1:i + a.L + 1] for i in idx])
        logits, _ = net(X)
        loss = nn.functional.cross_entropy(logits.reshape(-1, A), Y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % max(steps // 10, 1) == 0 or step == steps - 1:
            vb, wk = score(net, valid, a.L, a.bs)
            row = {"step": step + 1, "train_bpc": float(loss) / math.log(2), "valid_bpc": vb, **wk, "wall_s": round(time.time() - t0)}
            res["valid_curve"].append(row); print(json.dumps(row), flush=True)
            if vb < best[0]:
                best = (vb, {k: v.clone() for k, v in net.state_dict().items()}, step + 1)
    net.load_state_dict(best[1])
    tb, wk, pt = score(net, test, a.L, a.bs, dump=True)
    _, _, pv = score(net, valid, a.L, a.bs, dump=True)
    tag = f"tvlm_D{a.D}_p{a.passes:g}_r{a.retrieval}_M{a.M1}-{a.M2}-{a.Mr}_s{a.seed}"
    np.save(os.path.join(OUT, tag + "_ptrue_test.npy"), pt); np.save(os.path.join(OUT, tag + "_ptrue_valid.npy"), pv)
    res.update({"best_step": best[2], "best_valid_bpc": best[0], "test_bpc": tb, "test_work": wk, "wall_s": round(time.time() - t0)})
    print(json.dumps({"test_bpc": tb, "best_step": best[2], **wk}), flush=True)
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
