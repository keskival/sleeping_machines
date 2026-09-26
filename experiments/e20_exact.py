"""E20: the architecture's ceiling. Race networks trained with exact spike-time gradients (Adam).

    python experiments/e20_exact.py --task mnist|rhm --depth D [--width 400] [--epochs 10] [--seed 0]

Question: is the gap to dense networks in the race *architecture* or in the local learning rules? Same forward
pass as E14 (ramp synapses, k-of-10 group races, one spike per node), but trained end to end:

- within the realised piece, a node that fired crosses at T = (θ + Σ_{i∈S} w_i t_i) / A_S over its causal set S
  (inputs that arrived before T), so ∂T/∂w_i = (t_i − T)/A and ∂T/∂t_i = w_i/A (THEORY §44, §30.3);
- loss: cross-entropy on output logits −T/τ; an output that does not cross by the horizon uses its extrapolated
  crossing from all its inputs, so every output has a gradient;
- cancelled nodes carry no spike forward; backward they count as present with weight exp(−(T − T_k)/σ_b),
  the near-miss boundary term (THEORY §4, §21.6);
- θ fixed at 1 (only θ/w matters, the scale gauge of §30.3); weights learnt with Adam.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, group_race, latency_code, layer_race, mnist, to_events  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e20")


class ExactRaceNet:
    def __init__(self, d_in, widths, k_out, drive_mean, rng, group=10, winners=3, tau=0.1, sigma_b=0.1):
        self.group, self.winners, self.tau, self.sigma_b = group, winners, tau, sigma_b
        dims = [d_in] + list(widths)
        self.W = []
        expected = drive_mean
        for l, h in enumerate(widths):
            mu = 1.0 / (0.6 * expected)                       # a node crosses after ~60% of its expected drive
            self.W.append(rng.normal(mu, mu, (h, dims[l])).astype(np.float32))
            expected = h // group * winners * 0.5
        mu2 = 1.0 / (0.3 * expected)
        self.Wo = rng.normal(mu2, mu2, (k_out, dims[-1])).astype(np.float32)
        self.params = self.W + [self.Wo]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0

    @staticmethod
    def _cross(tin, W):
        """Crossing times of a ramp layer on dense input times tin (b, d); θ = 1."""
        t, idx = to_events(tin)
        Wp = np.concatenate([W, np.zeros((W.shape[0], 1), np.float32)], 1)   # padding column
        T, _, _ = layer_race(t, idx, Wp, np.ones(W.shape[0], np.float32), "ramp")
        return T

    def forward(self, tin):
        acts = []
        x = tin
        for W in self.W:
            T = self._cross(x, W)
            b, n = T.shape
            fired, _ = group_race(T, np.where(np.isfinite(T), 0.0, -np.inf), n // self.group, self.winners)
            fired = fired & np.isfinite(T)
            g = np.where(np.isfinite(T), T, np.inf).reshape(b, -1, self.group)
            kth = np.sort(g, 2)[:, :, self.winners - 1:self.winners]
            kth = np.repeat(np.where(np.isfinite(kth), kth, HORIZON), self.group, 2).reshape(b, n)
            presence = np.where(fired, 1.0, np.where(np.isfinite(T), np.exp(-np.clip(T - kth, 0, None) / self.sigma_b), 0.0))
            acts.append(dict(tin=x, T=T, fired=fired, presence=presence.astype(np.float32)))
            x = np.where(fired, T, np.inf).astype(np.float32)
        To = self._cross(x, self.Wo)
        fin = np.isfinite(x)
        A_all = (self.Wo[None] * fin[:, None, :]).sum(2)
        B_all = (self.Wo[None] * np.where(fin, x, 0)[:, None, :]).sum(2)
        with np.errstate(divide="ignore", invalid="ignore"):
            T_ext = np.where(A_all > 0, (1.0 + B_all) / A_all, HORIZON + 1.0)
        To = np.where(np.isfinite(To), To, np.maximum(T_ext, HORIZON))
        return acts, x, To

    def predict(self, tin):
        return self.forward(tin)[2].argmin(1)

    def _node_grads(self, tin, T, W, gT, soft_in=None):
        """For nodes with finite T: exact dL/dW over the real causal set (inputs with t_i <= T), and dL/dt_in.
        The input gradient also reaches soft-present inputs (cancelled near misses at their projected times,
        weighted by presence): a backward-only boundary term that does not change the forward A."""
        finT = np.isfinite(T)
        Tf = np.where(finT, T, 0.0)
        fin_in = np.isfinite(tin)
        C = fin_in[:, None, :] & (tin[:, None, :] <= Tf[:, :, None]) & finT[:, :, None]
        A = (W[None] * C).sum(2)
        A = np.where(np.abs(A) > 1e-6, A, np.inf)
        g_over_A = (gT / A)[:, :, None]
        tc = np.where(fin_in, tin, 0.0)
        gW = (g_over_A * C * (tc[:, None, :] - Tf[:, :, None])).sum(0)
        if soft_in is None:
            gt = (g_over_A * C * W[None]).sum(1)
        else:
            t_eff, pres = soft_in
            Cs = np.isfinite(t_eff)[:, None, :] & (t_eff[:, None, :] <= Tf[:, :, None]) & finT[:, :, None]
            gt = (g_over_A * Cs * W[None]).sum(1) * pres
        return gW.astype(np.float32), gt.astype(np.float32)

    def train_step(self, tin, y, lr):
        acts, x, To = self.forward(tin)
        z = -To / self.tau
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        gz = p
        gz[np.arange(len(y)), y] -= 1
        gT = -gz / self.tau / len(y)
        # output layer: soft-extended inputs = fired spikes plus cancelled near misses at their projected times
        grads = [None] * (len(self.W) + 1)
        last = acts[-1] if acts else None
        if last is not None:
            soft_t = np.where(np.isfinite(last["T"]) & (last["presence"] > 0), last["T"], np.inf).astype(np.float32)
            soft = (soft_t, last["presence"])
        else:
            soft = None
        grads[-1], gt = self._node_grads(x, To, self.Wo, gT, soft)
        for l in reversed(range(len(self.W))):
            L = acts[l]
            gTl = np.where(np.isfinite(L["T"]), gt, 0.0)
            if l > 0:
                P = acts[l - 1]
                soft_t = np.where(np.isfinite(P["T"]) & (P["presence"] > 0), P["T"], np.inf).astype(np.float32)
                soft = (soft_t, P["presence"])
            else:
                soft = None
            grads[l], gt = self._node_grads(L["tin"], L["T"], self.W[l], gTl, soft)
        self.step += 1
        for j, (prm, g) in enumerate(zip(self.params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            prm -= lr * (self.m[j] / (1 - 0.9 ** self.step)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.step)) + 1e-8)
        return loss


def data(a):
    if a.task == "mnist":
        x, y = mnist("train")
        xtr, ytr, xte, yte = x[:-10000], y[:-10000], x[-10000:], y[-10000:]
        if a.train_limit:
            xtr, ytr = xtr[:a.train_limit], ytr[:a.train_limit]
        return latency_code(xtr).astype(np.float32), ytr, latency_code(xte).astype(np.float32), yte, 10
    from e19_rhm import make_rules, one_hot, sample
    rules = make_rules(8, 8, 4, 2, 3, np.random.default_rng(1000))
    rng = np.random.default_rng(a.seed)
    xtr, ytr = sample(rules, a.train_limit or 16000, 8, 4, rng)
    xte, yte = sample(rules, 10000, 8, 4, np.random.default_rng(10_000 + a.seed))
    enc = lambda x: np.where(one_hot(x, 8) > 0, 0.0, np.inf).astype(np.float32)  # noqa: E731
    return enc(xtr), ytr, enc(xte), yte, 8


def main(a):
    rng = np.random.default_rng(a.seed)
    Ttr, ytr, Tte, yte, k = data(a)
    drive = float(np.where(np.isfinite(Ttr), HORIZON - Ttr, 0).sum(1).mean())
    net = ExactRaceNet(Ttr.shape[1], [a.width] * a.depth, k, drive, rng, tau=a.tau)

    def acc(T, y):
        return float(np.mean(np.concatenate([net.predict(T[i:i + 500]) for i in range(0, len(y), 500)]) == y))

    curve, t0 = [], time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        losses = [net.train_step(Ttr[perm[i:i + a.batch]], ytr[perm[i:i + a.batch]], a.lr)
                  for i in range(0, len(perm), a.batch)]
        curve.append(acc(Tte, yte))
        print(f"{a.task} depth {a.depth} epoch {ep + 1}: loss {np.mean(losses):.4f} val {curve[-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = {"config": vars(a), "curve": curve, "acc": curve[-1], "train_acc_5k": acc(Ttr[:5000], ytr[:5000])}
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.task}_d{a.depth}_w{a.width}_lr{a.lr:g}_P{a.train_limit or 'all'}_s{a.seed}"
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("mnist", "rhm"), default="mnist")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
