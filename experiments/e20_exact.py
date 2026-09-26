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
        self.scale = [float(np.abs(W).mean()) for W in self.params]   # Adam steps relative to each layer's scale
        self.nonneg = False
        self.th = [np.ones(h, np.float32) for h in widths]           # §36 prices: per-node thresholds
        self.homeo, self.center = 0.0, False
        self.tnorm, self.training, self.tstat = False, False, [None] * len(widths)
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0

    @staticmethod
    def _cross(tin, W, theta=None):
        """Crossing times of a ramp layer on dense input times tin (b, d); thresholds θ (default 1)."""
        t, idx = to_events(tin)
        Wp = np.concatenate([W, np.zeros((W.shape[0], 1), np.float32)], 1)   # padding column
        th = np.ones(W.shape[0], np.float32) if theta is None else theta
        T, _, _ = layer_race(t, idx, Wp, th, "ramp")
        return T

    def forward(self, tin):
        acts = []
        x = tin
        for W in self.W:
            T = self._cross(x, W, self.th[len(acts)])
            b, n = T.shape
            fired, _ = group_race(T, np.where(np.isfinite(T), 0.0, -np.inf), n // self.group, self.winners)
            fired = fired & np.isfinite(T)
            g = np.where(np.isfinite(T), T, np.inf).reshape(b, -1, self.group)
            kth = np.sort(g, 2)[:, :, self.winners - 1:self.winners]
            kth = np.repeat(np.where(np.isfinite(kth), kth, HORIZON), self.group, 2).reshape(b, n)
            presence = np.where(fired, 1.0, np.where(np.isfinite(T), np.exp(-np.clip(T - kth, 0, None) / self.sigma_b), 0.0))
            a_l = 1.0
            Tn = T
            if self.tnorm:
                # §49 temporal normalisation: an affine re-timing of the layer's output (a gauge choice, since race
                # neurons are equivariant under shift and dilation of time), from running (not batch) statistics
                fin = T[fired & np.isfinite(T)]
                l_ = len(acts)
                if self.training and fin.size:
                    m_, s_ = float(fin.mean()), float(fin.std() + 1e-4)
                    if self.tstat[l_] is None:
                        self.tstat[l_] = [m_, s_]
                    else:
                        self.tstat[l_][0] += 0.01 * (m_ - self.tstat[l_][0])
                        self.tstat[l_][1] += 0.01 * (s_ - self.tstat[l_][1])
                if self.tstat[l_] is not None:
                    a_l = 0.15 / self.tstat[l_][1]
                    Tn = np.clip(0.3 + a_l * (T - self.tstat[l_][0]), 0.0, 0.95)
            acts.append(dict(tin=x, T=T, Tn=Tn, a=a_l, fired=fired, presence=presence.astype(np.float32)))
            x = np.where(fired, Tn, np.inf).astype(np.float32)
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

    def train_step(self, tin, y, lr, clip=0.0):
        self.training = True
        acts, x, To = self.forward(tin)
        self.training = False
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
            soft_t = np.where(np.isfinite(last["T"]) & (last["presence"] > 0), last["Tn"], np.inf).astype(np.float32)
            soft = (soft_t, last["presence"])
        else:
            soft = None
        grads[-1], gt = self._node_grads(x, To, self.Wo, gT, soft)
        for l in reversed(range(len(self.W))):
            L = acts[l]
            gTl = np.where(np.isfinite(L["T"]), gt * L["a"], 0.0)          # through the affine re-timing
            if self.center:
                # §30.1 Ward identity: the layer's summed timing credit is the clock's (activity), owned by the
                # thresholds; the weights get only the zero-sum part, over the nodes that carry credit
                on = gTl != 0
                gTl = np.where(on, gTl - (gTl.sum(1, keepdims=True) / np.maximum(on.sum(1, keepdims=True), 1)), 0.0)
            if l > 0:
                P = acts[l - 1]
                soft_t = np.where(np.isfinite(P["T"]) & (P["presence"] > 0), P["Tn"], np.inf).astype(np.float32)
                soft = (soft_t, P["presence"])
            else:
                soft = None
            grads[l], gt = self._node_grads(L["tin"], L["T"], self.W[l], gTl, soft)
        if self.homeo:
            for l, L in enumerate(acts):                            # thresholds track a target firing rate
                self.th[l] += self.homeo * (L["fired"].mean(0) - self.winners / self.group)
                np.maximum(self.th[l], 0.05, out=self.th[l])
        self.step += 1
        if clip:                                                    # per-layer gradient-norm clipping
            if not hasattr(self, "gnorm"):
                self.gnorm = [None] * len(grads)
            for j, g in enumerate(grads):
                nrm = float(np.sqrt((g * g).sum()))
                self.gnorm[j] = nrm if self.gnorm[j] is None else 0.99 * self.gnorm[j] + 0.01 * min(nrm, clip * self.gnorm[j])
                if nrm > clip * self.gnorm[j]:
                    grads[j] = g * (clip * self.gnorm[j] / nrm)
        for j, (prm, g) in enumerate(zip(self.params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            prm -= lr * self.scale[j] * (self.m[j] / (1 - 0.9 ** self.step)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.step)) + 1e-8)
            if self.nonneg:
                np.maximum(prm, 0, out=prm)
        return loss


def data(a):
    if a.task == "shd":                                       # same shuffle and validation split as e14 --val 800
        z = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "shd", "shd_700.npz"))
        perm0 = np.random.default_rng(12345).permutation(len(z["ytr"]))
        X, y = z["Xtr"][perm0], z["ytr"][perm0]
        return X[:-800], y[:-800], X[-800:], y[-800:], 20
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
    if a.model == "mlp":                                  # backprop reference on the same inputs (intensities)
        from e19_rhm import run_mlp
        enc = lambda T: np.where(np.isfinite(T), HORIZON - T, 0).astype(np.float32)  # noqa: E731
        a.epochs_mlp = a.epochs
        res = run_mlp(a, enc(Ttr), ytr, enc(Tte), yte, k)
        res.update(config=vars(a), acc=res["test_acc"])
        os.makedirs(OUT, exist_ok=True)
        name = f"mlp_{a.task}_d{a.depth}_w{a.width}_P{a.train_limit or 'all'}_s{a.seed}"
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        print(name, res["acc"], flush=True)
        return
    drive = float(np.where(np.isfinite(Ttr), HORIZON - Ttr, 0).sum(1).mean())
    net = ExactRaceNet(Ttr.shape[1], [a.width] * a.depth, k, drive, rng, tau=a.tau, winners=a.winners)
    net.nonneg = bool(a.nonneg)
    net.homeo, net.center = a.homeo, bool(a.center)
    net.tnorm = bool(a.tnorm)

    def acc(T, y):
        return float(np.mean(np.concatenate([net.predict(T[i:i + 500]) for i in range(0, len(y), 500)]) == y))

    curve, t0 = [], time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        steps = (len(ytr) + a.batch - 1) // a.batch
        losses = []
        for si, i in enumerate(range(0, len(perm), a.batch)):
            frac = (ep * steps + si) / max(a.epochs * steps, 1)
            lr = a.lr * (1 - 0.9 * frac) if a.decay else a.lr     # linear decay to 10%
            losses.append(net.train_step(Ttr[perm[i:i + a.batch]], ytr[perm[i:i + a.batch]], lr, a.clip))
        curve.append(acc(Tte, yte))
        print(f"{a.task} depth {a.depth} epoch {ep + 1}: loss {np.mean(losses):.4f} val {curve[-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = {"config": vars(a), "curve": curve, "acc": curve[-1], "train_acc_5k": acc(Ttr[:5000], ytr[:5000])}
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.task}_d{a.depth}_w{a.width}_lr{a.lr:g}_tau{a.tau:g}_nn{a.nonneg}_ho{a.homeo:g}_c{a.center}_dc{a.decay}_cl{a.clip:g}_k{a.winners}_tn{a.tnorm}_P{a.train_limit or 'all'}_s{a.seed}"
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("mnist", "rhm", "shd"), default="mnist")
    ap.add_argument("--model", choices=("race", "mlp"), default="race")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--nonneg", type=int, default=0, help="excitatory weights only (no near-zero A)")
    ap.add_argument("--winners", type=int, default=3, help="winners per group of 10 (10 = no cancellation)")
    ap.add_argument("--decay", type=int, default=0, help="linear learning-rate decay to 10%")
    ap.add_argument("--clip", type=float, default=0.0, help="clip a layer's gradient norm at this multiple of its running norm")
    ap.add_argument("--tnorm", type=int, default=0, help="§49 temporal normalisation (running affine re-timing)")
    ap.add_argument("--homeo", type=float, default=0.0, help="§36: threshold (price) step towards the target rate")
    ap.add_argument("--center", type=int, default=0, help="§30.1: zero-sum timing credit per layer")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
