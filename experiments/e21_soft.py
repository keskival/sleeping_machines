"""E21: training race networks through the entropic (Fermi–Dirac) k-winner race (THEORY §48).

    python experiments/e21_soft.py --task mnist --depth 2 [--sigma0 0.1 --sigma1 0.003] [--winners 3]

Hard k-of-G cancellation gives cancelled nodes no gradient (a ~2-point cost under exact gradients, E20). Its
dequantization (§21, §25) is the entropic top-k: memberships m_n = 1/(1 + e^{(T_n − μ)/σ}) with the chemical
potential μ set so that Σ m = k per group, the Fermi–Dirac distribution. Training runs the soft race (every
member spikes at its crossing time with amplitude m_n, so every member gets a gradient) while σ is annealed
from sigma0 to sigma1; evaluation runs the hard race (σ → 0), the actual event network.

A node's crossing with graded inputs: T_j = (θ_j + Σ_{i∈S} a_i w_ji t_i) / Σ_{i∈S} a_i w_ji over the causal set S.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, ramp_crossing, to_events  # noqa: E402
from e20_exact import data  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e21")


def crossing(tin, amp, W, theta):
    """Ramp crossing times with graded input amplitudes. tin, amp: (b, d); W: (n, d)."""
    t, idx = to_events(tin)
    d = W.shape[1]
    Wp = np.concatenate([W, np.zeros((W.shape[0], 1), np.float32)], 1)
    ampp = np.concatenate([amp, np.zeros((len(amp), 1), np.float32)], 1)
    a_ev = np.take_along_axis(ampp, np.minimum(idx, d), 1)                       # (b, s)
    w = np.transpose(Wp[:, idx], (1, 0, 2)) * a_ev[:, None, :]                  # (b, n, s)
    A = np.cumsum(w, 2)
    Bc = np.cumsum(w * np.where(np.isfinite(t), t, 0)[:, None, :], 2)
    T, _ = ramp_crossing(t, A, Bc, theta)
    return T


def fermi_dirac(T, group, k, sigma):
    """Soft k-of-G memberships and f = m(1 − m) per node; groups with ≤ k finite members keep m = 1."""
    b, n = T.shape
    g = T.reshape(b, -1, group)
    fin = np.isfinite(g)
    nfin = fin.sum(2, keepdims=True)
    gf = np.where(fin, g, 0.0)
    lo = np.where(fin, g, np.inf).min(2, keepdims=True) - 20 * sigma
    hi = np.where(fin, g, -np.inf).max(2, keepdims=True) + 20 * sigma
    lo, hi = np.where(np.isfinite(lo), lo, 0), np.where(np.isfinite(hi), hi, 1)
    for _ in range(40):                                                         # bisection on the chemical potential
        mu = 0.5 * (lo + hi)
        m = np.where(fin, 1 / (1 + np.exp(np.clip((gf - mu) / sigma, -50, 50))), 0.0)
        too_many = m.sum(2, keepdims=True) > k
        hi, lo = np.where(too_many, mu, hi), np.where(too_many, lo, mu)
    m = np.where(nfin <= k, fin.astype(float), m)
    f = np.where(nfin <= k, 0.0, m * (1 - m))
    return m.reshape(b, n).astype(np.float32), f.reshape(b, n).astype(np.float32)


def hard_topk(T, group, k):
    b, n = T.shape
    g = T.reshape(b, -1, group)
    order = np.argsort(g, 2)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(group)[None, None, :].repeat(b, 0).repeat(g.shape[1], 1), 2)
    return ((rank < k) & np.isfinite(g)).reshape(b, n).astype(np.float32)


class SoftRaceNet:
    def __init__(self, d_in, widths, k_out, drive_mean, rng, group=10, winners=3, tau=0.1):
        self.group, self.k, self.tau = group, winners, tau
        dims = [d_in] + list(widths)
        self.W, expected = [], drive_mean
        for l, h in enumerate(widths):
            mu = 1.0 / (0.6 * expected)
            self.W.append(rng.normal(mu, mu, (h, dims[l])).astype(np.float32))
            expected = h // group * winners * 0.5
        mu2 = 1.0 / (0.3 * expected)
        self.Wo = rng.normal(mu2, mu2, (k_out, dims[-1])).astype(np.float32)
        self.params = self.W + [self.Wo]
        self.scale = [float(np.abs(p).mean()) for p in self.params]
        self.m_ = [np.zeros_like(p) for p in self.params]
        self.v_ = [np.zeros_like(p) for p in self.params]
        self.th = [np.ones(h, np.float32) for h in widths]
        self.step, self.homeo, self.center, self.gnorm = 0, 0.1, True, [None] * len(self.params)

    def forward(self, tin, sigma):
        """sigma > 0: soft (Fermi–Dirac) race; sigma == 0: hard race (inference)."""
        acts, x, amp = [], tin, np.isfinite(tin).astype(np.float32)
        for l, W in enumerate(self.W):
            T = crossing(x, amp, W, self.th[l])
            if sigma > 0:
                m, f = fermi_dirac(T, self.group, self.k, sigma)
            else:
                m, f = hard_topk(T, self.group, self.k), np.zeros(T.shape, np.float32)
            acts.append(dict(tin=x, amp=amp, T=T, m=m, f=f))
            x = np.where(m > 1e-6, T, np.inf).astype(np.float32)
            amp = np.where(m > 1e-6, m, 0).astype(np.float32)
        To = crossing(x, amp, self.Wo, np.ones(self.Wo.shape[0], np.float32))
        fin = np.isfinite(x)
        A_all = (self.Wo[None] * (amp * fin)[:, None, :]).sum(2)
        B_all = (self.Wo[None] * (amp * np.where(fin, x, 0))[:, None, :]).sum(2)
        with np.errstate(divide="ignore", invalid="ignore"):
            T_ext = np.where(A_all > 0, (1.0 + B_all) / A_all, HORIZON + 1.0)
        To = np.where(np.isfinite(To), To, np.maximum(T_ext, HORIZON))
        return acts, x, amp, To

    @staticmethod
    def _grads(tin, amp, T, W, gT):
        """dL/dW, dL/dt_in, dL/da_in for T = (θ + Σ a w t)/Σ a w over the causal set."""
        finT = np.isfinite(T)
        Tf = np.where(finT, T, 0.0)
        fin_in = np.isfinite(tin)
        C = (fin_in[:, None, :] & (tin[:, None, :] <= Tf[:, :, None]) & finT[:, :, None]).astype(np.float32)
        aw = W[None] * amp[:, None, :] * C
        A = aw.sum(2)
        A = np.where(np.abs(A) > 1e-6, A, np.inf)
        g_over_A = (gT / A)[:, :, None]
        dt = (np.where(fin_in, tin, 0.0)[:, None, :] - Tf[:, :, None]) * C              # (t_i − T_j) on S
        gW = (g_over_A * amp[:, None, :] * dt).sum(0)
        gt = (g_over_A * aw).sum(1)
        ga = (g_over_A * W[None] * dt).sum(1)
        return gW.astype(np.float32), gt.astype(np.float32), ga.astype(np.float32)

    def train_step(self, tin, y, lr, sigma, clip=3.0):
        acts, x, amp, To = self.forward(tin, sigma)
        z = -To / self.tau
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        gz = p.copy()
        gz[np.arange(len(y)), y] -= 1
        gT = -gz / self.tau / len(y)
        grads = [None] * len(self.params)
        grads[-1], gt, ga = self._grads(x, amp, To, self.Wo, gT)
        for l in reversed(range(len(self.W))):
            L = acts[l]
            # a node's time matters directly (gt) and through every group member's membership (ga, Fermi–Dirac)
            b, n = L["T"].shape
            f = L["f"].reshape(b, -1, self.group)
            gam = ga.reshape(b, -1, self.group)
            sf = np.maximum(f.sum(2, keepdims=True), 1e-12)
            g_mem = (f / max(sigma, 1e-12)) * (-gam + (gam * f).sum(2, keepdims=True) / sf)
            gTl = np.where(np.isfinite(L["T"]), gt + g_mem.reshape(b, n), 0.0)
            if self.center:
                on = gTl != 0
                gTl = np.where(on, gTl - gTl.sum(1, keepdims=True) / np.maximum(on.sum(1, keepdims=True), 1), 0.0)
            grads[l], gt, ga = self._grads(L["tin"], L["amp"], L["T"], self.W[l], gTl)
        for l, L in enumerate(acts):
            self.th[l] += self.homeo * ((L["m"] > 0.5).mean(0) - self.k / self.group)
            np.maximum(self.th[l], 0.05, out=self.th[l])
        self.step += 1
        for j, g in enumerate(grads):
            nrm = float(np.sqrt((g * g).sum()))
            self.gnorm[j] = nrm if self.gnorm[j] is None else 0.99 * self.gnorm[j] + 0.01 * min(nrm, clip * self.gnorm[j])
            if nrm > clip * self.gnorm[j]:
                g = g * (clip * self.gnorm[j] / nrm)
            self.m_[j] = 0.9 * self.m_[j] + 0.1 * g
            self.v_[j] = 0.999 * self.v_[j] + 0.001 * g * g
            self.params[j] -= lr * self.scale[j] * (self.m_[j] / (1 - 0.9 ** self.step)) / (
                np.sqrt(self.v_[j] / (1 - 0.999 ** self.step)) + 1e-8)
        return loss


def main(a):
    rng = np.random.default_rng(a.seed)
    Ttr, ytr, Tte, yte, k = data(a)
    drive = float(np.where(np.isfinite(Ttr), HORIZON - Ttr, 0).sum(1).mean())
    net = SoftRaceNet(Ttr.shape[1], [a.width] * a.depth, k, drive, rng, winners=a.winners, tau=a.tau)

    def acc(T, y, sigma):
        return float(np.mean(np.concatenate([net.forward(T[i:i + 500], sigma)[3].argmin(1)
                                             for i in range(0, len(y), 500)]) == y))

    curve, t0 = [], time.time()
    steps = (len(ytr) + a.batch - 1) // a.batch
    for ep in range(a.epochs):
        perm = rng.permutation(len(ytr))
        losses = []
        for si, i in enumerate(range(0, len(perm), a.batch)):
            frac = (ep * steps + si) / max(a.epochs * steps, 1)
            sigma = a.sigma0 * (a.sigma1 / a.sigma0) ** frac                 # geometric annealing
            lr = a.lr * (1 - 0.9 * frac)
            losses.append(net.train_step(Ttr[perm[i:i + a.batch]], ytr[perm[i:i + a.batch]], lr, sigma))
        hard, soft = acc(Tte, yte, 0.0), acc(Tte, yte, sigma)
        curve.append(hard)
        print(f"{a.task} depth {a.depth} k {a.winners} epoch {ep + 1}: loss {np.mean(losses):.4f} "
              f"val hard {hard:.4f} soft {soft:.4f} sigma {sigma:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res = {"config": vars(a), "curve": curve, "acc": curve[-1]}
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.task}_d{a.depth}_w{a.width}_k{a.winners}_s0{a.sigma0:g}_s1{a.sigma1:g}_ep{a.epochs}_P{a.train_limit or 'all'}_s{a.seed}"
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)


def gradcheck():
    rng = np.random.default_rng(0)
    d, b, sigma = 30, 16, 0.05
    tin = np.where(rng.random((b, d)) < 0.5, rng.random((b, d)) * 0.8, np.inf).astype(np.float32)
    y = rng.integers(0, 4, b)
    drive = float(np.where(np.isfinite(tin), 1 - tin, 0).sum(1).mean())
    net = SoftRaceNet(d, [20, 20], 4, drive, rng)
    net.homeo, net.center = 0.0, False

    def loss():
        To = net.forward(tin, sigma)[3]
        z = -To / net.tau
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(b), y] - np.log(np.exp(z).sum(1))).mean())
    saved = [p_.copy() for p_ in net.params]
    net.scale = [0.0] * len(net.params)                      # capture gradients without moving the weights
    cap = []
    orig = net._grads

    def spy(*args):
        out = orig(*args)
        cap.append(out[0])
        return out
    net._grads = spy
    net.train_step(tin, y, 0.0, sigma, clip=1e9)
    net._grads = orig
    grads = list(reversed(cap))                             # captured output layer first
    for p_, s_ in zip(net.params, saved):
        p_[...] = s_
    for name, P, G in (("W0", net.W[0], grads[0]), ("W1", net.W[1], grads[1]), ("Wo", net.Wo, grads[2])):
        errs, r2 = [], np.random.default_rng(1)
        for _ in range(60):
            i, j = r2.integers(0, P.shape[0]), r2.integers(0, P.shape[1])
            if abs(G[i, j]) < 1e-7:
                continue
            h = 1e-2 * max(abs(P[i, j]), 1e-2)
            old = P[i, j]
            P[i, j] = old + h; lp = loss()
            P[i, j] = old - h; lm = loss()
            P[i, j] = old
            fd = (lp - lm) / (2 * h)
            errs.append(abs(fd - G[i, j]) / max(abs(fd), abs(G[i, j]), 1e-8))
        print(f"{name}: {len(errs)} coords, median rel err {np.median(errs):.2e}, within 5%: "
              f"{np.mean(np.array(errs) < 0.05):.2f}", flush=True)


if __name__ == "__main__":
    if "--gradcheck" in sys.argv:
        gradcheck()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("mnist", "rhm"), default="mnist")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--winners", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--sigma1", type=float, default=0.003)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
