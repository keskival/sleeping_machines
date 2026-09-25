#!/usr/bin/env python3
"""Numerical checks of THEORY.md on small networks.

M3 (first version): which learning rule points along the true gradient?

The true objective is the expected error of the race under timing noise
(input spike times jittered by σ), which is smooth although each race is not.
Its gradient is estimated by central finite differences with common random
numbers, on a random subset of weight coordinates. Each rule's update, computed
on the same samples without noise, is compared with it by cosine similarity on
those coordinates.

    python experiments/theory_checks.py m3 --coords 200
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, RaceNet, layer_race, latency_code, mnist, to_events  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "theory")


def small_mnist(n, offset=0):
    """14×14 MNIST (2×2 average pooling), latency-coded."""
    x, y = mnist("train")
    x = x[offset:offset + n].reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
    return latency_code(x), y[offset:offset + n]


def make_net(cfg, times, seed):
    drive = np.where(np.isfinite(times), 1.0 - times, 0).mean(0)
    return RaceNet(cfg, times.shape[1], 10, float(drive.sum()), np.random.default_rng(seed), drive)


def pretrain(net, times, y, epochs, batch=32):
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        for i in range(0, len(y), batch):
            ii = rng.permutation(len(y))[:batch]
            t, idx = to_events(times[ii])
            net.teach(net.forward(t, idx), y[ii])


OBJECTIVE = "error"


def error_under_noise(net, noisy_events, y_rep):
    """Expected 0/1 error, or (OBJECTIVE = "margin") a smoother race margin: the target's
    distance to threshold at the decision minus the closest rival's (lower is better)."""
    t, idx = noisy_events
    st = net.forward(t, idx)
    if OBJECTIVE == "error":
        return float((st["winner"] != y_rep).mean())
    snap = st["snap2"]
    rows = np.arange(len(y_rep))
    target = snap[rows, y_rep]
    rival = snap.copy()
    rival[rows, y_rep] = np.inf
    return float(np.clip(target - rival.min(1), -0.5, 0.5).mean())


def rule_update(net, variant, t, idx, y):
    """ΔW1 and ΔW2 a rule would make on this batch (no noise, no homeostasis)."""
    W1, W2, th1 = net.W1.copy(), net.W2.copy(), net.th1.copy()
    saved = net.cfg
    net.cfg = replace(saved, variant=variant, homeo=0.0)
    net.teach(net.forward(t, idx), y)
    d1, d2 = net.W1 - W1, net.W2 - W2
    net.W1, net.W2, net.th1, net.cfg = W1, W2, th1, saved
    return d1, d2


# ── M19: greedy vs holistic gradients (THEORY §14) ────────────────────────────

def ramp_time_grad(t_in, idx_in, w_row, T, d):
    """∂T/∂w for a ramp node crossing at T: −(T − t_i)/A for inputs that arrived before T."""
    g = np.zeros(d + 1)
    ok = np.isfinite(t_in) & (t_in < T)
    A = float(w_row[idx_in[ok]].sum())
    if A <= 0:
        return g, A
    g[idx_in[ok]] = -(T - t_in[ok]) / A
    return g, A


def soft_race(net, h_times, y, sigma_o):
    """Output crossing times (extrapolated when no crossing), loss −log softmax(−T/σ)_y, ∂L/∂T."""
    k, h = net.W2.shape[0], net.h
    spk = np.flatnonzero(np.isfinite(h_times))
    order = spk[np.argsort(h_times[spk])]
    T = np.full(k, 3.0)
    for o in range(k):
        A = B = 0.0
        for j in order:                                       # add inputs until the crossing comes first
            if A > 0 and (net.th2[o] + B) / A <= h_times[j]:
                break
            A, B = A + net.W2[o, j], B + net.W2[o, j] * h_times[j]
        if A > 0:                                             # crossing, or its extrapolation past the horizon
            T[o] = min((net.th2[o] + B) / A, 3.0)
    moving = T < 3.0                                          # a capped time does not move with the weights
    z = -T / sigma_o
    p = np.exp(z - z.max()); p /= p.sum()
    L = float(-np.log(p[y] + 1e-12))
    dT = (np.eye(k)[y] - p) / sigma_o * moving
    return T, L, dT


def holistic_gradients(net, times, ys, sigma_o, sigma_fs):
    """Greedy exact path gradient and fork (existence-flip) terms, summed over samples."""
    d, h, size = net.d, net.h, net.cfg.group
    G2 = np.zeros_like(net.W2)
    G1_greedy = np.zeros_like(net.W1)
    G1_fork = {s: np.zeros_like(net.W1) for s in sigma_fs}
    for b in range(len(ys)):
        t, idx = to_events(times[b:b + 1])
        st = net.forward(t, idx)
        fired, freeze1 = st["fired"][0], st["freeze1"][0]
        h_times = np.where(fired, freeze1, np.inf)
        T1, _, _ = layer_race(t, idx, net.W1, net.th1, "ramp")
        T1 = T1[0]
        T, L, dT = soft_race(net, h_times, ys[b], sigma_o)
        # output weights and hidden spike times (pathwise)
        dt_h = np.zeros(h)
        for o in range(net.W2.shape[0]):
            ok = np.isfinite(h_times) & (h_times < T[o])
            A = float(net.W2[o, :h][ok].sum())
            if A <= 0:
                continue
            G2[o, :h][ok] += dT[o] * (-(T[o] - h_times[ok]) / A)
            dt_h[ok] += dT[o] * net.W2[o, :h][ok] / A
        for j in np.flatnonzero(fired):
            g, _ = ramp_time_grad(t[0], idx[0], net.W1[j], h_times[j], d)
            G1_greedy[j] += dt_h[j] * g
        # forks: in each group, the last winner m and the next strand n swap
        for grp in range(h // size):
            ids = np.arange(grp * size, (grp + 1) * size)
            f = ids[fired[ids]]
            cand = ids[~fired[ids] & np.isfinite(T1[ids])]
            if not len(f) or not len(cand):
                continue
            m = f[np.argmax(freeze1[f])]
            n = cand[np.argmin(T1[cand])]
            swapped = h_times.copy()
            swapped[m], swapped[n] = np.inf, T1[n]
            _, L_swap, _ = soft_race(net, swapped, ys[b], sigma_o)
            gap = T1[n] - freeze1[m]
            g_n, _ = ramp_time_grad(t[0], idx[0], net.W1[n], T1[n], d)
            g_m, _ = ramp_time_grad(t[0], idx[0], net.W1[m], freeze1[m], d)
            for s in sigma_fs:
                pr = 1.0 / (1.0 + np.exp(gap / s))
                dp_dgap = -pr * (1 - pr) / s
                G1_fork[s][n] += (L_swap - L) * dp_dgap * g_n          # ∂gap/∂w_n = ∂T1_n/∂w_n
                G1_fork[s][m] -= (L_swap - L) * dp_dgap * g_m          # ∂gap/∂w_m = −∂t_m/∂w_m
    return G1_greedy, G1_fork, G2


def m3(a):
    global OBJECTIVE
    OBJECTIVE = a.objective
    cfg = Config(hidden=a.hidden, group=10, winners=3, hid_frac=0.6, psp="ramp", deadline=1,
                 eta_out=0.01, eta_hid=0.01, homeo=0.001, seed=a.seed)
    times, y = small_mnist(a.train)
    net = make_net(cfg, times, a.seed)
    pretrain(net, times, y, a.pretrain)
    ct, cy = small_mnist(a.samples, offset=50000)

    rng = np.random.default_rng(a.seed + 7)
    coords = {}
    for name, W in (("W1", net.W1), ("W2", net.W2)):
        n, m = W.shape
        flat = rng.choice(n * (m - 1), a.coords, replace=False)          # never the dummy column
        coords[name] = np.stack(np.unravel_index(flat, (n, m - 1)), 1)
    y_rep = np.repeat(cy, a.draws)
    rep = np.repeat(ct, a.draws, 0)

    def fd_gradient(noise_seed):
        """Central differences of the expected error, common random numbers within one estimate."""
        r = np.random.default_rng(noise_seed)
        noisy = np.clip(rep + a.sigma * r.standard_normal(rep.shape).astype(np.float32), 0, 0.999)
        events = to_events(np.where(np.isfinite(rep), noisy, np.inf).astype(np.float32))
        out = {"base": error_under_noise(net, events, y_rep)}
        for name in ("W1", "W2"):
            W = getattr(net, name)
            eps = a.rel_eps * float(np.abs(W[:, :-1]).mean())
            g = np.empty(len(coords[name]))
            for k, (i, j) in enumerate(coords[name]):
                old = W[i, j]
                W[i, j] = old + eps
                up = error_under_noise(net, events, y_rep)
                W[i, j] = old - eps
                down = error_under_noise(net, events, y_rep)
                W[i, j] = old
                g[k] = (up - down) / (2 * eps)
            out[name] = g
        return out

    def cos(u, g):
        nu, ng = np.linalg.norm(u), np.linalg.norm(g)
        return float(u @ g / (nu * ng)) if nu and ng else None

    t0 = time.time()
    ga, gb = fd_gradient(a.seed + 100), fd_gradient(a.seed + 200)
    grad = {k: (ga[k] + gb[k]) / 2 for k in ("W1", "W2")}
    fd_time = time.time() - t0

    t, idx = to_events(ct)
    res = {"config": vars(a), "base_error": (ga["base"] + gb["base"]) / 2, "fd_seconds": round(fd_time, 1),
           # reliability: agreement of two independent noise estimates (the ceiling any rule can reach
           # against a single estimate is about sqrt of this against the average)
           "fd_split_cosine": {k: cos(ga[k], gb[k]) for k in ("W1", "W2")},
           "grad_nonzero_frac": {k: float((v != 0).mean()) for k, v in grad.items()}, "rules": {}}
    for variant in ("crl_fired_only", "crl_fa", "crl_sym", "crl_sign", "frozen_hidden"):
        d1, d2 = rule_update(net, variant, t, idx, cy)
        row = {}
        for name, d in (("W1", d1), ("W2", d2)):
            u = -d[coords[name][:, 0], coords[name][:, 1]]      # an update descends: compare −Δw with ∇
            row[name] = {"cosine": cos(u, grad[name]), "update_nonzero_frac": float((u != 0).mean())}
        res["rules"][variant] = row
    if a.check == "m19":
        g1, forks, g2 = holistic_gradients(net, ct, cy, a.sigma_o, (0.01, 0.03, 0.1))
        on = lambda G, name: G[coords[name][:, 0], coords[name][:, 1]]   # noqa: E731
        hol = {"greedy_W1": {"cosine": cos(on(g1, "W1"), grad["W1"]),
                             "nonzero_frac": float((on(g1, "W1") != 0).mean())},
               "greedy_W2": {"cosine": cos(on(g2, "W2"), grad["W2"]),
                             "nonzero_frac": float((on(g2, "W2") != 0).mean())}}
        for sf, gf in forks.items():
            u = on(gf, "W1")
            hol[f"fork_W1_s{sf}"] = {"cosine": cos(u, grad["W1"]), "nonzero_frac": float((u != 0).mean())}
            # the two terms are on different scales; report the best mix and the equal-norm mix
            a1, a2 = on(g1, "W1"), u
            if np.linalg.norm(a1) and np.linalg.norm(a2):
                mix = a1 / np.linalg.norm(a1) + a2 / np.linalg.norm(a2)
                hol[f"holistic_W1_s{sf}"] = {"cosine": cos(mix, grad["W1"]),
                                             "nonzero_frac": float((mix != 0).mean())}
        res["holistic"] = hol
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.check}_{a.tag or 'run'}_s{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=("m3", "m19"))
    ap.add_argument("--objective", default="error", choices=("error", "margin"))
    ap.add_argument("--sigma-o", type=float, default=0.05, help="output race temperature (M19)")
    ap.add_argument("--hidden", type=int, default=60)
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--pretrain", type=int, default=1)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--coords", type=int, default=200)
    ap.add_argument("--rel-eps", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    m3(ap.parse_args())
