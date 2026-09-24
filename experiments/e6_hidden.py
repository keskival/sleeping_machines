#!/usr/bin/env python3
"""E6: counterfactual race learning (CRL) through a hidden layer.

See experiments/E6_HIDDEN_LAYERS_DESIGN.md.

Network: input spikes -> hidden k-winner races in groups -> K-way output race.
Neurons are non-leaky integrate-to-threshold with one spike each, so the event
simulation has an exact closed form: sort the input spikes by time, accumulate
weights, and take the first threshold crossing. `test_equivalence` checks this
against the discrete-event engine.

    python experiments/e6_hidden.py xor  --variant crl_fa
    python experiments/e6_hidden.py mnist --variant crl_fa --epochs 3
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sleeping_machines.sim import Engine  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e6")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
HORIZON = 1.0

VARIANTS = ("crl_sym", "crl_fa", "crl_sign", "crl_fired_only", "frozen_hidden", "single_layer", "mlp")


# ── Spike encodings ───────────────────────────────────────────────────────────

def to_events(times):
    """Dense (B, D) spike times with inf for silence -> time-sorted (times, idx), padded.
    Padding points at a dummy input D whose weights are kept at zero."""
    b, d = times.shape
    s = int(np.isfinite(times).sum(1).max())
    order = np.argsort(times, axis=1, kind="stable")[:, :s]
    t = np.take_along_axis(times, order, 1)
    idx = np.where(np.isfinite(t), order, d)
    return t.astype(np.float32), idx


def mnist(split):
    os.makedirs(DATA, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    names = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
             "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}[split]
    arrays = []
    for name, offset in zip(names, (16, 8)):
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + name, path)
        with gzip.open(path) as f:
            arrays.append(np.frombuffer(f.read(), np.uint8, offset=offset))
    x = arrays[0].reshape(-1, 784).astype(np.float32) / 255
    return x, arrays[1].astype(np.int64)


def latency_code(x, cutoff=0.1):
    """Brighter pixel -> earlier spike; pixels below cutoff stay silent."""
    return np.where(x > cutoff, (1 - x) * 0.999, np.inf).astype(np.float32)


def xor_in_time(n, rng, per_group=16, groups=4):
    """Label = XOR(group 0 early, group 1 early). Groups 2.. are timing distractors."""
    early = rng.random((n, groups)) < 0.5
    y = (early[:, 0] ^ early[:, 1]).astype(np.int64)
    lo = np.where(early, 0.0, 0.6)[:, :, None]
    t = lo + rng.uniform(0, 0.4, (n, groups, per_group))
    t = np.where(rng.random(t.shape) < 0.8, t, np.inf)
    return t.reshape(n, -1).astype(np.float32), y


# ── Closed-form race dynamics ─────────────────────────────────────────────────

def integrate(t, idx, W, weighted=False):
    """Cumulative potential of every node after each input event: (B, N, S).
    With `weighted`, accumulate weight x input time instead (the ramp offset B)."""
    w = np.transpose(W[:, idx], (1, 0, 2))
    if weighted:
        w = w * t[:, None, :]
    return np.cumsum(w, axis=2)


def crossing_times(t, cum, theta):
    """First threshold crossing per node, and the overshoot at that instant.

    Simultaneous input events are integrated together: the threshold is only
    checked after the last event of each equal-time run."""
    last_of_run = np.ones_like(t, bool)
    last_of_run[:, :-1] = t[:, 1:] != t[:, :-1]
    crossed = (cum >= theta[None, :, None]) & (last_of_run & np.isfinite(t))[:, None, :]
    first = crossed.argmax(2)
    tc = np.take_along_axis(np.broadcast_to(t[:, None, :], cum.shape), first[:, :, None], 2)[:, :, 0]
    over = np.take_along_axis(cum, first[:, :, None], 2)[:, :, 0] - theta[None, :]
    has = crossed.any(2)
    return np.where(has, tc, np.inf), np.where(has, over, -np.inf)


def ramp_crossing(t, A, Bc, theta):
    """Current-based (ramp) synapses: after the inputs up to t_s, v(t) = A_s t - B_s.
    The node crosses at t* = (theta + B_s) / A_s if that falls before the next input
    (or the horizon). Returns first crossing times and the slope there (tie-break)."""
    last_of_run = np.ones_like(t, bool)
    last_of_run[:, :-1] = t[:, 1:] != t[:, :-1]
    t_next = np.full_like(t, HORIZON)
    t_next[:, :-1] = np.where(np.isfinite(t[:, 1:]), t[:, 1:], HORIZON)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = np.where(A > 0, (theta[None, :, None] + Bc) / A, np.inf)
    valid = ((last_of_run & np.isfinite(t))[:, None, :] & (A > 0)
             & (tstar < np.minimum(t_next, HORIZON)[:, None, :]) & (tstar >= t[:, None, :] - 1e-9))
    first = valid.argmax(2)
    has = valid.any(2)
    T = np.take_along_axis(tstar, first[:, :, None], 2)[:, :, 0]
    slope = np.take_along_axis(A, first[:, :, None], 2)[:, :, 0]
    T = np.maximum(T, np.take_along_axis(np.broadcast_to(t[:, None, :], A.shape), first[:, :, None], 2)[:, :, 0])
    return np.where(has, T, np.inf), np.where(has, slope, -np.inf)


def ramp_potential_at(t, A, Bc, when):
    """v(when) = A_s when - B_s using the inputs that arrived by `when`."""
    b, n, s = A.shape
    count = (t[:, None, :] <= when[:, :, None]).sum(2)
    idx = np.clip(count - 1, 0, s - 1)[:, :, None]
    v = np.take_along_axis(A, idx, 2)[:, :, 0] * when - np.take_along_axis(Bc, idx, 2)[:, :, 0]
    return np.where(count > 0, v, 0.0), count


def layer_race(t, idx, W, theta, psp):
    """Integrate one layer; returns crossing times, tie-break key and a potential-at function."""
    if psp == "ramp":
        A = integrate(t, idx, W)
        Bc = integrate(np.where(np.isfinite(t), t, 0), idx, W, weighted=True)
        T, key = ramp_crossing(t, A, Bc, theta)
        return T, key, lambda when: ramp_potential_at(t, A, Bc, when)
    cum = integrate(t, idx, W)
    T, key = crossing_times(t, cum, theta)
    return T, key, lambda when: potential_at(t, cum, when)


def race_order(T, over):
    """Rank along the last axis: earlier crossing first; at the same instant,
    larger overshoot first (two stable sorts, fully vectorised)."""
    o1 = np.argsort(-over, axis=-1, kind="stable")
    o2 = np.argsort(np.take_along_axis(T, o1, -1), axis=-1, kind="stable")
    return np.take_along_axis(o1, o2, -1)


def potential_at(t, cum, when):
    """Potential of each node at time `when` (B, N): sum of inputs with t <= when."""
    b, n, s = cum.shape
    count = (t[:, None, :] <= when[:, :, None]).sum(2)
    v = np.take_along_axis(cum, np.clip(count - 1, 0, s - 1)[:, :, None], 2)[:, :, 0]
    return np.where(count > 0, v, 0.0), count


def group_race(T, over, groups, winners):
    """Within each group the first `winners` crossings fire; the rest are cancelled
    at the instant the last winner fires (or at the horizon if too few crossed)."""
    b, n = T.shape
    size = n // groups
    g, o = T.reshape(b, groups, size), over.reshape(b, groups, size)
    order = race_order(g, o)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(size)[None, None, :].repeat(b, 0).repeat(groups, 1), 2)
    fired = (rank < winners) & np.isfinite(g)
    kth = np.take_along_axis(g, order[:, :, winners - 1:winners], 2)[:, :, 0]
    cancel = np.where(np.isfinite(kth), kth, HORIZON)
    freeze = np.where(fired, g, cancel[:, :, None])
    return fired.reshape(b, n), freeze.reshape(b, n)


# ── Network ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    variant: str = "crl_fa"
    hidden: int = 1000
    group: int = 10          # hidden nodes per competition group
    winners: int = 1         # hidden winners per group
    eta_out: float = 0.003
    eta_hid: float = 0.003
    sigma: float = 0.15
    margin: float = 0.1      # also learn when correct but a competitor came within this Δ
    homeo: float = 0.001     # hidden adaptive threshold rate
    init_frac: float = 0.3   # nodes reach θ after about this fraction of their inputs at init
    hid_frac: float = 0.0    # same for hidden nodes only (0 = use init_frac)
    patch: int = 0           # local receptive fields: each hidden group sees a patch x patch window (0 = whole image)
    stride: int = 4          # spacing of patch positions; groups are assigned to positions round-robin
    lateral: int = 0         # output race on relative evidence: each spike also inhibits all outputs by the mean weight
    theta_out: float = 1.0   # output threshold
    lr_decay: float = 1.0    # learning rates multiplied by this after every epoch
    deadline: int = 0        # collapse the output threshold at the horizon so the leader fires
    psp: str = "step"        # "step": each spike adds w at once; "ramp": it injects a constant current w
    scaling: int = 0         # subtractive synaptic scaling (constant summed weight per node)
    zero_sum: int = 0        # normalise competitor credit so the output signal sums to zero
    batch: int = 32
    epochs: int = 3
    seed: int = 0


class RaceNet:
    def __init__(self, cfg, d_in, k, mean_spikes, rng, rates=None):
        """mean_spikes / rates: expected drive per sample and per input (spike counts
        for step synapses, charge sum(H - t) for ramp synapses)."""
        self.cfg, self.k, self.d = cfg, k, d_in
        h = cfg.hidden if cfg.variant != "single_layer" else 0
        self.h = h
        self.M1 = None
        if h:
            self.W1 = np.zeros((h, d_in + 1), np.float32)
            if cfg.patch:
                self.M1 = patch_mask(h, cfg.group, cfg.patch, cfg.stride)
                rates = np.full(d_in, mean_spikes / d_in) if rates is None else rates
                expected = self.M1[:, :d_in] @ rates                  # expected input spikes per node
                mu = (1.0 / ((cfg.hid_frac or cfg.init_frac) * np.maximum(expected, 1.0)))[:, None]
                self.W1[:, :d_in] = rng.normal(mu, mu, (h, d_in)) * self.M1[:, :d_in]
            else:
                mu = 1.0 / ((cfg.hid_frac or cfg.init_frac) * mean_spikes)
                self.W1[:, :d_in] = rng.normal(mu, mu, (h, d_in))
            self.th1 = np.ones(h, np.float32)
            fired_per_sample = h // cfg.group * cfg.winners * (0.5 if cfg.psp == "ramp" else 1.0)
            mu2 = 1.0 / (cfg.init_frac * fired_per_sample)
            self.W2 = np.zeros((k, h + 1), np.float32)
            self.W2[:, :h] = rng.normal(mu2, mu2, (k, h))
            self.B = rng.normal(0, mu2, (k, h)).astype(np.float32)   # fixed feedback for crl_fa
        else:
            mu2 = 1.0 / (cfg.init_frac * mean_spikes)
            self.W2 = np.zeros((k, d_in + 1), np.float32)
            self.W2[:, :d_in] = rng.normal(mu2, mu2, (k, d_in))
        self.th2 = np.full(k, cfg.theta_out, np.float32)
        self.lr_mult = 1.0
        self.work = {"synops": 0, "plasticity": 0, "hidden_spikes": 0, "input_spikes": 0, "samples": 0}

    # Forward: one race per layer ------------------------------------------------

    def forward(self, t, idx):
        cfg, st = self.cfg, {}
        st["t_in"], st["idx_in"] = t, idx
        self.work["input_spikes"] += int(np.isfinite(t).sum())
        self.work["samples"] += len(t)
        if self.h:
            T1, over1, v_at1 = layer_race(t, idx, self.W1, self.th1, cfg.psp)
            fired, freeze1 = group_race(T1, over1, self.h // cfg.group, cfg.winners)
            v1, n_before = v_at1(freeze1)
            st.update(fired=fired, freeze1=freeze1, snap1=np.clip((self.th1 - v1) / self.th1, 0, None))
            if self.M1 is None:
                self.work["synops"] += int(n_before.sum())      # inputs after inhibition are not integrated
            else:                                               # only existing synapses carry events
                before = t[:, None, :] <= freeze1[:, :, None]
                self.work["synops"] += int((np.transpose(self.M1[:, idx], (1, 0, 2)) & before).sum())
            self.work["hidden_spikes"] += int(fired.sum())
            h_times = np.where(fired, freeze1, np.inf)
            t2, idx2 = to_events(h_times)
            idx2 = np.where(idx2 == self.h, self.h, idx2)
        else:
            t2, idx2 = t, idx
        T2, over2, v_at2 = layer_race(t2, idx2, self.w_out(), self.th2, cfg.psp)
        winner = np.where(np.isfinite(T2).any(1), race_order(T2, over2)[:, 0], -1)
        t_dec = np.where(winner >= 0, T2.min(1), HORIZON)
        freeze2 = np.repeat(t_dec[:, None], self.k, 1)
        v2, n_before2 = v_at2(freeze2)
        urgent = winner < 0
        if cfg.deadline:            # collapsing bound: at the horizon the leading output fires
            winner = np.where(urgent, v2.argmax(1), winner)
        snap2 = np.clip((self.th2 - v2) / self.th2, 0, None)
        snap2[winner >= 0, winner[winner >= 0]] = 0
        self.work["synops"] += int(n_before2.sum())
        st.update(t2=t2, idx2=idx2, winner=winner, freeze2=freeze2, snap2=snap2, urgent=urgent)
        return st

    def w_out(self):
        """Effective output weights. With lateral inhibition every incoming spike also
        subtracts the mean weight from all outputs, so the race runs on relative evidence."""
        if not self.cfg.lateral:
            return self.W2
        W = self.W2 - self.W2.mean(0, keepdims=True)
        W[:, -1] = 0
        return W

    # Teaching event -------------------------------------------------------------

    def teach(self, st, y):
        cfg = self.cfg
        b = len(y)
        rows = np.arange(b)
        snap2 = st["snap2"]
        elig2 = np.exp(-snap2 / cfg.sigma)
        rival = snap2.copy()
        rival[rows, y] = np.inf
        # a decision forced by the deadline is uncertain, like a close call: it still teaches
        update = (st["winner"] != y) | (rival.min(1) < cfg.margin) | st["urgent"]
        s = -elig2 * (elig2 >= 0.05)                       # competitors, near-miss weighted
        s[rows, y] = 0.0
        if cfg.zero_sum:                                   # competitors share a total of -1
            s /= np.maximum(-s.sum(1, keepdims=True), 1e-9)
        s[rows, y] = 1.0
        s *= update[:, None]

        mask2 = self._elig(st["t2"], st["freeze2"])
        self._apply(self.W2, st["idx2"], cfg.eta_out * self.lr_mult * s, mask2, self.W2.shape[1] - 1)

        if not self.h or cfg.variant == "frozen_hidden":
            return
        if cfg.variant in ("crl_sym", "crl_sign"):
            # Class-specific part only: the common component would push every hidden
            # node the same way whenever the output signal does not sum to zero.
            B = self.W2[:, :self.h] - self.W2[:, :self.h].mean(0, keepdims=True)
            if cfg.variant == "crl_sign":
                B = np.sign(B) * np.abs(B).mean()
        else:
            B = self.B
        delta = s @ B                                      # feedback events o -> h
        fired = st["fired"]
        if cfg.variant == "crl_fired_only":
            elig1 = fired.astype(np.float32)
        else:
            elig1 = np.where(fired, 1.0, np.exp(-st["snap1"] / cfg.sigma))
            elig1 *= elig1 >= 0.05
        coef = cfg.eta_hid * self.lr_mult * delta * elig1
        mask1 = self._elig(st["t_in"], st["freeze1"])
        self._apply(self.W1, st["idx_in"], coef, mask1, self.d, self.M1)
        if cfg.homeo:
            target_rate = cfg.winners / cfg.group
            self.th1 += cfg.homeo * (fired.mean(0) - target_rate)
            np.maximum(self.th1, 0.05, out=self.th1)

    def _elig(self, t, freeze):
        """Presynaptic eligibility at the node's fire/cancel time: 1 for step synapses
        that had arrived; for ramp synapses the charge they had injected, freeze - t."""
        if self.cfg.psp == "ramp":
            return np.clip(freeze[:, :, None] - np.where(np.isfinite(t), t, np.inf)[:, None, :], 0, None)
        return t[:, None, :] <= freeze[:, :, None]

    def _apply(self, W, idx, coef, mask, dummy, exists=None):
        """W[n, idx[b, s]] += coef[b, n] * mask[b, n, s]; each input spikes once per sample.

        With `scaling`, each node's summed weight is held constant (subtractive
        synaptic scaling). It is applied as one per-node scalar, equivalent to
        feedforward inhibition per incoming spike, and counted as one update per node."""
        added = np.zeros(W.shape[0], np.float32)
        for bi in range(len(idx)):
            active = np.flatnonzero(coef[bi])
            if not len(active):
                continue
            m = mask[bi, active]
            if exists is not None:
                m = m * exists[np.ix_(active, idx[bi])]
            upd = coef[bi, active, None] * m
            W[np.ix_(active, idx[bi])] += upd
            added[active] += upd.sum(1)
            self.work["plasticity"] += int((m > 0).sum())
        if self.cfg.scaling:
            nz = np.flatnonzero(added)
            W[nz, :dummy] -= added[nz, None] / dummy
            self.work["plasticity"] += len(nz)
        W[:, dummy] = 0


def patch_mask(h, group, patch, stride, side=28):
    """Hidden-to-input connectivity: group g sees the patch at position g mod n_positions."""
    starts = range(0, side - patch + 1, stride)
    positions = [(r, c) for r in starts for c in starts]
    M = np.zeros((h, side * side + 1), bool)
    for g in range(h // group):
        r, c = positions[g % len(positions)]
        win = np.zeros((side, side), bool)
        win[r:r + patch, c:c + patch] = True
        M[g * group:(g + 1) * group, :side * side] = win.ravel()
    return M


def evaluate(net, times, y, batch=250, count=False):
    saved = dict(net.work)
    correct = 0
    for i in range(0, len(y), batch):
        t, idx = to_events(times[i:i + batch])
        correct += int((net.forward(t, idx)["winner"] == y[i:i + batch]).sum())
    if not count:
        net.work = saved                     # evaluation is not part of the training bill
    return correct / len(y)


def train(cfg, xtr, ytr, xte, yte, log_every=0):
    rng = np.random.default_rng(cfg.seed)
    if cfg.psp == "ramp":
        drive = np.where(np.isfinite(xtr), HORIZON - xtr, 0).mean(0)
    else:
        drive = np.isfinite(xtr).mean(0)
    mean_spikes, rates = float(drive.sum()), drive
    net = RaceNet(cfg, xtr.shape[1], int(ytr.max()) + 1, mean_spikes, rng, rates)
    curve = []
    for ep in range(cfg.epochs):
        perm = rng.permutation(len(ytr))
        t0 = time.time()
        for j, i in enumerate(range(0, len(perm), cfg.batch)):
            ii = perm[i:i + cfg.batch]
            t, idx = to_events(xtr[ii])
            net.teach(net.forward(t, idx), ytr[ii])
            if log_every and j % log_every == 0:
                print(f"  ep {ep} batch {j} ({time.time() - t0:.0f}s)", flush=True)
        work_train = dict(net.work)
        net.lr_mult *= cfg.lr_decay
        acc = evaluate(net, xte, yte)
        curve.append(acc)
        print(f"{cfg.variant} epoch {ep + 1}: test {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.work = {k: 0 for k in net.work}
    evaluate(net, xte, yte, count=True)
    per_sample = {k: v / net.work["samples"] for k, v in net.work.items() if k != "samples"}
    return {"config": asdict(cfg), "curve": curve, "test_acc": curve[-1],
            "inference_work_per_sample": per_sample, "train_work": work_train}


def train_mlp(cfg, xtr, ytr, xte, yte, lr=0.05):
    """Dense reference: one ReLU hidden layer, softmax output, plain SGD with backprop.
    Input is spike-derived intensity (1 - t for spiking pixels, 0 otherwise)."""
    rng = np.random.default_rng(cfg.seed)
    def feats(t):
        return np.where(np.isfinite(t), 1 - t, 0).astype(np.float32)
    ftr, fte = feats(xtr), feats(xte)
    d, h, k = ftr.shape[1], cfg.hidden, int(ytr.max()) + 1
    W1 = rng.normal(0, np.sqrt(2 / d), (d, h)).astype(np.float32)
    W2 = rng.normal(0, np.sqrt(1 / h), (h, k)).astype(np.float32)
    b1, b2 = np.zeros(h, np.float32), np.zeros(k, np.float32)
    curve = []
    for ep in range(cfg.epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), cfg.batch):
            ii = perm[i:i + cfg.batch]
            x = ftr[ii]
            a1 = np.maximum(x @ W1 + b1, 0)
            z = a1 @ W2 + b2
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            p[np.arange(len(ii)), ytr[ii]] -= 1
            p /= len(ii)
            g1 = (p @ W2.T) * (a1 > 0)
            W2 -= lr * a1.T @ p; b2 -= lr * p.sum(0)
            W1 -= lr * x.T @ g1; b1 -= lr * g1.sum(0)
        acc = float(((np.maximum(fte @ W1 + b1, 0) @ W2 + b2).argmax(1) == yte).mean())
        curve.append(acc)
        print(f"mlp epoch {ep + 1}: test {acc:.4f}", flush=True)
    macs = d * h + h * k
    return {"config": asdict(cfg), "curve": curve, "test_acc": curve[-1],
            "inference_work_per_sample": {"macs": macs},
            "train_work": {"macs": 3 * macs * cfg.epochs * len(ytr)}}


# ── Equivalence with the event engine ─────────────────────────────────────────

def event_forward(net, times_row):
    """Run one sample through the same network on the discrete-event engine.
    Events at the same instant are delivered as one batch (same semantics as the
    closed form); simultaneous crossings are resolved by overshoot."""
    e = Engine()
    cfg = net.cfg
    h, k = net.h, net.k
    v1, v2 = np.zeros(h), np.zeros(k)
    group_fired = np.zeros(h // cfg.group, int)
    W2e = net.w_out()
    inhibited = np.zeros(h, bool)
    done = {}

    def inputs(chans):
        live = np.flatnonzero(~inhibited)
        v1[live] += net.W1[np.ix_(live, chans)].sum(1)
        over = v1[live] - net.th1[live]
        spikes = []
        for n in live[over >= 0][np.argsort(-over[over >= 0], kind="stable")]:
            g = n // cfg.group
            if group_fired[g] < cfg.winners:
                group_fired[g] += 1
                inhibited[n] = True
                spikes.append(int(n))
                if group_fired[g] == cfg.winners:
                    inhibited[g * cfg.group:(g + 1) * cfg.group] = True
        if spikes:
            e.schedule(0.0, "hidden", spikes)

    def hidden(ns):
        if "winner" in done:
            return
        v2[:] += W2e[:, ns].sum(1)
        over = v2 - net.th2
        if (over >= 0).any():
            done["winner"] = int(np.argmax(np.where(over >= 0, over, -np.inf)))

    e.on("in", inputs)
    e.on("hidden", hidden)
    active = np.flatnonzero(np.isfinite(times_row))
    for tval in np.unique(times_row[active]):
        e.schedule(float(tval), "in", active[times_row[active] == tval])
    e.run()
    if "winner" not in done and cfg.deadline:
        return int(np.argmax(v2))
    return done.get("winner", -1)


def event_forward_ramp(net, times_row):
    """Ramp synapses on the event engine. Every input changes a node's slope, so its
    predicted fire event is cancelled and rescheduled: the pending-future-event pool
    in action. Simultaneous inputs are delivered as one batch."""
    e = Engine()
    cfg = net.cfg
    h, k = net.h, net.k
    A1, B1, A2, B2 = np.zeros(h), np.zeros(h), np.zeros(k), np.zeros(k)
    W2e = net.w_out()
    group_fired = np.zeros(h // cfg.group, int)
    inhibited = np.zeros(h, bool)
    pend1, pend2, done = {}, {}, {}

    def predict(A, B, th, n):
        return (th[n] + B[n]) / A[n] if A[n] > 0 else np.inf

    def reschedule(pend, n, tstar, kind):
        if n in pend:
            e.cancel(pend.pop(n))
        if np.isfinite(tstar) and tstar < HORIZON:
            pend[n] = e.schedule(max(tstar - e.now, 0.0), kind, int(n))

    def inputs(chans):
        live = np.flatnonzero(~inhibited)
        dw = net.W1[np.ix_(live, chans)].sum(1)
        A1[live] += dw
        B1[live] += dw * e.now
        for n in live:
            reschedule(pend1, n, predict(A1, B1, net.th1, n), "hfire")

    def hfire(n):
        pend1.pop(n, None)
        g = n // cfg.group
        if inhibited[n]:
            return
        inhibited[n] = True
        group_fired[g] += 1
        if group_fired[g] == cfg.winners:
            for m in range(g * cfg.group, (g + 1) * cfg.group):
                inhibited[m] = True
                if m in pend1:
                    e.cancel(pend1.pop(m))
        if "winner" in done:
            return
        A2[:] += W2e[:, n]
        B2[:] += W2e[:, n] * e.now
        for o in range(k):
            reschedule(pend2, o, predict(A2, B2, net.th2, o), "ofire")

    def ofire(o):
        if "winner" not in done:
            done["winner"] = int(o)
            for ev in pend2.values():
                e.cancel(ev)
            pend2.clear()

    e.on("in", inputs); e.on("hfire", hfire); e.on("ofire", ofire)
    active = np.flatnonzero(np.isfinite(times_row))
    for tval in np.unique(times_row[active]):
        e.schedule(float(tval), "in", active[times_row[active] == tval])
    e.run(until=HORIZON)
    if "winner" not in done and cfg.deadline:
        return int(np.argmax(A2 * HORIZON - B2))
    return done.get("winner", -1)


def test_equivalence(net, times, n=200):
    t, idx = to_events(times[:n])
    closed = net.forward(t, idx)["winner"]
    ref = event_forward_ramp if net.cfg.psp == "ramp" else event_forward
    events = np.array([ref(net, times[i]) for i in range(n)])
    return float((closed == events).mean())


# ── Entry points ──────────────────────────────────────────────────────────────

def load(task, n_val):
    if task == "xor":
        rng = np.random.default_rng(0)
        xtr, ytr = xor_in_time(8000, rng)
        xte, yte = xor_in_time(2000, rng)
        return xtr, ytr, xte, yte
    x, y = mnist("train")
    xt, yt = mnist("test")
    if n_val:                      # tuning uses the last n_val training images as validation
        return latency_code(x[:-n_val]), y[:-n_val], latency_code(x[-n_val:]), y[-n_val:]
    return latency_code(x), y, latency_code(xt), yt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("xor", "mnist"))
    ap.add_argument("--variant", default="crl_fa", choices=VARIANTS)
    ap.add_argument("--val", type=int, default=0, help="hold out N training samples as validation")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--log-every", type=int, default=0)
    for f, v in asdict(Config()).items():
        if f != "variant":
            ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = Config(**{f: getattr(a, f) for f in asdict(Config())})
    xtr, ytr, xte, yte = load(a.task, a.val)
    if a.train_limit:
        xtr, ytr = xtr[:a.train_limit], ytr[:a.train_limit]
    res = train(cfg, xtr, ytr, xte, yte, a.log_every) \
        if cfg.variant != "mlp" else train_mlp(cfg, xtr, ytr, xte, yte)
    res["task"], res["val"] = a.task, a.val
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.task}_{cfg.variant}{'_' + a.tag if a.tag else ''}_s{cfg.seed}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("test_acc", "inference_work_per_sample")}, indent=1))
