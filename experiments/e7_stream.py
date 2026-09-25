#!/usr/bin/env python3
"""E7: a race network that lives in a causal stream.

See experiments/E7_PREREGISTRATION.md.

There are no epochs and no separate training phase. The learner sees one stream
of frames, predicts each frame before it may learn from it, and carries slow
state across frames. Resets mark episode limits (one object seen as several
views). Fast state (the race itself) starts afresh at every frame, exactly as in
E6, so the closed-form solver and its equivalence test still apply. Slow state
carries over:

  prior        a winner's threshold stays lowered for a while (decays over frames)
  tags         synapses keep a decaying trace of their charge, so a late label still
               reaches the synapses that caused the decision
  carry        a label seen in an episode teaches the rest of that episode
  continuity   hidden winners of the previous view are pulled toward winning again
  consolidation  synapses that have changed a lot become harder to change

Labels are either random (a fixed rate) or requested: the learner asks when its
own race is uncertain (a close call, or a decision forced by the deadline), under
a token bucket with the same average budget.

    python experiments/e7_stream.py race --order episodes --label-rate 0.1 --carry 1
    python experiments/e7_stream.py mlp  --order blocked --replay 1000
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import Config, RaceNet, latency_code, mnist, to_events, evaluate  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e7")
TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


# ── The stream ────────────────────────────────────────────────────────────────

@dataclass
class Stream:
    order: str = "episodes"   # iid | episodes | shuffled | blocked
    frames: int = 10000
    length: int = 5           # views per episode (iid: 1)
    shift: int = 2            # views are the image shifted by up to this many pixels
    label_rate: float = 1.0   # average fraction of frames that get a label
    labels: str = "random"    # random | ask (the learner requests a label when uncertain)
    delay: int = 0            # a label arrives this many frames after its frame
    bucket: float = 20.0      # most labels the learner can have saved up (ask)
    seed: int = 0


def views(images, rng, shift):
    """One randomly shifted view per image (zero fill)."""
    n = len(images)
    img = images.reshape(n, 28, 28)
    pad = np.pad(img, ((0, 0), (shift, shift), (shift, shift)))
    dy, dx = rng.integers(0, 2 * shift + 1, (2, n))
    out = np.empty_like(img)
    for i in range(n):
        out[i] = pad[i, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28]
    return out.reshape(n, 784)


def make_stream(sc, x, y):
    """Frames, labels, reset flags (start of an episode) and task ids."""
    rng = np.random.default_rng(sc.seed + 1000)
    L = 1 if sc.order == "iid" else sc.length
    if sc.order == "blocked":
        per_task = sc.frames // len(TASKS)
        src, task = [], []
        for k, cls in enumerate(TASKS):
            pool = np.flatnonzero(np.isin(y, cls))
            n_ep = per_task // L
            src.append(np.repeat(rng.choice(pool, n_ep, replace=n_ep > len(pool)), L))
            task.append(np.full(n_ep * L, k))
        src, task = np.concatenate(src), np.concatenate(task)
    else:
        n_ep = sc.frames // L
        src = np.repeat(rng.choice(len(y), n_ep, replace=n_ep > len(y)), L)
        task = np.zeros(len(src), int)
    frames = views(x[src], rng, sc.shift)
    reset = np.zeros(len(src), bool)
    reset[::L] = True
    labels = y[src]
    if sc.order == "shuffled":        # same frames and labels, temporal order destroyed
        perm = rng.permutation(len(src))
        frames, labels, src = frames[perm], labels[perm], src[perm]
    return frames, labels, reset, task


class Labeller:
    """Decides which frames get a label: random at a fixed rate, or on request."""

    def __init__(self, sc, n):
        self.sc = sc
        self.mask = np.random.default_rng(sc.seed + 2000).random(n) < sc.label_rate
        self.tokens = 0.0
        self.used = 0

    def __call__(self, i, uncertain):
        sc = self.sc
        if sc.labels == "random":
            give = bool(self.mask[i])
        else:
            self.tokens = min(self.tokens + sc.label_rate, sc.bucket)
            give = uncertain and self.tokens >= 1.0
            if give:
                self.tokens -= 1.0
        self.used += give
        return give


# ── Learners ──────────────────────────────────────────────────────────────────

@dataclass
class Mech:
    prior: float = 0.0        # threshold lowering of the last winner (fraction of θ)
    tau_prior: float = 3.0    # frames
    prior_across: int = 0     # keep the prior across episode resets
    tags: int = 1             # late labels reach the synapses of their own frame (0: the current frame)
    tau_tag: float = 5.0      # frames
    carry: int = 0            # a label seen in an episode teaches its later frames
    continuity: float = 0.0   # learning rate of the view-to-view pull on hidden nodes
    consolidation: float = 0.0  # effective rate η / (1 + c · accumulated |Δw|)


class StreamRace(RaceNet):
    """E6 RaceNet with slow, carried state. The race itself is unchanged."""

    def __init__(self, cfg, mech, *args):
        super().__init__(cfg, *args)
        self.mech = mech
        self.prior = np.zeros(self.k, np.float32)
        self.imp = {}

    def predict(self, frame_times):
        t, idx = to_events(frame_times[None])
        base = self.th2
        if self.mech.prior:
            self.th2 = base * (1 - self.mech.prior * self.prior)
        st = self.forward(t, idx)
        self.th2 = base
        rival = st["snap2"][0].copy()
        if st["winner"][0] >= 0:
            rival[st["winner"][0]] = np.inf
        st["uncertain"] = bool(st["urgent"][0] or rival.min() < self.cfg.margin)
        st["used"] = int((t[0] <= st["freeze2"][0, 0]).sum())   # input events before the decision
        return st

    def learn(self, st, label, scale=1.0):
        saved, homeo = self.lr_mult, self.cfg.homeo
        self.lr_mult, self.cfg.homeo = saved * scale, 0.0     # homeostasis runs once per frame, in after()
        self.teach(st, np.array([label]))
        self.lr_mult, self.cfg.homeo = saved, homeo

    def pull(self, st, prev_fired):
        """Continuity: hidden nodes that won on the previous view and nearly won now
        are strengthened toward the inputs that drove them (no label involved)."""
        cfg = self.cfg
        near = np.exp(-st["snap1"] / cfg.sigma)
        near *= near >= 0.05
        coef = self.mech.continuity * (prev_fired & ~st["fired"]) * near
        mask = self._elig(st["t_in"], st["freeze1"])
        self._apply(self.W1, st["idx_in"], coef, mask, self.d, self.M1)

    def after(self, st, reset_next):
        cfg = self.cfg
        if self.h and cfg.homeo:                  # homeostasis needs no label: every frame
            self.th1 += cfg.homeo * (st["fired"].mean(0) - cfg.winners / cfg.group)
            np.maximum(self.th1, 0.05, out=self.th1)
            self.work["homeo"] += int(st["fired"].sum())
        self.prior *= np.exp(-1.0 / self.mech.tau_prior)
        w = st["winner"][0]
        if w >= 0:
            self.prior[w] = 1.0
        if reset_next and not self.mech.prior_across:
            self.prior[:] = 0

    def _apply(self, W, idx, coef, mask, dummy, exists=None):
        c = self.mech.consolidation
        if not c:
            return super()._apply(W, idx, coef, mask, dummy, exists)
        imp = self.imp.setdefault(id(W), np.zeros_like(W))
        slow = 1.0 / (1.0 + c * np.transpose(imp[:, idx], (1, 0, 2)))
        before = W.copy()
        super()._apply(W, idx, coef, mask * slow, dummy, exists)
        imp += np.abs(W - before)


class StreamMLP:
    """Dense reference trained by SGD one frame at a time, optionally with replay."""

    def __init__(self, cfg, d, k, replay, rng, lr=0.01):
        self.W1 = rng.normal(0, np.sqrt(2 / d), (d, cfg.hidden)).astype(np.float32)
        self.W2 = rng.normal(0, np.sqrt(1 / cfg.hidden), (cfg.hidden, k)).astype(np.float32)
        self.b1, self.b2 = np.zeros(cfg.hidden, np.float32), np.zeros(k, np.float32)
        self.lr, self.replay, self.rng = lr, replay, rng
        self.buf, self.seen = [], 0
        self.macs = 0

    @staticmethod
    def feats(t):
        return np.where(np.isfinite(t), 1 - t, 0).astype(np.float32)

    def predict(self, frame_times):
        x = self.feats(frame_times)[None]
        a1 = np.maximum(x @ self.W1 + self.b1, 0)
        z = (a1 @ self.W2 + self.b2)[0]
        self.macs += self.W1.size + self.W2.size
        p = np.exp(z - z.max()); p /= p.sum()
        top = np.sort(p)[-2:]
        return {"x": x, "winner": np.array([int(p.argmax())]), "uncertain": bool(top[1] - top[0] < 0.2),
                "used": int(np.isfinite(frame_times).sum())}

    def _step(self, x, label):
        a1 = np.maximum(x @ self.W1 + self.b1, 0)
        z = a1 @ self.W2 + self.b2
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        p[0, label] -= 1
        g1 = (p @ self.W2.T) * (a1 > 0)
        self.W2 -= self.lr * a1.T @ p; self.b2 -= self.lr * p[0]
        self.W1 -= self.lr * x.T @ g1; self.b1 -= self.lr * g1[0]
        self.macs += 3 * (self.W1.size + self.W2.size)

    def learn(self, st, label, scale=1.0):
        self._step(st["x"], label)
        if self.replay:                      # reservoir buffer, one replayed example per step
            self.seen += 1
            if len(self.buf) < self.replay:
                self.buf.append((st["x"], label))
            elif (j := self.rng.integers(self.seen)) < self.replay:
                self.buf[j] = (st["x"], label)
            xb, yb = self.buf[self.rng.integers(len(self.buf))]
            self._step(xb, yb)

    def test(self, times, y):
        a1 = np.maximum(self.feats(times) @ self.W1 + self.b1, 0)
        return (a1 @ self.W2 + self.b2).argmax(1)


# ── Living in the stream ──────────────────────────────────────────────────────

def overlap(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def run(kind, cfg, mech, sc, x, y, xe, ye, replay=0, window=1000):
    rng = np.random.default_rng(cfg.seed)
    frames, labels, reset, task = make_stream(sc, x, y)
    times = latency_code(frames)
    te_times = latency_code(xe)
    if kind == "race":
        drive = np.where(np.isfinite(times), 1.0 - times, 0).mean(0)
        net = StreamRace(cfg, mech, 784, 10, float(drive.sum()), rng, drive)
    else:
        net = StreamMLP(cfg, 784, 10, replay, rng)
    labeller = Labeller(sc, len(labels))
    pending = {}                                  # arrival frame -> [(frame, label, state, episode)]
    episode, ep_label, prev_fired = -1, None, None
    rec = {k: np.zeros(len(labels)) for k in ("correct", "used", "taught", "asked")}
    first = reset.copy()
    ov_within, ov_across, last_code = [], [], None
    task_acc = []
    t0 = time.time()
    for i in range(len(labels)):
        if reset[i]:
            episode += 1
            ep_label, prev_fired = None, None
        st = net.predict(times[i])
        rec["correct"][i] = st["winner"][0] == labels[i]
        rec["used"][i] = st["used"]
        if kind == "race" and net.h:
            code = st["fired"][0]
            if last_code is not None:
                (ov_across if reset[i] else ov_within).append(overlap(code, last_code))
            last_code = code
        if labeller(i, st["uncertain"]):
            rec["asked"][i] = 1
            pending.setdefault(i + sc.delay, []).append((i, labels[i], st, episode))
        taught = False
        for j, lab, st_j, ep_j in pending.pop(i, []):
            if mech.tags or kind != "race":
                net.learn(st_j, lab, np.exp(-(i - j) / mech.tau_tag) if kind == "race" else 1.0)
            else:                                 # no tags: the late label lands on the current frame
                net.learn(st, lab)
            taught |= j == i or not mech.tags
            if mech.carry and ep_j == episode:
                ep_label = lab
        if not taught and mech.carry and ep_label is not None:
            net.learn(st, ep_label)
            taught = True
        if kind == "race":
            if not taught and mech.continuity and prev_fired is not None:
                net.pull(st, prev_fired)
            if net.h:
                prev_fired = st["fired"][0][None]
            net.after(st, i + 1 < len(labels) and reset[i + 1])
        rec["taught"][i] = taught
        if sc.order == "blocked" and (i + 1 == len(labels) or task[i + 1] != task[i]):
            task_acc.append(per_task_acc(kind, net, te_times[:2000], ye[:2000]))
    wall = time.time() - t0
    n = len(labels)
    wins = [slice(a, min(a + window, n)) for a in range(0, n, window)]
    later = ~first
    res = {
        "kind": kind, "config": asdict(cfg), "mech": asdict(mech), "stream": asdict(sc), "replay": replay,
        "frames": n, "labels_used": int(labeller.used),
        "prequential_acc": float(rec["correct"].mean()),
        "curve_acc": [float(rec["correct"][w].mean()) for w in wins],
        "curve_used": [float(rec["used"][w].mean()) for w in wins],
        "acc_first_view": float(rec["correct"][first].mean()),
        "acc_later_views": float(rec["correct"][later].mean()) if later.any() else None,
        "used_first_view": float(rec["used"][first].mean()),
        "used_later_views": float(rec["used"][later].mean()) if later.any() else None,
        "late_acc": float(rec["correct"][-n // 5:].mean()),
        "late_used": float(rec["used"][-n // 5:].mean()),
        "taught_frames": int(rec["taught"].sum()),
        "held_out_acc": held_out(kind, net, te_times, ye),
        "wall_s": round(wall, 1),
    }
    if kind == "race":
        res["work_per_frame"] = {k: v / n for k, v in net.work.items() if k != "samples"}
        if ov_within or ov_across:
            res["code_overlap"] = {"within_episode": float(np.mean(ov_within)) if ov_within else None,
                                   "across_episodes": float(np.mean(ov_across)) if ov_across else None}
    else:
        res["work_per_frame"] = {"macs": net.macs / n}
    if task_acc:
        A = np.array(task_acc)                    # row: after block r; column: task accuracy
        res["task_acc"] = A.tolist()
        res["forgetting"] = float(np.mean([A[:, k].max() - A[-1, k] for k in range(len(TASKS) - 1)]))
        res["final_task_mean"] = float(A[-1].mean())
    return res


def held_out(kind, net, te_times, ye):
    if kind == "race":
        saved = dict(net.work)
        acc = evaluate(net, te_times, ye)
        net.work = saved
        return acc
    return float((net.test(te_times, ye) == ye).mean())


def per_task_acc(kind, net, te_times, ye):
    if kind == "race":
        saved = dict(net.work)
        pred = np.concatenate([net.forward(*to_events(te_times[i:i + 250]))["winner"]
                               for i in range(0, len(ye), 250)])
        net.work = saved
    else:
        pred = net.test(te_times, ye)
    return [float((pred[np.isin(ye, c)] == ye[np.isin(ye, c)]).mean()) for c in TASKS]


def load(n_val):
    x, y = mnist("train")
    if n_val:                          # tuning: the last n_val training images are the held-out set
        return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]
    xt, yt = mnist("test")
    return x, y, xt, yt


# E6 round-3 configuration, per frame. Homeostasis is per update, so with one frame
# per update it is divided by E6's batch of 32 to keep the same rate per sample.
R3 = dict(winners=3, hid_frac=0.6, eta_out=0.01, eta_hid=0.01, deadline=1, psp="ramp",
          homeo=0.001 / 32, batch=1, epochs=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("race", "mlp"))
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--replay", type=int, default=0)
    groups = ((Config, R3), (Mech, {}), (Stream, {}))
    for cls, over in groups:
        for f, v in asdict(cls()).items():
            if cls is Stream and f == "seed":
                continue
            ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=over.get(f, v))
    a = ap.parse_args()
    cfg = Config(**{f: getattr(a, f) for f in asdict(Config())})
    mech = Mech(**{f: getattr(a, f) for f in asdict(Mech())})
    sc = Stream(**{f: getattr(a, f) for f in asdict(Stream()) if f != "seed"}, seed=cfg.seed)
    x, y, xe, ye = load(a.val)
    res = run(a.kind, cfg, mech, sc, x, y, xe, ye, a.replay)
    res["val"] = a.val
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.kind}_{a.tag or 'run'}_s{cfg.seed}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("prequential_acc", "late_acc", "held_out_acc", "labels_used",
                                          "used_first_view", "used_later_views", "wall_s")}, indent=1),
          flush=True)
