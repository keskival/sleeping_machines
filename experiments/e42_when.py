"""E42: learning when to transact on the BTCUSDT trade stream (E42_PREREGISTRATION.md).

Stage 1 (this file's `prep` and `baselines`): decision points, the cost-aware hindsight teacher, inputs, and the simple
baselines, with profit after costs. The event learner and B-logit run prequentially over the pilot days first and, once
frozen and committed, once over the confirmatory days.

Decision points: trades whose price differs by >= 0.5 bp from the previous decision point's price, at most one per second.
Profit between decision points k and k+1: x_k * 1e4 * ln(P_{k+1} / P_k) bp; cost c * |x_k - x_{k-1}| at each change.
Teacher: over each 10-minute lookahead, the profit-maximizing position sequence given c (dynamic programming on {-1,0,1}).
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import load_day, size_edges, window_features, PILOT, CONF  # noqa: E402
from e35_free import Free  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e42")
STEP_BP, MIN_GAP_US, LOOK_US, WIN_US = 0.5, 1_000_000, 600_000_000, 10_000_000
POS = np.array([-1, 0, 1])


def decision_points(t, p):
    """indices of trades that are decision points (price moved >= STEP_BP since the last one, >= 1 s apart)."""
    idx, last_p, last_t = [0], p[0], t[0]
    lp = np.log(p)
    # vectorized scan in chunks: candidates are trades far enough in price from the running reference
    for i in range(1, len(t)):
        if t[i] - last_t >= MIN_GAP_US and abs(lp[i] - np.log(last_p)) * 1e4 >= STEP_BP:
            idx.append(i); last_p, last_t = p[i], t[i]
    return np.array(idx)


def teacher(r, c, look):
    """r[k]: log-return (bp) from decision k to k+1. For each k, the position that the profit-maximizing sequence over
    decisions k .. k+look-1 (with cost c per unit change, starting from any position) takes at k, via backward DP."""
    n = len(r)
    lab = np.zeros(n, int)
    for k0 in range(0, n, max(1, look // 4)):                 # rolling windows; labels taken from each window's start
        k1 = min(n, k0 + look)
        V = np.zeros(3)                                        # value-to-go from the window end
        choice = np.zeros((k1 - k0, 3), int)                   # best position at step k given previous position
        for k in range(k1 - 1, k0 - 1, -1):
            gain = POS * r[k]                                  # holding position x over step k
            tot = gain[None, :] - c * np.abs(POS[:, None] - POS[None, :]) + V[None, :]   # prev x (row) -> new x (col)
            choice[k - k0] = tot.argmax(1)
            V = tot.max(1)
        prev = int(lab[k0 - 1]) + 1 if k0 > 0 else 1           # continue from the position actually held
        for k in range(k0, min(k1, k0 + max(1, look // 4))):
            nx = choice[k - k0, prev]; lab[k] = POS[nx]; prev = nx
    return lab


def pnl(pos, r, c):
    """net bp: positions pos[k] held over r[k], cost on changes (starting flat)."""
    change = np.abs(np.diff(np.r_[0, pos]))
    return float((pos * r).sum() - c * change.sum()), int((change > 0).sum())


def prep_day(day):
    t, p, q, sell = load_day(day)
    dp = decision_points(t, p)
    P = p[dp]
    r = np.r_[1e4 * np.diff(np.log(P)), 0.0]
    return dict(t=t[dp], P=P, r=r, idx=dp)


def momentum(t, P, h, lookback_us=60_000_000):
    """position: +1 if 60 s return > h, -1 if < -h, 0 if |return| < h/2, else keep."""
    j = np.searchsorted(t, t - lookback_us)
    ret = 1e4 * np.log(P / P[j])
    pos = np.zeros(len(P), int); x = 0
    for k in range(len(P)):
        if ret[k] > h: x = 1
        elif ret[k] < -h: x = -1
        elif abs(ret[k]) < h / 2: x = 0
        pos[k] = x
    return pos


def inputs(day, D, edges):
    """spike times (seconds before... expressed as time within the 10 s window, 0..10) of the first trade of each of the
    24 E17 event types in the window ending at each decision point; inf if absent."""
    t, p, q, sell = load_day(day)
    tick = np.sign(np.diff(p, prepend=p[0])).astype(np.int64) + 1
    typ = sell.astype(np.int64) * 12 + np.searchsorted(edges, q).astype(np.int64) * 3 + tick
    X = np.full((len(D["t"]), 24), np.inf)
    lo = np.searchsorted(t, D["t"] - WIN_US); hi = D["idx"] + 1
    for k in range(len(D["t"])):
        tt, yy = t[lo[k]:hi[k]], typ[lo[k]:hi[k]]
        if len(tt):
            u, first = np.unique(yy, return_index=True)
            X[k, u] = (tt[first] - (D["t"][k] - WIN_US)) / 1e6
    F = window_features(D["t"] - WIN_US, (t, p, q, sell))
    return X, F


def run_learners(days, c, edges, eta_ev=0.1, eta_lg=0.05, seed=0, freeze=None, m=1.0):
    """prequential over `days` in order: act at each decision with current models; learn from label k once t >= t_k + L."""
    rng = np.random.default_rng(seed)
    ev = freeze["ev"] if freeze else Free(27, 3, rng)
    Wl = freeze["Wl"] if freeze else np.zeros((3, 9))
    out = []
    for day in days:
        D = prep_day(day)
        X, F = inputs(day, D, edges)
        n = len(D["r"])
        look = max(1, int(LOOK_US / max(np.median(np.diff(D["t"])), 1)))
        lab = teacher(D["r"], c * m, look)                      # teacher charges m × c (uncertainty margin)
        pos = {"event": np.zeros(n, int), "logit": np.zeros(n, int)}
        xe = xl = 0
        pend, ops = [], 0
        for k in range(n):
            # release labels whose lookahead has passed (teacher target and the inputs seen then)
            while pend and D["t"][pend[0][0]] + LOOK_US <= D["t"][k]:
                j, xin, fin, prev_e, prev_l = pend.pop(0)
                if freeze is None:
                    tgt = lab[j] + 1
                    ev.teach(xin, tgt if lab[j] != prev_e else 3, eta_ev)
                    z = Wl @ fin; pz = np.exp(z - z.max()); pz /= pz.sum(); pz[tgt] -= 1
                    Wl -= eta_lg * np.outer(pz, fin)
            pe = np.full(3, np.inf); pe[xe + 1] = 0.0                    # current position as spike channels
            xin = np.r_[X[k], pe]
            c_ev, ft, _ = ev.forward(xin)
            ops += ev.synaptic_events(xin, ft)
            if c_ev < 3:
                xe = int(POS[c_ev])
            fin = np.r_[np.nan_to_num(F[k]), np.eye(3)[xl + 1], 1.0]
            xl = int(POS[int(np.argmax(Wl @ fin))]) if np.abs(Wl).sum() else xl
            pos["event"][k], pos["logit"][k] = xe, xl
            pend.append((k, xin, fin, xe, xl))
        row = {"day": str(day), "c": c, "m": m, "decisions": n, "teacher": pnl(lab, D["r"], c), "flat": [0.0, 0],
               "hold": pnl(np.ones(n, int), D["r"], c), "event": pnl(pos["event"], D["r"], c),
               "logit": pnl(pos["logit"], D["r"], c), "event_ops_per_decision": ops / n}
        out.append(row)
        print(json.dumps(row), flush=True)
    return out, {"ev": ev, "Wl": Wl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="baselines", choices=("baselines", "learners"))
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--m", type=float, default=1.0, help="teacher cost multiplier (tuned on pilot days)")
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--days", default="pilot", choices=("pilot",))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    days = PILOT
    if a.stage == "learners":
        out, _ = run_learners(days[:a.ndays], a.c, size_edges(), eta_ev=a.eta, m=a.m)
        with open(os.path.join(OUT, f"pilot_learners_c{a.c:g}_eta{a.eta:g}_m{a.m:g}_d{a.ndays}.json"), "w") as f:
            json.dump(out, f, indent=1)
        return
    res = {"days": [str(d) for d in days], "per_day": []}
    for day in days:
        D = prep_day(day)
        n = len(D["r"])
        look = int(np.median(np.diff(D["t"]))) and max(1, int(LOOK_US / max(np.median(np.diff(D["t"])), 1)))
        row = {"day": str(day), "decisions": n}
        for c in (2.0, 10.0):
            lab = teacher(D["r"], c, look)
            row[f"teacher_c{c:g}"] = pnl(lab, D["r"], c)
            row[f"hold_c{c:g}"] = pnl(np.ones(n, int), D["r"], c)
            for h in (2, 5, 10, 20, 40):
                row[f"mom{h}_c{c:g}"] = pnl(momentum(D["t"], D["P"], h), D["r"], c)
        res["per_day"].append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(OUT, "pilot_baselines.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
