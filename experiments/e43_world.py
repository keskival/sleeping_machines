"""E43: an online world model of the trade stream, with learning to learn (surprise-gated plasticity).

World model: predict the direction of the next price event (the next >= 0.5 bp move, E42's decision points) from the
trade-event spikes of the preceding 10 s (E17/E42 encoding, 24 channels). Every price event is a label for the
prediction made at the previous one: dense, immediate, self-supervised, learned online through the whole stream.
  event  E35 detectors (nothing given) for "up next" and "down next"; they may abstain (neither fires).
  logit  online logistic regression on E17's window features (same labels).
Learning to learn (--meta 1): each learner's step is scaled by its surprise ratio, the short-term error rate over the
long-term one (EMAs over ~50 and ~2000 events), clipped to [1/4, 4]: fast plasticity after a change, slow when reliable
(THEORY §45's tracking rule, eta* ∝ sqrt(process / observation noise), estimated locally).
Decisions (--trade 1): up/down predictions feed E42's evidence accumulators and profit-learned prices.
Metrics per day: accuracy when predicting, coverage, and (with --trade) net bp after costs.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import size_edges, PILOT, CONF  # noqa: E402
from e35_free import Free  # noqa: E402
from e42_when import prep_day, inputs, pnl, POS, LOOK_US  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e43")


class Surprise:
    def __init__(self, fast=0.02, slow=0.0005):
        self.f, self.s, self.ef, self.es = fast, slow, 0.5, 0.5

    def update(self, err):
        self.ef += self.f * (err - self.ef); self.es += self.s * (err - self.es)

    def ratio(self):
        return float(np.clip(self.ef / max(self.es, 1e-3), 0.25, 4.0))


def run(days, edges, meta, trade, c, eta_ev, eta_lg, seed=0, pop=1):
    rng = np.random.default_rng(seed)
    evs = [Free(24, 2, np.random.default_rng(seed * 100 + i)) for i in range(pop)]   # a population of detector pairs
    Wl = np.zeros(6)
    s_ev, s_lg = Surprise(), Surprise()
    price = np.full(3, 2.0); gamma, kappa = 0.8, 0.5
    out = []
    for day in days:
        D = prep_day(day)
        X, F = inputs(day, D, edges)
        n = len(D["r"])
        up = (D["r"] > 0).astype(int)                         # label for decision k: is the next move up?
        st = {"ev_pred": 0, "ev_right": 0, "lg_right": 0}
        acc = np.zeros(3); xp = 0; pos = np.zeros(n, int); trades = []
        lp = np.log(D["P"])
        for k in range(n - 1):
            x = X[k]
            votes = np.zeros(2)
            for e_ in evs:                                       # each detector pair votes; the class with more
                ci, _, _ = e_.forward(x)                         # votes wins (an accumulator race over the population)
                if ci < 2: votes[ci] += 1
            c_ev = 2 if votes.sum() == 0 or votes[0] == votes[1] else int(votes.argmax())
            f = np.r_[np.nan_to_num(F[k]), 1.0]
            pl = 1 / (1 + np.exp(-Wl @ f))
            y = up[k]
            if c_ev < 2:
                st["ev_pred"] += 1; st["ev_right"] += int((c_ev == 0) == bool(y))
            st["lg_right"] += int((pl > 0.5) == bool(y))
            # learn immediately: the label is the next price event (self-supervised world model)
            err_ev = float(c_ev < 2 and (c_ev == 0) != bool(y))
            s_ev.update(err_ev); s_lg.update(float((pl > 0.5) != bool(y)))
            for e_ in evs:
                e_.teach(x, 0 if y else 1, eta_ev * (s_ev.ratio() if meta else 1.0))
            Wl -= eta_lg * (s_lg.ratio() if meta else 1.0) * (pl - y) * f
            if trade:                                             # decisions from the world model (E42 priced)
                while trades and D["t"][trades[0][0]] + LOOK_US <= D["t"][k]:
                    j, tgt, frm = trades.pop(0)
                    jj = min(np.searchsorted(D["t"], D["t"][j] + LOOK_US), n - 1)
                    gain = (tgt - frm) * 1e4 * (lp[jj] - lp[j]) - c * abs(tgt - frm)
                    price[tgt + 1] = max(1.0, price[tgt + 1] + (kappa if gain < 0 else -kappa))
                acc *= gamma
                if c_ev == 0: acc[2] += 1.0; acc[1] += 0.5
                if c_ev == 1: acc[0] += 1.0; acc[1] += 0.5
                cand = [i for i in range(3) if POS[i] != xp and acc[i] > price[i]]
                if cand:
                    i = max(cand, key=lambda i: acc[i] - price[i])
                    trades.append((k, int(POS[i]), xp)); xp = int(POS[i]); acc[:] = 0.0
                pos[k] = xp
        row = {"day": str(day), "decisions": n, "event_coverage": st["ev_pred"] / (n - 1),
               "event_acc_when_predicting": st["ev_right"] / max(st["ev_pred"], 1),
               "logit_acc": st["lg_right"] / (n - 1), "surprise_ratio_end": [s_ev.ratio(), s_lg.ratio()]}
        if trade:
            row.update({"net_bp": pnl(pos, D["r"], c), "prices": [round(float(v), 2) for v in price]})
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=int, default=1)
    ap.add_argument("--trade", type=int, default=0)
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--eta-lg", type=float, default=0.01)
    ap.add_argument("--ndays", type=int, default=7)
    ap.add_argument("--pop", type=int, default=1, help="detector pairs voting (population)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = run(PILOT[:a.ndays], size_edges(), a.meta, a.trade, a.c, a.eta, a.eta_lg, pop=a.pop)
    with open(os.path.join(OUT, f"pilot_meta{a.meta}_trade{a.trade}_c{a.c:g}_pop{a.pop}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
