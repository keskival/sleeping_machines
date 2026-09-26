"""E17 analysis: the preregistered decision rules (E17_PREREGISTRATION.md), from per-episode files.

Day-block bootstrap (resampling the 21 confirmatory days) gives 95% intervals for every contrast.
"""
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(__file__), "results", "e17")
FLAT_BP, COST = 1.0, 2.0
B = 2000


def load(name):
    p = os.path.join(RES, f"{name}_d1_w200_s0_episodes.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def per_day(e, stat):
    return np.array([stat({k: v[e["day"] == d] for k, v in e.items()}) for d in range(21)])


def acc_parts(x):
    nf = x["decided"] & (np.abs(x["ret"]) >= FLAT_BP)
    sgn = np.where(x["pred"] == 0, 1.0, -1.0)
    return np.array([((sgn * x["ret"] > 0) & nf).sum(), nf.sum()], float)


def profit_parts(x):
    sgn = np.where(x["pred"] == 0, 1.0, -1.0)
    return np.array([np.where(x["decided"], sgn * x["ret"] - COST, 0.0).sum(), len(x["ret"])], float)


def tdec_parts(x):
    return np.array([x["tdec"][x["decided"]].sum() / 1e6, x["decided"].sum()], float)


def ratio_ci(parts_a, parts_b=None, rng=np.random.default_rng(0)):
    """Point estimate and 95% day-block interval of ratio(a) or ratio(a) − ratio(b)."""
    def val(idx):
        ra = parts_a[idx, 0].sum() / max(parts_a[idx, 1].sum(), 1)
        if parts_b is None:
            return ra
        return ra - parts_b[idx, 0].sum() / max(parts_b[idx, 1].sum(), 1)
    point = val(np.arange(21))
    boots = [val(rng.integers(0, 21, 21)) for _ in range(B)]
    return float(point), [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def top_fraction(e, frac):
    """B1 restricted to its most confident episodes, each day, at a given trade fraction."""
    out = {k: v.copy() for k, v in e.items()}
    dec = np.zeros(len(e["conf"]), bool)
    for d in range(21):
        m = e["day"] == d
        c = e["conf"][m]
        thr = np.quantile(c, 1 - frac) if frac < 1 else -np.inf
        dec[np.flatnonzero(m)[c >= thr]] = True
    out["decided"] = dec
    return out


def main():
    E = {k: load(k) for k in ("race", "race_frozen", "race_hold", "b1", "b0")}
    rep = {}
    for k, e in E.items():
        if e is None:
            continue
        rep[k] = {"acc": ratio_ci(per_day(e, acc_parts)), "profit_bp_per_episode": ratio_ci(per_day(e, profit_parts)),
                  "coverage": float(e["decided"].mean()),
                  "tdec_s": ratio_ci(per_day(e, tdec_parts))[0]}
    if E["race"] is not None and E["b1"] is not None:
        rep["race_minus_b1_acc"] = ratio_ci(per_day(E["race"], acc_parts), per_day(E["b1"], acc_parts))
        cov = float(E["race"]["decided"].mean())               # fairness: B1 at the race's coverage
        rep["race_minus_b1_same_coverage_acc"] = ratio_ci(per_day(E["race"], acc_parts),
                                                          per_day(top_fraction(E["b1"], cov), acc_parts))
        if E["b0"] is not None:
            rep["race_minus_b0_acc"] = ratio_ci(per_day(E["race"], acc_parts), per_day(E["b0"], acc_parts))
        rep["race_earlier_frac"] = 1 - rep["race"]["tdec_s"] / 10.0
        d, (lo, hi) = rep["race_minus_b1_acc"]
        rep["verdict_competitive"] = bool(d >= -0.01 and rep["race_earlier_frac"] >= 0.2)
        rep["verdict_better"] = bool(d >= 0.01 and lo > 0)
    if E["race"] is not None and E["race_frozen"] is not None:
        rep["continual_minus_frozen_acc"] = ratio_ci(per_day(E["race"], acc_parts),
                                                     per_day(E["race_frozen"], acc_parts))
        d, (lo, hi) = rep["continual_minus_frozen_acc"]
        rep["verdict_continual_helps"] = bool(d >= 0.01 and lo > 0)
    if E["race_hold"] is not None and E["b1"] is not None and E["race"] is not None:
        frac = float(E["race_hold"]["decided"].mean())
        b1_top = top_fraction(E["b1"], frac)
        rep["hold_trade_fraction"] = frac
        rep["b1_top_same_fraction"] = {"acc": ratio_ci(per_day(b1_top, acc_parts)),
                                       "profit_bp_per_episode": ratio_ci(per_day(b1_top, profit_parts))}
        ph = per_day(E["race_hold"], profit_parts)
        rep["hold_minus_race_profit"] = ratio_ci(ph, per_day(E["race"], profit_parts))
        rep["hold_minus_b1top_profit"] = ratio_ci(ph, per_day(b1_top, profit_parts))
        rep["verdict_trade_selection_helps"] = bool(rep["hold_minus_race_profit"][1][0] > 0
                                                    and rep["hold_minus_b1top_profit"][1][0] > 0)
    with open(os.path.join(RES, "analysis.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
