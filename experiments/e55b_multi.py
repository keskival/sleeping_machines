"""E55b: the edge audit across markets. Each of BTC spot, ETH spot, SOL spot and the BTC perpetual is the traded target in
turn, with the other three as leaders (lead-lag states). Executable round trips from the target's own tape (buy at the
last buyer-initiated price, sell at the last seller-initiated price H later, or the reverse), fees on top.
Information sets: the target's own event state; own + each leader; own + all leaders.
Protocol (§87): state means fitted on the 7 pilot days, then (a) a pilot-internal check (fit days 1-5, score 6-7) and
(b) the 21 confirmatory days, scored once. Fee levels (round trip, bp): 0, 2 (maker-like), 5, 9 (perp taker ≈ 4.5 per
side), 15 (spot taker ≈ 7.5 per side).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e55_edge as E  # noqa: E402
from e17_market import PILOT, CONF  # noqa: E402
from e42_when import decision_points  # noqa: E402

E.SRC["sol"] = ("sol_spot", "SOLUSDT")
OUT = os.path.join(os.path.dirname(__file__), "results", "e55")
MARKETS = ("btc", "eth", "sol", "perp")
AGE = E.AGE


def features(day, H, target):
    t, p, sell = E.load(target, day)
    dp = decision_points(t, p)[2:]
    idx = np.arange(len(t))
    ask_i = np.maximum.accumulate(np.where(~sell, idx, -1)); bid_i = np.maximum.accumulate(np.where(sell, idx, -1))
    dp = dp[(ask_i[dp] >= 0) & (bid_i[dp] >= 0)]; td = t[dp]
    out = {"n": len(dp)}
    for h in H:
        j = np.searchsorted(t, td + int(h * 1e6), side="right") - 1
        valid = (j > dp) & (ask_i[j] >= 0) & (bid_i[j] >= 0)
        a0, b0 = p[ask_i[dp]], p[bid_i[dp]]; a1, b1 = p[ask_i[j]], p[bid_i[j]]
        out[f"long{h}"] = np.where(valid, (b1 - a0) / a0 * 1e4, np.nan)
        out[f"short{h}"] = np.where(valid, (b0 - a1) / b0 * 1e4, np.nan)
    d_all = np.sign(np.diff(p[dp], prepend=p[dp[0]]))
    d1 = np.r_[0, d_all[:-1]] > 0; d2 = np.r_[0, 0, d_all[:-2]] > 0
    gap = np.diff(td, prepend=td[0]) / 1e6
    out["own"] = ((d1 * 2 + d2) * 10 + np.searchsorted(AGE, gap) * 2 + sell[dp]).astype(np.int64)
    _, tm_own = E.last_move(t, p, td - 1)
    for sym in MARKETS:
        if sym == target:
            continue
        ts, ps, _ = E.load(sym, day)
        dr, tm = E.last_move(ts, ps, td)
        out[sym] = ((dr > 0) * 20 + np.searchsorted(AGE, (td - tm) / 1e6) * 2 + (tm_own > tm)).astype(np.int64)
    return out


def evaluate(tr, te, keys, H, fees, min_count):
    def key(dd):
        k = np.zeros(dd["n"], np.int64)
        for kk in keys:
            k = k * 64 + dd[kk]
        return k
    ktr = np.concatenate([key(d) for d in tr]); kte = np.concatenate([key(d) for d in te])
    rows = []
    for h in H:
        for side in ("long", "short"):
            ytr = np.concatenate([d[f"{side}{h}"] for d in tr]); yte = np.concatenate([d[f"{side}{h}"] for d in te])
            m = np.isfinite(ytr); u, inv = np.unique(ktr[m], return_inverse=True)
            cnt = np.bincount(inv); mean = np.bincount(inv, ytr[m]) / cnt
            for fee in fees:
                good = u[(cnt >= min_count) & (mean > fee)]
                sel = np.isin(kte, good) & np.isfinite(yte)
                rows.append({"H": h, "side": side, "fee": fee, "states": int(len(good)),
                             "trades_per_day": float(sel.sum() / len(te)),
                             "net_bp": float(np.mean(yte[sel]) - fee) if sel.any() else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", default="1,5,30,120,300")
    ap.add_argument("--fees", default="0,2,5,9,15")
    ap.add_argument("--min_count", type=int, default=300)
    ap.add_argument("--targets", default="btc,eth,sol,perp")
    a = ap.parse_args()
    H = [float(x) for x in a.H.split(",")]; fees = [float(x) for x in a.fees.split(",")]
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for target in a.targets.split(","):
        pilot = [features(d, H, target) for d in PILOT]; conf = [features(d, H, target) for d in CONF]
        leaders = [m for m in MARKETS if m != target]
        sets = {"own": ["own"], **{f"own+{m}": ["own", m] for m in leaders}, "own+all": ["own"] + leaders}
        res[target] = {}
        for name, keys in sets.items():
            res[target][name] = {"pilot": evaluate(pilot[:5], pilot[5:], keys, H, fees, a.min_count),
                                 "confirm": evaluate(pilot, conf, keys, H, fees, a.min_count)}
            best = {}
            for fee in fees:
                c = [r for r in res[target][name]["confirm"] if r["fee"] == fee and r["net_bp"] is not None]
                best[fee] = max(c, key=lambda r: r["net_bp"] * r["trades_per_day"]) if c else None
            print(json.dumps({"target": target, "info": name, "best_by_fee": {
                f"{k:g}": (None if v is None else {"H": v["H"], "side": v["side"], "net_bp": round(v["net_bp"], 2),
                                                    "trades_per_day": round(v["trades_per_day"], 1)}) for k, v in best.items()}}),
                  flush=True)
    with open(os.path.join(OUT, "edge_multi.json"), "w") as f:
        json.dump({"args": vars(a), "results": res}, f)
    print("EXIT-OK")


if __name__ == "__main__":
    main()
