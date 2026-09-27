"""E55: an edge audit for the market stream. Is 'stay out' the right policy, and do more trading pairs create an edge?

At each BTC spot decision point (E42's price events) the executable round trip over a horizon H is measured from the
trade tape itself, so bid-ask bounce cannot masquerade as predictability: a long buys at the last buyer-initiated
price (the ask) and sells at the last seller-initiated price H seconds later (the bid); a short the reverse. Fees are
added on top (round trip, bp).
Information sets (discrete states, the kind of state an event world model carries):
  own   BTC spot: directions of its last two price moves, the time since the last move (5 buckets), aggressor side now
  perp  BTC USD-M perpetual: direction of its last price move, its age (5 buckets), whether spot has moved since
  eth   ETH spot: the same as perp
A state's mean executable return (long and short) is fitted on pilot days 1-5 and a policy trades in a state when the
fitted mean exceeds the fee. Reported: opportunities per day and the realized mean net return per trade on pilot days
6-7 (held out), and the in-sample upper bound (fitted and scored on days 1-5). Confirmatory days are not read here.
"""
import argparse
import io
import itertools
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e17_market import PILOT  # noqa: E402
from e42_when import decision_points  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "binance")
OUT = os.path.join(os.path.dirname(__file__), "results", "e55")
SRC = {"btc": ("", "BTCUSDT"), "perp": ("btc_perp", "BTCUSDT"), "eth": ("eth_spot", "ETHUSDT")}
AGE = np.array([0.05, 0.25, 1.0, 5.0])                     # seconds


def load(sym, day):
    sub, name = SRC[sym]
    cache = os.path.join(DATA, "npz", f"{sym}_{day}.npz")
    if os.path.exists(cache):
        z = np.load(cache); return z["t"], z["p"], z["sell"]
    with zipfile.ZipFile(os.path.join(DATA, sub, f"{name}-aggTrades-{day}.zip")) as zf:
        raw = zf.read(zf.namelist()[0])
    if raw[:3] == b"agg":
        raw = raw[raw.index(b"\n") + 1:]
    raw = raw.replace(b"True", b"1").replace(b"False", b"0").replace(b"true", b"1").replace(b"false", b"0")
    a = np.loadtxt(io.BytesIO(raw), delimiter=",", usecols=(1, 5, 6), dtype=np.float64)
    t = a[:, 1].astype(np.int64)
    t = np.where(t < 10 ** 14, t * 1000, t)                 # ms (futures) -> µs
    p, sell = a[:, 0], a[:, 2] > 0.5                        # buyer is maker: a seller-initiated trade (at the bid)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, t=t, p=p, sell=sell)
    return t, p, sell


def last_move(t, p, at):
    """for query times `at`: direction (+1/-1) of the stream's last price change before `at`, and its time."""
    ch = np.flatnonzero(np.diff(p) != 0) + 1
    tc, dc = t[ch], np.sign(p[ch] - p[ch - 1])
    k = np.searchsorted(tc, at, side="right") - 1
    ok = k >= 0
    return np.where(ok, dc[np.maximum(k, 0)], 0), np.where(ok, tc[np.maximum(k, 0)], -10 ** 18)


def features(day, H):
    t, p, sell = load("btc", day)
    dp = decision_points(t, p)[2:]
    # executable quotes as of each trade: last buyer-initiated price = ask, last seller-initiated = bid
    idx = np.arange(len(t))
    ask_i = np.maximum.accumulate(np.where(~sell, idx, -1)); bid_i = np.maximum.accumulate(np.where(sell, idx, -1))
    ok = (ask_i[dp] >= 0) & (bid_i[dp] >= 0); dp = dp[ok]
    td = t[dp]
    out = {"n": len(dp)}
    for h in H:
        j = np.searchsorted(t, td + int(h * 1e6), side="right") - 1
        valid = (j > dp) & (ask_i[j] >= 0) & (bid_i[j] >= 0)
        a0, b0 = p[ask_i[dp]], p[bid_i[dp]]; a1, b1 = p[ask_i[j]], p[bid_i[j]]
        out[f"long{h}"] = np.where(valid, (b1 - a0) / a0 * 1e4, np.nan)
        out[f"short{h}"] = np.where(valid, (b0 - a1) / b0 * 1e4, np.nan)
    # own state: last two move directions, time since last move, aggressor side now
    d_all = np.sign(np.diff(p[dp], prepend=p[dp[0]]))
    d1 = np.r_[0, d_all[:-1]] > 0; d2 = np.r_[0, 0, d_all[:-2]] > 0
    gap = np.diff(td, prepend=td[0]) / 1e6
    own = (d1 * 2 + d2) * 10 + np.searchsorted(AGE, gap) * 2 + sell[dp]
    out["own"] = own.astype(np.int64)
    for sym in ("perp", "eth"):
        ts, ps, _ = load(sym, day)
        dr, tm = last_move(ts, ps, td)
        spot_dir, spot_tm = last_move(t, p, td - 1)          # spot's last move strictly before this one
        moved_since = spot_tm > tm
        out[sym] = ((dr > 0) * 20 + np.searchsorted(AGE, (td - tm) / 1e6) * 2 + moved_since).astype(np.int64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", default="1,5,30,120")
    ap.add_argument("--fees", default="0,2,5,10")
    ap.add_argument("--min_count", type=int, default=300)
    a = ap.parse_args()
    H = [float(x) for x in a.H.split(",")]; fees = [float(x) for x in a.fees.split(",")]
    os.makedirs(OUT, exist_ok=True)
    days = [features(d, H) for d in PILOT]
    tr, te = days[:5], days[5:]
    sets = {"own": ["own"], "own+perp": ["own", "perp"], "own+eth": ["own", "eth"], "own+perp+eth": ["own", "perp", "eth"],
            "perp": ["perp"]}
    rows = []
    for name, keys in sets.items():
        def key(dd):
            k = np.zeros(dd["n"], np.int64)
            for kk in keys:
                k = k * 64 + dd[kk]
            return k
        ktr = np.concatenate([key(d) for d in tr]); kte = np.concatenate([key(d) for d in te])
        for h in H:
            for side in ("long", "short"):
                ytr = np.concatenate([d[f"{side}{h}"] for d in tr])
                yte = np.concatenate([d[f"{side}{h}"] for d in te])
                m = np.isfinite(ytr); u, inv = np.unique(ktr[m], return_inverse=True)
                cnt = np.bincount(inv); mean = np.bincount(inv, ytr[m]) / cnt
                for fee in fees:
                    good = set(u[(cnt >= a.min_count) & (mean > fee)].tolist())
                    sel_tr = np.isin(ktr, list(good)) & np.isfinite(ytr)
                    sel_te = np.isin(kte, list(good)) & np.isfinite(yte)
                    rows.append({"info": name, "H": h, "side": side, "fee": fee, "states": len(good),
                                 "in_sample_trades_per_day": float(sel_tr.sum() / 5),
                                 "in_sample_net_bp": float(np.mean(ytr[sel_tr]) - fee) if sel_tr.any() else None,
                                 "heldout_trades_per_day": float(sel_te.sum() / 2),
                                 "heldout_net_bp": float(np.mean(yte[sel_te]) - fee) if sel_te.any() else None})
    for r in rows:
        if r["fee"] in (0, 2) and r["heldout_trades_per_day"] > 0:
            print(json.dumps(r), flush=True)
    uncond = {f"{s}{h}": float(np.nanmean(np.concatenate([d[f"{s}{h}"] for d in tr]))) for s in ("long", "short") for h in H}
    with open(os.path.join(OUT, "edge_pilot.json"), "w") as f:
        json.dump({"args": vars(a), "unconditional_mean_bp_days1_5": uncond, "rows": rows}, f, indent=1)
    print("unconditional", json.dumps(uncond)); print("EXIT-OK")


if __name__ == "__main__":
    main()
