"""E17: a continually learning race network on the BTCUSDT trade stream (E17_PREREGISTRATION.md).

    python experiments/e17_market.py --learner race|race_frozen|race_hold|b1|b0 [--depth 1] [--seed 0]

Each run passes once through all days in order (one stream). Decision episodes start every W = 10 s;
the race commits at its first output crossing (or abstains), and the label is the price move over
τ = 10 s after the decision. Causality: chunk j (32 episodes) is predicted with weights taught only
through chunk j−2; chunk j−1 is taught after chunk j has been predicted (labels arrive ≤ 20 s after
an episode starts, a chunk spans 320 s).
"""
import argparse
import datetime as dt
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, Config, to_events  # noqa: E402
from e14_depth import DeepRaceNet  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "binance")
OUT = os.path.join(os.path.dirname(__file__), "results", "e17")
W_US, TAU_US = 10_000_000, 10_000_000          # window and label horizon, microseconds
FLAT_BP, COST_BP = 1.0, (2.0, 10.0)
HOLD_BP = 2.0                                   # race_hold: moves below the 2 bp cost are "hold"
RANKS, CHUNK = 8, 32
PILOT = [dt.date(2026, 8, 25) + dt.timedelta(d) for d in range(7)]
CONF = [dt.date(2026, 9, 1) + dt.timedelta(d) for d in range(21)]


def load_day(day):
    """Trades of one day as (t µs int64, price float64, qty float32, sell bool), cached as npz."""
    cache = os.path.join(ROOT, "npz", f"{day}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["t"], z["p"], z["q"], z["sell"]
    with zipfile.ZipFile(os.path.join(ROOT, f"BTCUSDT-aggTrades-{day}.zip")) as zf:
        raw = zf.read(zf.namelist()[0])
    raw = raw.replace(b"True", b"1").replace(b"False", b"0")
    a = np.loadtxt(io.BytesIO(raw), delimiter=",", usecols=(1, 2, 5, 6), dtype=np.float64)
    del raw
    t = a[:, 2].astype(np.int64)
    if t[0] < 10 ** 14:                          # milliseconds in older files
        t = t * 1000
    p, q, sell = a[:, 0], a[:, 1].astype(np.float32), a[:, 3] > 0.5   # buyer is maker: a sell aggressor
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, t=t, p=p, q=q, sell=sell)
    return t, p, q, sell


def size_edges():
    qs = np.concatenate([load_day(d)[2][::20] for d in PILOT])
    return np.quantile(qs, [0.25, 0.5, 0.75])


def episodes(day, edges):
    """Spike-time matrix (n_ep, 192), episode starts, and the trade arrays for labels."""
    t, p, q, sell = load_day(day)
    start = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1e6)
    n_ep = (86_400_000_000 - W_US - TAU_US) // W_US          # labels must stay inside the day
    t0 = start + W_US * np.arange(n_ep, dtype=np.int64)
    e = (t - start) // W_US
    keep = (e >= 0) & (e < n_ep)
    tick = np.sign(np.diff(p, prepend=p[0])).astype(np.int64) + 1          # 0 down, 1 same, 2 up
    typ = sell.astype(np.int64) * 12 + np.searchsorted(edges, q).astype(np.int64) * 3 + tick
    ek, tk, yk, tt = e[keep], t[keep], typ[keep], t[keep]
    order = np.lexsort((tk, yk, ek))
    ek, yk, tt = ek[order], yk[order], tt[order]
    key = ek * 24 + yk
    first = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    rank = np.arange(len(key)) - np.repeat(first, np.diff(np.r_[first, len(key)]))
    m = rank < RANKS
    X = np.full((n_ep, 24 * RANKS), np.inf, np.float32)
    X[ek[m], yk[m] * RANKS + rank[m]] = ((tt[m] - t0[ek[m]]) / W_US * HORIZON).astype(np.float32)
    return X, t0, (t, p, q, sell)


def price_at(t, p, u):
    return p[np.clip(np.searchsorted(t, u, "right") - 1, 0, len(p) - 1)]


def window_features(t0, tr):
    """B1 features at the end of each window."""
    t, p, q, sell = tr
    lo, hi = np.searchsorted(t, t0), np.searchsorted(t, t0 + W_US)
    sv = np.r_[0, np.cumsum(np.where(sell, -q, q).astype(np.float64))]
    sc = np.r_[0, np.cumsum(np.where(sell, -1.0, 1.0))]
    p0, p1 = price_at(t, p, t0 - 1), price_at(t, p, t0 + W_US)
    last_side = np.where(hi > lo, np.where(sell[np.maximum(hi - 1, 0)], -1.0, 1.0), 0.0)
    return np.stack([np.sign(sv[hi] - sv[lo]) * np.log1p(np.abs(sv[hi] - sv[lo])),
                     np.log1p(hi - lo), (sc[hi] - sc[lo]) / np.sqrt(np.maximum(hi - lo, 1)),
                     1e4 * np.log(p1 / p0), last_side], 1)


def main(a):
    rng = np.random.default_rng(a.seed)
    edges = size_edges()
    days = PILOT + CONF
    k_out = 3 if a.learner == "race_hold" else 2
    net = None
    if a.learner.startswith("race"):
        X0, _, _ = episodes(PILOT[0], edges)
        drive = np.where(np.isfinite(X0), HORIZON - X0, 0).mean(0)
        cfg = Config(variant="crl_fa", winners=3, hid_frac=0.6, eta_out=0.01, eta_hid=0.01, deadline=1,
                     psp="ramp", homeo=0.001, sigma=0.15, zero_sum=1, seed=a.seed)
        net = DeepRaceNet(cfg, [a.width] * a.depth, X0.shape[1], k_out, drive, rng)
        for attr, v in dict(window=0.15, nonneg=False, eg=0.0, homeo_mode="linear", homeo_rate=0.001,
                            info_capacity=False, group_conserve=False, pivot_top=False, share_jac=False,
                            causal=False, center_credit=0, gauge=False, self_sigma=0).items():
            setattr(net, attr, v)
    lr_w, wb = np.zeros(5), 0.0                                  # B1: online logistic regression
    mu, var, nstat = np.zeros(5), np.ones(5), 0
    per_day, rows = [], {k: [] for k in ("day", "decided", "pred", "ret", "tdec", "conf")}
    for day in days:
        X, t0, tr = episodes(day, edges)
        t, p = tr[0], tr[1]
        learning = not (a.learner == "race_frozen" and day not in PILOT)
        if a.learner.startswith("race"):
            pend = None
            dec_all, pred_all, tdec_all = [], [], []
            for c0 in range(0, len(X), CHUNK):
                sl = slice(c0, min(c0 + CHUNK, len(X)))
                st = net.forward(*to_events(X[sl]))
                urgent, win = st["urgent"], st["winner"]
                tdec = np.where(urgent, W_US, (st["t_dec"] / HORIZON * W_US)).astype(np.int64)
                decided = ~urgent & ((win < 2) if k_out == 3 else True)
                dec_all.append(decided), pred_all.append(win), tdec_all.append(tdec)
                if pend is not None and learning:                   # teach chunk j-1 now
                    ps, ptd = pend
                    tc = t0[ps] + ptd
                    r = 1e4 * np.log(price_at(t, p, tc + TAU_US) / price_at(t, p, tc))
                    if k_out == 2:
                        ok = np.abs(r) >= FLAT_BP
                        y = (r < 0).astype(np.int64)
                    else:
                        ok = np.ones(len(r), bool)
                        y = np.where(np.abs(r) < HOLD_BP, 2, (r < 0).astype(np.int64))
                    if ok.any():
                        Xs = X[ps][ok]
                        net.teach(net.forward(*to_events(Xs)), y[ok])
                pend = (sl, tdec)
            decided, pred, tdec = map(np.concatenate, (dec_all, pred_all, tdec_all))
            conf = np.zeros(len(pred))
        else:
            F = window_features(t0, tr)
            tdec = np.full(len(t0), W_US, np.int64)
            r_lab = 1e4 * np.log(price_at(t, p, t0 + W_US + TAU_US) / price_at(t, p, t0 + W_US))
            if a.learner == "b0":                                  # reversal = the complement, on non-flat labels
                s = np.sign(F[:, 3])
                pred = np.where(s >= 0, 0, 1)                      # momentum: up if the window rose
                decided = s != 0
                conf = np.abs(F[:, 3])
            else:
                pred, conf = np.zeros(len(t0), np.int64), np.zeros(len(t0))
                for c0 in range(0, len(t0), CHUNK):               # predict chunk j, then learn chunk j-1
                    sl = slice(c0, min(c0 + CHUNK, len(t0)))
                    z = (F[sl] - mu) / np.sqrt(var + 1e-9)
                    pu = 1 / (1 + np.exp(-(z @ lr_w + wb)))
                    pred[sl], conf[sl] = (pu < 0.5).astype(np.int64), np.abs(pu - 0.5)
                    if c0 >= CHUNK:
                        ps = slice(c0 - CHUNK, c0)
                        for f, rr in zip(F[ps], r_lab[ps]):
                            nstat += 1
                            d_ = f - mu
                            mu += d_ / nstat
                            var += (d_ * (f - mu) - var) / nstat
                            if abs(rr) >= FLAT_BP:
                                zz = (f - mu) / np.sqrt(var + 1e-9)
                                g = 1 / (1 + np.exp(-(zz @ lr_w + wb))) - float(rr > 0)
                                lr_w -= 0.01 * g * zz
                                wb -= 0.01 * g
                decided = np.ones(len(t0), bool)
        tc = t0 + tdec
        r = 1e4 * np.log(price_at(t, p, tc + TAU_US) / price_at(t, p, tc))
        sgn = np.where(pred == 0, 1.0, -1.0)
        nf = decided & (np.abs(r) >= FLAT_BP)
        rec = {"day": str(day), "episodes": int(len(t0)), "decided": int(decided.sum()),
               "nonflat_decided": int(nf.sum()), "correct": int(((sgn * r > 0) & nf).sum()),
               "tdec_mean_s": float(tdec[decided].mean() / 1e6) if decided.any() else None}
        for c in COST_BP:
            rec[f"profit_bp_sum_c{c:g}"] = float(np.where(decided, sgn * r - c, 0.0).sum())
        per_day.append(rec)
        if day in CONF:
            for key, v in (("day", np.full(len(t0), (day - CONF[0]).days)), ("decided", decided), ("pred", pred),
                           ("ret", r), ("tdec", tdec), ("conf", conf)):
                rows[key].append(v)
        acc = rec["correct"] / max(rec["nonflat_decided"], 1)
        print(f"{day} {a.learner}: acc {acc:.4f} decided {rec['decided']}/{rec['episodes']} "
              f"tdec {rec['tdec_mean_s']} profit@2bp {rec['profit_bp_sum_c2']:.0f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    name = f"{a.learner}_d{a.depth}_w{a.width}_s{a.seed}"
    np.savez_compressed(os.path.join(OUT, name + "_episodes.npz"), **{k: np.concatenate(v) for k, v in rows.items()})
    conf_days = [r for r in per_day if dt.date.fromisoformat(r["day"]) in CONF]
    tot = lambda k: sum(r[k] for r in conf_days)  # noqa: E731
    summary = {"config": vars(a), "per_day": per_day,
               "confirmatory": {"acc": tot("correct") / max(tot("nonflat_decided"), 1),
                                "coverage": tot("decided") / tot("episodes"),
                                "profit_bp_per_episode_c2": tot("profit_bp_sum_c2") / tot("episodes"),
                                "profit_bp_per_episode_c10": tot("profit_bp_sum_c10") / tot("episodes")}}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("confirmatory:", json.dumps(summary["confirmatory"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True, choices=("race", "race_frozen", "race_hold", "b1", "b0"))
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=0, help="smoke test: only the first N days")
    a = ap.parse_args()
    if a.days:
        PILOT[:] = PILOT[:min(a.days, 7)]
        CONF[:] = CONF[:max(a.days - 7, 1)]
    main(a)
