#!/usr/bin/env python3
"""M20 (THEORY.md §14.6): beams computed asynchronously, as shadow events.

One discrete-event pass computes the factual history and, alongside it, one branch per
hidden-group collapse: "the group's last winner m does not fire". In a branch:

  - the group's remaining members keep integrating as tagged shadow strands
    (shadow continuation), and the first to cross takes m's place;
  - output nodes carry per-branch deltas (m's spike removed, the replacement added),
    so their branch potential is the factual one plus the delta (ramp integration is
    linear), and their shadow fire events are rescheduled whenever inputs change;
  - the branch resolves at its own first output spike, or at the deadline.

Check: each branch's winner equals the batch computation of the same swap (M19's
definition), on every sample; and the extra events are counted.

    python experiments/m20_shadow.py --samples 200
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, Config, layer_race, to_events  # noqa: E402
from sleeping_machines.sim import Engine  # noqa: E402
from theory_checks import make_net, pretrain, small_mnist  # noqa: E402
from m18_repair import output_race  # noqa: E402


def shadow_forward(net, times_row):
    e = Engine()
    cfg, h, k = net.cfg, net.h, net.k
    W1, W2, th1, th2 = net.W1, net.W2, net.th1, net.th2
    A1, B1 = np.zeros(h), np.zeros(h)
    A2, B2 = np.zeros(k), np.zeros(k)
    inhibited = np.zeros(h, bool)
    fired_t = np.full(h, np.inf)
    group_fired = np.zeros(h // cfg.group, int)
    pend1, pend2 = {}, {}
    state = {"winner": None}
    branches = []          # dict(m, members, sA, sB, dA, dB, pend_h, pend_o, winner)

    def crossing(A, B, th):
        return (th + B) / A if A > 0 else np.inf

    def resched(pend, key, tstar, kind, data):
        if key in pend:
            e.cancel(pend.pop(key))
        if np.isfinite(tstar) and tstar < HORIZON:
            pend[key] = e.schedule(max(tstar - e.now, 0.0), kind, data)

    def out_refresh(bi=None):
        """Reschedule factual output fire events, and those of live branches."""
        if state["winner"] is None and bi is None:
            for o in range(k):
                resched(pend2, o, crossing(A2[o], B2[o], th2[o]), "ofire", o)
        for j, br in enumerate(branches):
            if br["winner"] is not None or (bi is not None and j != bi):
                continue
            for o in range(k):
                resched(br["pend_o"], o, crossing(A2[o] + br["dA"][o], B2[o] + br["dB"][o], th2[o]),
                        "bofire", (j, o))
                e.count("shadow_output_updates")

    def inputs(chans):
        for n in np.flatnonzero(~inhibited):
            dw = W1[n, chans].sum()
            A1[n] += dw
            B1[n] += dw * e.now
            resched(pend1, n, crossing(A1[n], B1[n], th1[n]), "hfire", int(n))
        for j, br in enumerate(branches):           # shadow continuation of the group's members
            if br["done_h"]:
                continue
            for n in br["members"]:
                dw = W1[n, chans].sum()
                br["sA"][n] += dw
                br["sB"][n] += dw * e.now
                resched(br["pend_h"], n, crossing(br["sA"][n], br["sB"][n], th1[n]), "bhfire", (j, int(n)))
                e.count("shadow_hidden_updates")

    def hfire(n):
        pend1.pop(n, None)
        if inhibited[n]:
            return
        inhibited[n], fired_t[n] = True, e.now
        g = n // cfg.group
        group_fired[g] += 1
        # the spike reaches the outputs (factual, and every live branch shares it)
        A2[:] += W2[:, n]
        B2[:] += W2[:, n] * e.now
        if group_fired[g] == cfg.winners:
            members = [m for m in range(g * cfg.group, (g + 1) * cfg.group) if not inhibited[m]]
            for m in range(g * cfg.group, (g + 1) * cfg.group):
                inhibited[m] = True
                if m in pend1:
                    e.cancel(pend1.pop(m))
            if state["winner"] is None and members:  # fork: "n (the last winner) does not fire"
                br = dict(m=n, members=members, sA={m: A1[m] for m in members}, sB={m: B1[m] for m in members},
                          dA=-W2[:, n].copy(), dB=-W2[:, n] * e.now, pend_h={}, pend_o={}, winner=None,
                          done_h=False)
                branches.append(br)
                j = len(branches) - 1
                for m in members:
                    resched(br["pend_h"], m, crossing(br["sA"][m], br["sB"][m], th1[m]), "bhfire", (j, int(m)))
                out_refresh(j)
        out_refresh()

    def bhfire(data):
        j, n = data
        br = branches[j]
        br["pend_h"].pop(n, None)
        if br["done_h"] or br["winner"] is not None:
            return
        br["done_h"] = True                          # the first member to cross replaces m
        for key in list(br["pend_h"]):
            e.cancel(br["pend_h"].pop(key))
        br["dA"] += W2[:, n]
        br["dB"] += W2[:, n] * e.now
        br["replacement"] = (n, e.now)
        out_refresh(j)

    def ofire(o):
        pend2.pop(o, None)
        if state["winner"] is None:
            state["winner"] = o
            for key in list(pend2):
                e.cancel(pend2.pop(key))

    def bofire(data):
        j, o = data
        br = branches[j]
        br["pend_o"].pop(o, None)
        if br["winner"] is None:
            br["winner"] = o
            for key in list(br["pend_o"]):
                e.cancel(br["pend_o"].pop(key))
            for key in list(br["pend_h"]):
                e.cancel(br["pend_h"].pop(key))

    e.on("in", inputs)
    e.on("hfire", hfire)
    e.on("ofire", ofire)
    e.on("bhfire", bhfire)
    e.on("bofire", bofire)
    active = np.flatnonzero(np.isfinite(times_row))
    for tval in np.unique(times_row[active]):
        e.schedule(float(tval), "in", active[times_row[active] == tval])
    e.run(until=HORIZON)
    if state["winner"] is None and cfg.deadline:     # collapsing bound at the horizon
        state["winner"] = int(np.argmax(A2 * HORIZON - B2))
    for br in branches:
        if br["winner"] is None and cfg.deadline:
            br["winner"] = int(np.argmax((A2 + br["dA"]) * HORIZON - (B2 + br["dB"])))
    return state["winner"], branches, fired_t, e.work


def batch_branches(net, times_row, branches, fired_t):
    """The same swaps computed in batch: m removed, the replacement at its uninhibited crossing."""
    t, idx = to_events(times_row[None])
    T1, _, _ = layer_race(t, idx, net.W1, net.th1, "ramp")
    rows = []
    for br in branches:
        h_times = fired_t.copy()
        h_times[br["m"]] = np.inf
        cand = [m for m in br["members"] if np.isfinite(T1[0, m]) and T1[0, m] < HORIZON]
        if cand:
            n = min(cand, key=lambda m: T1[0, m])
            h_times[n] = T1[0, n]
        rows.append(h_times)
    if not rows:
        return []
    return list(output_race(net, np.stack(rows).astype(np.float32))[0])


def main(a):
    cfg = Config(hidden=a.hidden, group=10, winners=3, hid_frac=0.6, psp="ramp", deadline=1,
                 eta_out=0.01, eta_hid=0.01, homeo=0.001, seed=a.seed)
    x, y = small_mnist(3000)
    net = make_net(cfg, x, a.seed)
    pretrain(net, x, y, 1)
    xt, _ = small_mnist(a.samples, offset=50000)
    agree = total = factual_ok = 0
    work = {}
    for i in range(a.samples):
        w, branches, fired_t, wk = shadow_forward(net, xt[i])
        t, idx = to_events(xt[i:i + 1])
        factual_ok += int(w == net.forward(t, idx)["winner"][0])
        ref = batch_branches(net, xt[i], branches, fired_t)
        agree += sum(int(br["winner"] == r) for br, r in zip(branches, ref))
        total += len(branches)
        for key, v in wk.items():
            work[key] = work.get(key, 0) + v
    res = {"samples": a.samples, "factual_agreement": factual_ok / a.samples,
           "branches_per_sample": total / a.samples, "branch_agreement": agree / max(total, 1),
           "work_per_sample": {kk: v / a.samples for kk, v in work.items()}}
    print(json.dumps(res, indent=1), flush=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results", "theory"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "theory", f"m20_s{a.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=60)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
