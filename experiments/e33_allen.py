"""E33: Allen's thirteen interval relations wired from hold/veto nodes; temporal depth = order-chain length (§64).

Two intervals A = [a0, a1], B = [b0, b1], each given by an onset and an offset spike. Relations (tolerance eps for
"equal"): before, meets, overlaps, starts, during, finishes, equals, and the inverses of the first six.

Node (one-shot, event semantics): fires at the arrival of its trigger input iff every hold input arrived at most
its duration earlier (and not later than the trigger), and no veto input arrived within its duration before the
trigger. Every input has a delay (>= 0, causal), a duration, and a sign (hold or veto). A node's output is a spike
at its firing time, so nodes chain: a node checking "p2 after p1" hands a spike at p2's time to a node checking p3.

Claim checked here: a pair relation ("q within [lo, hi] after p", with exclusions) is one node; a relation fixing
the order of m points needs a chain of m - 1 nodes, and no single node can do it (two open holds do not record which
opened first). Every relation is verified against its definition on random interval pairs with planted equalities.
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results", "e33")
INF = np.inf
BIG = 1e9


def node(trig, holds=(), vetoes=()):
    """trig: (time, delay). holds: [(time, delay, duration)]. vetoes: [(time, delay, duration)]. Returns fire time."""
    t, d = trig
    if not np.isfinite(t):
        return INF
    ft = t + d
    for (h, dh, w) in holds:
        a = h + dh
        if not (np.isfinite(a) and ft - w <= a <= ft):
            return INF
    for (v, dv, w) in vetoes:
        a = v + dv
        if np.isfinite(a) and ft - w <= a <= ft:
            return INF
    return ft


def after(p, q, gap):
    """spike at q's time iff q comes more than `gap` after p: one node (trigger q, hold p delayed by gap)."""
    return node((q, 0.0), holds=[(p, gap, BIG)])


def near(p, q, eps):
    """spike iff |p - q| <= eps, at max(p, q) + eps: one node (trigger q delayed eps, hold p for 2 eps)."""
    return node((q, eps), holds=[(p, 0.0, 2 * eps)])


def relations(a0, a1, b0, b1, eps):
    """Each relation as a netlist of nodes (a chain where an order of points is fixed). Returns fired flags."""
    R = {}
    # E = near(x, y) fires at y + eps; later comparisons against it use gap 0 (the eps is already in its time)
    R["before"] = after(a1, b0, eps)                                            # a1 < b0            1 node
    R["meets"] = near(a1, b0, eps)                                              # a1 = b0            1 node
    R["overlaps"] = after(after(after(a0, b0, eps), a1, eps), b1, eps)          # a0 < b0 < a1 < b1  3 nodes
    R["starts"] = after(after(near(a0, b0, eps), a1, 0.0), b1, eps)             # a0 = b0 < a1 < b1  3 nodes
    R["during"] = after(after(after(b0, a0, eps), a1, 0.0), b1, eps)            # b0 < a0 < a1 < b1  3 nodes
    R["finishes"] = near(after(after(b0, a0, eps), a1, 0.0), b1, eps)           # b0 < a0 < a1 = b1  3 nodes
    R["equals"] = after(near(a0, b0, eps), near(a1, b1, eps), 0.0)              # a0 = b0, a1 = b1   3 nodes
    R["after"] = after(b1, a0, eps)
    R["met-by"] = near(b1, a0, eps)
    R["overlapped-by"] = after(after(after(b0, a0, eps), b1, eps), a1, eps)
    R["started-by"] = after(after(near(a0, b0, eps), b1, 0.0), a1, eps)
    R["contains"] = after(after(after(a0, b0, eps), b1, 0.0), a1, eps)
    R["finished-by"] = near(after(after(a0, b0, eps), b1, 0.0), a1, eps)
    R = {k: bool(np.isfinite(v)) for k, v in R.items()}
    return R


def truth(a0, a1, b0, b1, eps):
    eq = lambda x, y: abs(x - y) <= eps                                          # noqa: E731
    lt = lambda x, y: x < y - eps                                                # noqa: E731
    return {
        "before": lt(a1, b0), "meets": eq(a1, b0),
        "overlaps": lt(a0, b0) and lt(b0, a1) and lt(a1, b1),
        "starts": eq(a0, b0) and lt(a1, b1),
        "during": lt(b0, a0) and lt(a1, b1),
        "finishes": lt(b0, a0) and eq(a1, b1),
        "equals": eq(a0, b0) and eq(a1, b1),
        "after": lt(b1, a0), "met-by": eq(b1, a0),
        "overlapped-by": lt(b0, a0) and lt(a0, b1) and lt(b1, a1),
        "started-by": eq(a0, b0) and lt(b1, a1),
        "contains": lt(a0, b0) and lt(b1, a1),
        "finished-by": lt(a0, b0) and eq(a1, b1),
    }


def sample(rng, eps):
    """random interval pair; a third of the time plant one or two equalities; both intervals valid (onset < offset)."""
    while True:
        a0, a1, b0, b1 = _sample(rng, eps)
        if a1 > a0 + 0.2 and b1 > b0 + 0.2:
            return a0, a1, b0, b1


def _sample(rng, eps):
    a0, b0 = rng.uniform(0, 10, 2)
    a1, b1 = a0 + rng.uniform(0.5, 5), b0 + rng.uniform(0.5, 5)
    u = rng.random()
    if u < 0.1:
        b0 = a0 + rng.uniform(-eps / 2, eps / 2)
    elif u < 0.2:
        b1 = a1 + rng.uniform(-eps / 2, eps / 2)
    elif u < 0.25:
        b0 = a0 + rng.uniform(-eps / 2, eps / 2); b1 = a1 + rng.uniform(-eps / 2, eps / 2)
    elif u < 0.3:
        b0 = a1 + rng.uniform(-eps / 2, eps / 2)
    elif u < 0.35:
        a0 = b1 + rng.uniform(-eps / 2, eps / 2); a1 = a0 + rng.uniform(0.5, 5)
    return a0, a1, b0, b1


def main():
    eps, n = 0.05, 20000
    rng = np.random.default_rng(0)
    wrong, seen = {}, {}
    ambiguous = 0
    for _ in range(n):
        a0, a1, b0, b1 = sample(rng, eps)
        tr = truth(a0, a1, b0, b1, eps)
        # skip samples within eps-bands of a tolerance boundary (the definitions are then ambiguous at 2 eps)
        pts = sorted([a0, a1, b0, b1])
        if any(eps * 0.9 < y - x < eps * 1.1 for x, y in zip(pts, pts[1:])):
            ambiguous += 1
            continue
        got = relations(a0, a1, b0, b1, eps)
        for k in tr:
            seen[k] = seen.get(k, 0) + int(tr[k])
            if bool(got[k]) != bool(tr[k]):
                wrong[k] = wrong.get(k, 0) + 1
    res = {"samples": n, "ambiguous_skipped": ambiguous, "positives": seen, "errors": wrong,
           "all_exact": not wrong, "errors_total": int(sum(wrong.values())),
           "nodes": {"before": 1, "meets": 1, "after": 1, "met-by": 1, "overlaps": 3, "starts": 3, "during": 3,
                     "finishes": 3, "equals": 3, "overlapped-by": 3, "started-by": 3, "contains": 3,
                     "finished-by": 3}}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "allen.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
