"""E39: checks of THEORY §71's depth hierarchy.

(1) Exhaustive single-node search. Times on an integer grid {0..G-1} (G small: bounded domain, where the theorem
    does not apply; G large relative to the node's delays and windows: the scale-free regime of the theorem), all distinct (strict orders are then exact).
    A node: trigger k; every other input gets a role in {none, hold, veto} with delay d and window w on small grids
    (w = inf allowed); trigger delay d_T. Does some node accept exactly the configurations with t_1 < ... < t_m?
    Theorem 1: yes for m = 3, no for m = 4. The search enumerates role/parameter choices per input independently
    (Lemma 1 makes acceptance a product over inputs, so each input's choice can be scored separately against the
    target's per-input requirement; the search is still exhaustive over all combinations).
(2) Depth-2 Allen relations: every one of the thirteen as atom nodes + one max node, checked against its definition
    on random interval pairs with planted equalities (as E33, whose 3-node chains had depth 3).
"""
import itertools
import json
import os

import numpy as np

import e33_allen as E33

OUT = os.path.join(os.path.dirname(__file__), "results", "e39")
INF = np.inf


def node_accepts(t, k, roles, dT):
    """t: tuple of times; k: trigger index; roles[i] = (role, d, w) for i != k."""
    tau = t[k] + dT
    for i, (role, d, w) in roles.items():
        a = t[i] + d
        inside = (tau - w <= a <= tau)
        if role == "hold" and not inside:
            return False
        if role == "veto" and inside:
            return False
    return True


def search(m, G=5, delays=(0, 1, 2), windows=(0, 1, 2, INF)):
    """vectorized exhaustive search (Lemma 1: acceptance is an AND over inputs of per-input conditions)."""
    confs = np.array(list(itertools.permutations(range(G), m)), float)      # distinct times
    target = np.all(confs[:, 1:] > confs[:, :-1], 1)
    choices = [("none", 0, 0)] + [(r, d, w) for r in ("hold", "veto") for d in delays for w in windows]
    found = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        for dT in delays:
            tau = confs[:, k] + dT
            ok = {}
            for i in others:
                for c in choices:
                    r, d, w = c
                    a = confs[:, i] + d
                    inside = (tau - w <= a) & (a <= tau)
                    ok[(i, c)] = np.ones(len(confs), bool) if r == "none" else (inside if r == "hold" else ~inside)
            for combo in itertools.product(choices, repeat=len(others)):
                acc = np.ones(len(confs), bool)
                for i, c in zip(others, combo):
                    acc &= ok[(i, c)]
                if np.array_equal(acc, target):
                    found.append({"trigger": k, "dT": dT, "roles": {str(i): [c[0], c[1], None if c[2] == INF else c[2]]
                                                                     for i, c in zip(others, combo)}})
                    if len(found) >= 3:
                        return found, len(confs)
    return found, len(confs)


def at_node(p, q, lo, hi):
    """atom node: fires at q iff q - p in [lo, hi] (trigger q, hold p with delay lo and window hi - lo)."""
    return E33.node((q, 0.0), holds=[(p, lo, hi - lo)])


def allen_depth2(a0, a1, b0, b1, eps):
    """each relation = atoms (depth 1) + one max node (depth 2). Output: fired or not."""
    BIG = 1e9
    lt = lambda p, q: at_node(p, q, eps + 1e-12, BIG)          # noqa: E731   q - p > eps
    eq = lambda p, q: E33.near(p, q, eps)                          # noqa: E731   |q - p| <= eps, one node
    def AND(*xs):
        return max(xs) if all(np.isfinite(x) for x in xs) else INF
    R = {
        "before": lt(a1, b0), "meets": eq(a1, b0),
        "overlaps": AND(lt(a0, b0), lt(b0, a1), lt(a1, b1)),
        "starts": AND(eq(a0, b0), lt(a1, b1)),
        "during": AND(lt(b0, a0), lt(a1, b1)),
        "finishes": AND(lt(b0, a0), eq(a1, b1)),
        "equals": AND(eq(a0, b0), eq(a1, b1)),
        "after": lt(b1, a0), "met-by": eq(b1, a0),
        "overlapped-by": AND(lt(b0, a0), lt(a0, b1), lt(b1, a1)),
        "started-by": AND(eq(a0, b0), lt(b1, a1)),
        "contains": AND(lt(a0, b0), lt(b1, a1)),
        "finished-by": AND(lt(a0, b0), eq(a1, b1)),
    }
    return {k: bool(np.isfinite(v)) for k, v in R.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for m, G in ((2, 12), (3, 12), (4, 5), (4, 8), (4, 12)):
        found, n = search(m, G)
        res[f"single_node_order_m{m}_G{G}"] = {"configurations": n, "nodes_found": len(found), "examples": found[:2]}
        print(m, "grid", G, "configs", n, "single nodes computing the order:", len(found), found[:1], flush=True)
    eps, rng = 0.05, np.random.default_rng(1)
    wrong, seen, skipped = {}, {}, 0
    for _ in range(20000):
        a0, a1, b0, b1 = E33.sample(rng, eps)
        pts = sorted([a0, a1, b0, b1])
        if any(eps * 0.9 < y - x < eps * 1.1 for x, y in zip(pts, pts[1:])):
            skipped += 1
            continue
        tr, got = E33.truth(a0, a1, b0, b1, eps), allen_depth2(a0, a1, b0, b1, eps)
        for k in tr:
            seen[k] = seen.get(k, 0) + int(tr[k])
            if tr[k] != got[k]:
                wrong[k] = wrong.get(k, 0) + 1
    res["allen_depth2"] = {"exact": not wrong, "errors": wrong, "positives": seen, "skipped": skipped}
    print("allen depth 2 exact:", not wrong, wrong)
    with open(os.path.join(OUT, "depth.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
