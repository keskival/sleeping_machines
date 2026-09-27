"""E39b: THEORY §75 (M75a): interval exclusion "B after A, and C not between A and B" is computed by one stateful
arm/disarm node and by no stateless node (exhaustive search over e39's single-node family on a wide grid).

Stateful node: A arms it, C disarms it, B fires it if armed (event semantics, one shot).
"""
import itertools
import json
import os

import numpy as np

import e39_depth as E39

OUT = os.path.join(os.path.dirname(__file__), "results", "e39")
INF = np.inf


def stateful(tA, tB, tC):
    events = sorted([(tA, "A"), (tB, "B"), (tC, "C")])
    armed = False
    for _, e in events:
        if e == "A":
            armed = True
        elif e == "C":
            armed = False
        else:
            return armed
    return False


def main(G=12, delays=(0, 1, 2), windows=(0, 1, 2, INF)):
    confs = np.array(list(itertools.permutations(range(G), 3)), float)     # (tA, tB, tC), distinct
    target = (confs[:, 1] > confs[:, 0]) & ~((confs[:, 2] > confs[:, 0]) & (confs[:, 2] < confs[:, 1]))
    st = np.array([stateful(*c) for c in confs])
    choices = [("none", 0, 0)] + [(r, d, w) for r in ("hold", "veto") for d in delays for w in windows]
    found = 0
    for k in range(3):
        others = [i for i in range(3) if i != k]
        for dT in delays:
            tau = confs[:, k] + dT
            ok = {}
            for i in others:
                for c in choices:
                    r, d, w = c
                    a = confs[:, i] + d
                    inside = (tau - w <= a) & (a <= tau)
                    ok[(i, c)] = np.ones(len(confs), bool) if r == "none" else (inside if r == "hold" else ~inside)
            for combo in itertools.product(choices, repeat=2):
                acc = ok[(others[0], combo[0])] & ok[(others[1], combo[1])]
                found += bool(np.array_equal(acc, target))
    res = {"grid": G, "configurations": len(confs), "stateful_exact": bool(np.array_equal(st, target)),
           "stateless_nodes_found": found}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "stateful.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(G=5); main(G=12)
