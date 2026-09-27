"""Data for the mechanism figures (report/figures_mech.py): trains E53's depth-3 network once and records
(1) the anatomy of one positive episode and one decoy episode, (2) the drive at the prefix and at the valid instant
of one class over training under three credit rules. Writes report/data/mech.json. ~30 s, one thread."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import e53_depth3 as E  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "data", "mech.json")
N, H, Q = 16, 16.0, 0.25


def unit_desc(net, u):
    if u < net.P:
        return {"level": 0, "parts": [[int(c) for c in net.pp[u]]]}
    a, b = divmod(u - net.P, net.P)
    return {"level": 1, "parts": [[int(c) for c in net.pp[a]], [int(c) for c in net.pp[b]]]}


def instants(net, t, cls, motifs):
    """times of the prefix instant (end of the 2nd motif) and the valid instant (end of the 3rd)."""
    return float(t[motifs[cls[1]][1]]), float(t[motifs[cls[2]][1]])


def drive(net, k, t, tau):
    U, x = net.units(t)
    win = (x < tau) & (x >= tau - E.W); same = x == tau
    return float(min(net.h[k, U[win]].sum(), net.g[k, U[same]].sum()))


def main():
    rng = np.random.default_rng(0)
    task = E.make_task(N, 6, 5, 4, rng); motifs, classes, decoys = task
    # classes whose two-motif prefix is shared with another class (a prefix route is tempting and wrong)
    shared = [i for i, c in enumerate(classes) if any(j != i and classes[j][:2] == c[:2] for j in range(len(classes)))]
    pr = np.random.default_rng(11)
    probes = {}
    while len(probes) < len(shared):
        t, y = E.sample(task, N, H, Q, pr)
        if y in shared and y not in probes:
            probes[y] = t
    inst = {i: instants(None, probes[i], classes[i], motifs) for i in shared}

    # (2) credit dynamics: drive at the prefix and at the valid instant, on a fixed probe per class
    dyn_all = {}
    for name, credit, temp in (("every candidate (union)", "union", 0.0), ("greedy instant", "instant", 0.0),
                               ("instant + cooled exploration", "instant", 0.3)):
        r2 = np.random.default_rng(0); E.make_task(N, 6, 5, 4, r2)       # same task draw, same stream
        net = E.Net(N, len(classes), 3, r2, 0.6, 1.0, 0.5, credit); net.temp = temp
        rows = {i: [] for i in shared}
        for step in range(1, 8001):
            t, y = E.sample(task, N, H, Q, r2); net.teach(t, y)
            if step % 25 == 0:
                for i in shared:
                    rows[i].append([step, drive(net, i, probes[i], inst[i][0]), drive(net, i, probes[i], inst[i][1])])
        dyn_all[name] = rows
    # the class greedy traps longest on its prefix
    k = max(shared, key=lambda i: sum(r[1] > 0.6 for r in dyn_all["greedy instant"][i]))
    other = next(j for j in range(len(classes)) if j != k and classes[j][:2] == classes[k][:2])
    dyn = {name: rows[k] for name, rows in dyn_all.items()}
    t_pos = probes[k]; tauB, tauC = inst[k]
    dec = next(d for d in decoys if sorted(d) == sorted(classes[k]))
    mch = [c for m in classes[k] for c in motifs[m][:2]]
    start = min(t_pos[c] for c in mch)
    for _ in range(200):                               # decoy: same noise as the positive, motifs re-planted in a
        t_dec = t_pos.copy(); t_dec[mch] = np.inf      # non-class order (a controlled contrast: only order differs)
        E.plant_seq(t_dec, motifs, dec, start, pr)
        if not any(E.holds(motifs, c, t_dec) for c in classes):
            break

    # (1) anatomy with the converged network (exploration, margin)
    r3 = np.random.default_rng(0); E.make_task(N, 6, 5, 4, r3)
    net = E.Net(N, len(classes), 3, r3, 0.6, 1.0, 0.5, "instant"); net.temp = 0.3; net.margin = 0.9
    for _ in range(8000):
        t, y = E.sample(task, N, H, Q, r3); net.teach(t, y)
    anat = {}
    for name, t in (("positive", t_pos), ("decoy", t_dec)):
        c, U, win, same, inst, ft = net.forward(t)
        Ux, x = net.units(t)
        units = [{**unit_desc(net, int(u)), "t": float(xx),
                  "h": float(net.h[k, u]), "g": float(net.g[k, u])} for u, xx in zip(Ux, x)]
        anat[name] = {"spikes": [float(v) for v in t], "units": units, "winner": int(c),
                      "fire_times": [float(v) if np.isfinite(v) else None for v in ft]}
    res = {"class": int(k), "class_order": list(classes[k]), "other_class": int(other),
           "other_order": list(classes[other]), "decoy_order": list(dec),
           "motifs": [[int(i), int(j), float(D)] for i, j, D in motifs], "W": E.W, "theta": 0.6,
           "prefix_instant": tauB, "valid_instant": tauC, "dynamics": dyn, "selection": "the prefix-sharing class that greedy credit trapped longest",
           "trapped_fraction_greedy": {str(i): sum(r[1] > 0.6 for r in dyn_all["greedy instant"][i]) / len(dyn_all["greedy instant"][i]) for i in shared}, "anatomy": anat,
           "n_candidates": int(net.P + net.P ** 2), "K": len(classes)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f)
    print("wrote", OUT, "class", k, classes[k], "winner pos/dec", anat["positive"]["winner"], anat["decoy"]["winner"])


if __name__ == "__main__":
    main()
