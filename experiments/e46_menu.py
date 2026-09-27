"""E46: E34's compositional task with per-class route menus and reliability prices (structure discovery per class, §80).

Parts: as E34 (directional hold nodes on channel pairs, a bank of window scales). Each class keeps a menu of up to M
candidate routes (hold part, trigger part). A route fires at its trigger part's spike if its hold part fired within W
before. Every route carries a price, a running estimate of its error: whenever it fires (right iff the episode is its
class) or its class was the answer and it did not fire (a miss), the price moves toward 1 on errors and 0 when right.
A class fires through its cheapest route that fires, if that route's price is below 1/2; classes race by firing time.
On a miss of class y with free menu slots (or a menu whose best price is poor), the counterfactual route (the pair of
fired parts, earlier -> later within W, that is most frequent in y's recent misses) is added, replacing the worst.
No weights: routes are discrete choices selected by reliability, errors only.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

import e28_routing as T
import e34_compose as C

OUT = os.path.join(os.path.dirname(__file__), "results", "e46")
INF = np.inf


class Menu:
    def __init__(self, N, K, W, scales, M=4, rate=0.05):
        self.parts = C.Compose(N, K, 2, W, np.random.default_rng(0), scales=scales)   # part basis only
        self.K, self.W, self.M, self.rate = K, W, M, rate
        self.routes = [[] for _ in range(K)]                 # per class: list of [hold, trig, price]
        self.cand = [Counter() for _ in range(K)]
        self.updates = 0

    def fire_time(self, x, r):
        h, g, _ = r
        if np.isfinite(x[g]) and np.isfinite(x[h]) and x[g] - self.W <= x[h] < x[g]:
            return x[g]
        return INF

    def forward(self, t):
        x = self.parts.parts(t)
        best = np.full(self.K, INF); used = [None] * self.K
        for c in range(self.K):
            for r in sorted(self.routes[c], key=lambda r: r[2]):
                if r[2] >= 0.5:
                    break
                ft = self.fire_time(x, r)
                if np.isfinite(ft):
                    best[c], used[c] = ft, r
                    break
        c = int(best.argmin()) if np.isfinite(best).any() else self.K
        return c, x, used

    def teach(self, t, y):
        c, x, used = self.forward(t)
        for k in range(self.K):                              # score every route (full information, §80)
            for r in self.routes[k]:
                fired = np.isfinite(self.fire_time(x, r))
                if fired or k == y:
                    r[2] += self.rate * ((0.0 if (fired and k == y) else 1.0) - r[2])
        if c == y:
            return
        self.updates += 1
        if y < self.K and used[y] is None:                   # miss: propose the counterfactual route
            fired = np.flatnonzero(np.isfinite(x))
            if len(fired) >= 2:
                xe, xl = x[fired][:, None], x[fired][None, :]
                ok = (xe < xl) & (xe >= xl - self.W)
                for i, j in zip(*np.nonzero(ok)):
                    self.cand[y][(int(fired[i]), int(fired[j]))] += 1
                (h, g), n = self.cand[y].most_common(1)[0]
                if not any(r[0] == h and r[1] == g for r in self.routes[y]):
                    if len(self.routes[y]) < self.M:
                        self.routes[y].append([h, g, 0.3])
                    else:
                        worst = max(self.routes[y], key=lambda r: r[2])
                        if worst[2] > 0.4:
                            worst[:] = [h, g, 0.3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        motifs, classes = T.make_task(16, 6, 15, rng)
        net = Menu(16, 15, 3.5, [float(v) for v in a.scales.split(",")], a.M)
        for step in range(a.steps):
            t, y = T.sample(motifs, classes, 16, 12.0, 0.25, rng); net.teach(t, y)
        ev = np.random.default_rng(99); ok = 0
        for _ in range(1500):
            t, y = T.sample(motifs, classes, 16, 12.0, 0.25, ev); ok += net.forward(t)[0] == y
        rows.append({"seed": s, "test": ok / 1500, "updates": net.updates})
        print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(OUT, f"menu_M{a.M}_{a.steps}{'_' + a.tag if a.tag else ''}.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows, "wall_s": round(time.time() - t0, 1)}, f)
    print("EXIT-OK", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
