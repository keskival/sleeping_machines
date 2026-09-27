"""E53 dips, behavioural: which update breaks a converged class? Probe each class on 20 fixed positives every 200 episodes;
when a class falls from >= 0.95 to < 0.5, report the updates that touched it since the previous probe."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import e53_depth3 as E

for seed in (1, 4, 0):
    rng = np.random.default_rng(seed); task = E.make_task(16, 6, 5, 4, rng); motifs, classes, decoys = task
    K = len(classes)
    net = E.Net(16, K, 3, rng, 0.6, 1.0, 0.5, 'instant'); net.temp = 0.3
    pr = np.random.default_rng(7); probes = {k: [] for k in range(K)}
    while min(len(v) for v in probes.values()) < 20:
        t, y = E.sample(task, 16, 16.0, 0.25, pr)
        if y < K and len(probes[y]) < 20: probes[y].append(t)
    def acc(k): return np.mean([net.forward(t)[0] == k for t in probes[k]])
    prev = np.array([acc(k) for k in range(K)]); touched = {k: [] for k in range(K)}; breaks = []
    for step in range(1, 40001):
        t, y = E.sample(task, 16, 16.0, 0.25, rng)
        c, U, win, same, inst, ft = net.forward(t)
        if c != y:
            kind = 'miss' if (y < K and not np.isfinite(ft[y])) else ('falsefire_none' if y == K else 'lost_race')
            if y < K: touched[y].append((step, kind, 'as_true'))
            if c < K: touched[c].append((step, kind, 'as_false_winner(true=%s)' % ('none' if y == K else y)))
        net.teach(t, y)
        if step % 200 == 0:
            cur = np.array([acc(k) for k in range(K)])
            for k in (np.flatnonzero((prev >= 0.95) & (cur < 0.5)) if step > 5000 else []):
                breaks.append({'step': step, 'class': int(k), 'acc': [float(prev[k]), float(cur[k])], 'updates': touched[k][-8:], 'n_updates_window': len(touched[k])})
            prev = cur; touched = {k: [] for k in range(K)}
    kinds = {}
    for b in breaks:
        for u in b['updates']:
            kinds[u[1] + ':' + u[2].split('(')[0]] = kinds.get(u[1] + ':' + u[2].split('(')[0], 0) + 1
    print(json.dumps({'seed': seed, 'n_breaks': len(breaks), 'kinds_before_breaks': kinds, 'late_breaks': breaks}), flush=True)
