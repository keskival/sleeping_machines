"""Diagnostic for E53's transient dips: after convergence, which updates break a class's valid route?"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
import e53_depth3 as E

for seed in (1, 4):
    rng = np.random.default_rng(seed); task = E.make_task(16, 6, 5, 4, rng); motifs, classes, decoys = task
    K = len(classes)
    net = E.Net(16, K, 3, rng, 0.6, 1.0, 0.5, 'instant'); net.temp = 0.3
    pid = {tuple(p): i for i, p in enumerate(net.pp)}
    tgt = []
    for (a, b, c) in classes:
        pa, pb, pc = pid[motifs[a][:2]], pid[motifs[b][:2]], pid[motifs[c][:2]]
        tgt.append((net.P + pa * net.P + pb, pc))
    def ok(k): return net.h[k, tgt[k][0]] > 0.6 and net.g[k, tgt[k][1]] > 0.6
    log = {}
    for step in range(1, 40001):
        t, y = E.sample(task, 16, 16.0, 0.25, rng)
        before = [ok(k) for k in range(K)]
        c = net.forward(t)[0]
        net.teach(t, y)
        if step <= 5000 or c == y: continue
        after = [ok(k) for k in range(K)]
        kind = 'miss' if (y < K and c == K) else ('falsefire_on_none' if y == K else 'wrong_class')
        for k in range(K):
            if before[k] and not after[k]:
                role = 'miss_promote_other' if k == y else 'demoted_as_false_winner'
                key = f'{kind}:{role}'; log[key] = log.get(key, 0) + 1
        log[f'updates:{kind}'] = log.get(f'updates:{kind}', 0) + 1
    print(json.dumps({'seed': seed, **log}), flush=True)
