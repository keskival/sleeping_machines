"""NeuroBench non-human-primate motor prediction (primate reaching) with the integrated native core
(experiments/SOTA_TARGETS.md, target 2).

Official data and preprocessing: neurobench 2.3.0 PrimateReaching (vendored, unmodified) with the leaderboard settings of
the baselines and the top entries (num_steps=1, train_ratio=0.5, bin_width=0.004, biological_delay=0,
remove_segments_inactive=False): 4 ms bins of multi-unit spikes (96 channels for indy, 192 for loco) and fingertip
velocity labels.  The test bins are dataset.ind_test (the last 25% of each of the official chunks); the first 75%
(ind_train + ind_val) is ours for fitting, as in the BioCAS 2024 top entries (fmi-basel/neural-decoding-RSNN), which
hold out the last part of it for validation.  R2 is the official NeuroBench R2: per output dimension
1 - SSE / (N var), averaged over x and y, over all test bins of a session; the leaderboard averages sessions.

Model: one event per 4 ms bin (timestamp = bin index, content = the bin's spike vector), the integrated native core with
route credit and a 2-output regression head (standardized velocity).  Training: random windows of the fitting bins,
MSE after a warmup; inference: the exact winner-only stepper over the test stream from a fresh state (as the top entry
evaluates its stream with batch size 1).  One model per session.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/vendor/neurobench_2_3_0'))
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.sparse_inference import SparseStepper  # noqa: E402

DATA = ROOT / 'data/neurobench/primate_reaching'
SESSIONS = ['indy_20160622_01', 'indy_20160630_01', 'indy_20170131_02',
            'loco_20170210_03', 'loco_20170215_02', 'loco_20170301_05']


def load_session(name):
    from neurobench.datasets import PrimateReaching
    ds = PrimateReaching(file_path=str(DATA), filename=name, num_steps=1, train_ratio=0.5, bin_width=0.004,
                         biological_delay=0, remove_segments_inactive=False, download=False)
    spikes = ds.samples.T.numpy().astype(np.float32)        # (T, channels)
    labels = ds.labels.T.numpy().astype(np.float64)         # (T, 2) velocity
    fit = np.array(list(ds.ind_train) + list(ds.ind_val), dtype=np.int64)
    test = np.array(ds.ind_test, dtype=np.int64)
    return spikes, labels, fit, test


def r2(pred, label):
    """official NeuroBench R2 (metrics/workload/r2.py): mean over x, y of 1 - SSE / (N var)."""
    pred, label = np.asarray(pred, np.float64), np.asarray(label, np.float64)
    out = []
    for d in range(2):
        out.append(1 - np.sum((label[:, d] - pred[:, d]) ** 2) / (np.var(label[:, d]) * len(label)))
    return float(np.mean(out))


def make_model(a, channels):
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=channels, classes=2, payload=a.payload, depth=a.depth,
                                            heads=a.heads, pool=a.pool)
    if a.tie_pools:
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    return model


def run_session(a, name):
    spikes, labels, fit, test = load_session(name)
    n_val = int(round(len(fit) * a.val_fraction))
    train, val = fit[:len(fit) - n_val], fit[len(fit) - n_val:]
    mu, sd = labels[train].mean(0), labels[train].std(0)
    target = ((labels - mu) / sd).astype(np.float32)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed + 1)
    model = make_model(a, spikes.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    S, B = a.segment, a.lanes
    rc = None if a.route_credit == 'none' else a.route_credit
    logits_fn = batched_logits
    if a.compiled:
        from sleeping_machines.compiled_episodes import compiled_logits
        logits_fn = compiled_logits
    losses = []
    started = time.perf_counter()
    for step in range(a.steps):
        starts = rng.integers(0, len(train) - S, B)
        idx = np.stack([train[s:s + S] for s in starts])                       # (B, S) bin indices
        rows = [dict(events=[(float(t), spikes[i]) for t, i in enumerate(r)]) for r in idx]
        y = torch.tensor(target[idx])
        model.train(); opt.zero_grad(set_to_none=True)
        out = logits_fn(model, rows, 1000 + step, all_logits=True, route_credit=rc)
        loss = F.mse_loss(out[:, a.warmup:], y[:, a.warmup:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); schedule.step()
        losses.append(float(loss))
        if step % 200 == 0:
            print(json.dumps(dict(session=name, step=step, loss=float(np.mean(losses[-50:])))), flush=True)
    fit_s = time.perf_counter() - started
    model.eval()

    def stream(indices):
        stepper = SparseStepper(model, 1, 777); preds = []
        for t, i in enumerate(indices):
            preds.append(stepper.step(torch.tensor([float(t)]), torch.tensor(spikes[i])[None])[0].numpy())
        return np.array(preds) * sd + mu
    val_r2 = r2(stream(val), labels[val]) if len(val) else None
    test_r2 = r2(stream(test), labels[test])
    return dict(session=name, test_r2=test_r2, val_r2=val_r2, fit_bins=len(train), val_bins=len(val), test_bins=len(test),
                channels=int(spikes.shape[1]), final_train_mse=float(np.mean(losses[-50:])), fit_s=fit_s,
                parameters=sum(p.numel() for p in model.parameters()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--sessions', default=','.join(SESSIONS))
    p.add_argument('--payload', type=int, default=32); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--tie-pools', action='store_true')
    p.add_argument('--segment', type=int, default=250); p.add_argument('--lanes', type=int, default=32)
    p.add_argument('--steps', type=int, default=2000); p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--weight-decay', type=float, default=0.); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--warmup', type=int, default=25); p.add_argument('--val-fraction', type=float, default=.1333)
    p.add_argument('--route-credit', choices=('none', 'linear'), default='linear')
    p.add_argument('--compiled', action='store_true'); p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    out = ROOT / 'experiments/results/neurobench_primate' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); started = time.perf_counter()
    rows = []
    for name in a.sessions.split(','):
        rows.append(run_session(a, name)); print(json.dumps(rows[-1]), flush=True)
    result = dict(status='completed', args=vars(a), sessions=rows, mean_test_r2=float(np.mean([r['test_r2'] for r in rows])),
                  protocol='NeuroBench primate reaching (neurobench 2.3.0 loader, train_ratio .5, 4 ms bins); official R2',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/primate_reaching_native.py', 'sleeping_machines/sparse_inference.py',
                                  'sleeping_machines/batched_episodes.py', 'sleeping_machines/compiled_episodes.py',
                                  'experiments/vendor/neurobench_2_3_0/neurobench/datasets/primate_reaching.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mean_test_r2=result['mean_test_r2'])), flush=True)


if __name__ == '__main__':
    main()
