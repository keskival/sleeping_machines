"""NeuroBench Mackey-Glass chaotic prediction with the integrated native core (experiments/SOTA_TARGETS.md, target 1).

Official protocol (neurobench 2.3.0, examples/mackey_glass/lstm_benchmark.py and datasets/mackey_glass.py): the downloaded
series mg_<tau>.npy; repeat r uses the slice [int(37.5 r), int(37.5 r) + 1501) (start offsets 0, 0.5, ..., in Lyapunov
times of 75 points); points 0..749 are the training inputs with next-value targets; the model then predicts the next 750
values autonomously, feeding back its own predictions, and the official SMAPE = 200 mean(|p - y| / (|p| + |y|)) is
computed on targets 751..1500 of the slice; scores are averaged over the repeats.

Model: the integrated core (temporal races, sparse addressed writes, rotating memories, transport; route credit) with a
one-output regression head.  Each event is one sample (timestamp = sample index, content = the last --taps standardized
values, newest first; zeros before the series start, as the official pre-padding).  Training: random windows of the
750 training points, teacher forced, MSE on all positions after --warmup; standardization from the training points only.
Inference: the exact winner-only stepper (sleeping_machines/sparse_inference.SparseStepper), warmed on the training
points, then autonomous.  Development uses other tau series; the tau = 17 protocol is run once with fixed settings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.sparse_inference import SparseStepper  # noqa: E402

DATA = ROOT / 'data/neurobench/mackey_glass/data'
TRAIN, TEST = 750, 750


def official_slice(tau, repeat):
    series = np.load(DATA / f'mg_{tau}.npy')
    offset = int(torch.arange(0., 0.5 * (repeat + 1), 0.5)[repeat].item() * 75)
    part = series[offset:offset + TRAIN + TEST + 1]
    if len(part) != TRAIN + TEST + 1:
        raise ValueError('Series too short for this repeat')
    return part.astype(np.float64)


def smape(pred, target):
    pred, target = torch.as_tensor(pred, dtype=torch.float64), torch.as_tensor(target, dtype=torch.float64)
    return torch.nan_to_num(200 * torch.mean(torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target))), nan=200.).item()


def taps_of(z, i, k):
    """content at sample i: z[i], z[i-1], ..., z[i-k+1] (zeros before the start)."""
    out = np.zeros(k, np.float32)
    for j in range(k):
        if i - j >= 0:
            out[j] = z[i - j]
    return out


def make_model(a):
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=a.taps, classes=1, payload=a.payload, depth=a.depth,
                                            heads=a.heads, pool=a.pool)
    if a.tie_pools:
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    return model


def fit_and_forecast(a, z, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 1)
    model = make_model(a)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    S, B = a.segment, a.lanes
    rc = None if a.route_credit == 'none' else a.route_credit
    logits_fn = batched_logits
    if a.compiled:
        from sleeping_machines.compiled_episodes import compiled_logits
        logits_fn = compiled_logits
    losses = []
    for step in range(a.steps):
        starts = rng.integers(0, TRAIN - S + 1, B)
        rows = [dict(events=[(float(i), taps_of(z, i, a.taps)) for i in range(s, s + S)]) for s in starts]
        y = torch.tensor(np.stack([z[s + 1:s + S + 1] for s in starts]), dtype=torch.float32)
        if a.delta:                                    # predict the increment z[t+1] - z[t]
            y = y - torch.tensor(np.stack([z[s:s + S] for s in starts]), dtype=torch.float32)
        model.train(); opt.zero_grad(set_to_none=True)
        closed = a.closed_loop and step >= a.closed_from * a.steps
        kw = dict(feedback=(a.closed_loop, lambda prev, logits: torch.cat([logits[:, :1].to(prev.dtype), prev[:, :-1]], -1)))\
            if closed else {}
        if closed and a.delta:
            kw = dict(feedback=(a.closed_loop, lambda prev, logits: torch.cat([prev[:, :1] + logits[:, :1].to(prev.dtype),
                                                                             prev[:, :-1]], -1)))
        out = logits_fn(model, rows, 1000 + step, all_logits=True, route_credit=rc, **kw)[..., 0]
        loss = F.mse_loss(out[:, a.warmup:], y[:, a.warmup:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); schedule.step()
        losses.append(float(loss))
    model.eval()
    stepper = SparseStepper(model, 1, 777, deterministic=a.argmax)
    with torch.no_grad():
        for i in range(TRAIN):                            # teacher-forced warm pass over the training inputs
            pred = stepper.step(torch.tensor([float(i)]), torch.tensor(taps_of(z, i, a.taps))[None])
        if a.delta:
            pred = pred + float(z[TRAIN - 1])
        history = list(z[:TRAIN])
        preds = []
        for i in range(TRAIN, TRAIN + TEST):              # autonomous: the newest input is the previous prediction
            history.append(float(pred[0, 0]))
            pred = stepper.step(torch.tensor([float(i)]), torch.tensor(taps_of(np.array(history), i, a.taps))[None])
            if a.delta:
                pred = pred + history[-1]
            preds.append(float(pred[0, 0]))
    return np.array(preds), losses, sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--tau', type=int, default=17)
    p.add_argument('--repeats', type=int, default=30); p.add_argument('--first-repeat', type=int, default=0)
    p.add_argument('--payload', type=int, default=16); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--tie-pools', action='store_true'); p.add_argument('--taps', type=int, default=1)
    p.add_argument('--segment', type=int, default=64); p.add_argument('--lanes', type=int, default=32)
    p.add_argument('--steps', type=int, default=1000); p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--clip', type=float, default=1.); p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--route-credit', choices=('none', 'linear'), default='linear')
    p.add_argument('--compiled', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--closed-loop', type=int, default=0, help='closed-loop training: positions >= this use the model\'s own '
                   'previous prediction as the newest tap (gradients through it); 0 = teacher forcing only (needs --compiled)')
    p.add_argument('--delta', action='store_true', help='predict the increment z[t+1] - z[t]')
    p.add_argument('--argmax', action='store_true', help='deterministic routing at inference (highest score wins)')
    p.add_argument('--closed-from', type=float, default=.5, help='fraction of training after which closed-loop windows start')
    a = p.parse_args()
    if a.closed_loop and not a.compiled:
        raise ValueError('--closed-loop needs --compiled')
    out = ROOT / 'experiments/results/neurobench_mg' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); started = time.perf_counter()
    rows = []
    for r in range(a.first_repeat, a.first_repeat + a.repeats):
        raw = official_slice(a.tau, r)
        mu, sd = raw[:TRAIN + 1].mean(), raw[:TRAIN + 1].std()
        z = ((raw - mu) / sd).astype(np.float32)
        t = time.perf_counter()
        preds, losses, params = fit_and_forecast(a, z, a.seed * 1000 + r)
        forecast = preds * sd + mu
        score = smape(forecast, raw[TRAIN + 1:TRAIN + TEST + 1])
        rows.append(dict(repeat=r, smape=score, final_train_mse=float(np.mean(losses[-20:])), wall_s=time.perf_counter() - t))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(status='completed', args=vars(a), parameters=params, footprint_bytes_float32=params * 4,
                  repeats=rows, mean_smape=float(np.mean([r['smape'] for r in rows])),
                  protocol='NeuroBench Mackey-Glass (neurobench 2.3.0 slices, split and SMAPE); autonomous 750-step forecast',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/mackey_glass_native.py', 'sleeping_machines/sparse_inference.py',
                                  'sleeping_machines/batched_episodes.py', 'sleeping_machines/compiled_episodes.py')},
                  data_sha256=hashlib.sha256((DATA / f'mg_{a.tau}.npy').read_bytes()).hexdigest(),
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mean_smape=result['mean_smape'], parameters=params)), flush=True)


if __name__ == '__main__':
    main()
