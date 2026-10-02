"""Frozen, fitting-only score-coordinate variance screen for corrected replay critics.

This measures conditional subsampling variance, not parameter-gradient variance
or prediction quality. No producer update, development selection, or test access.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_critic_le_benchmark as CR
import dvs_local_expectation_benchmark as LE
import dvs_native_benchmark as N
import aws_coarse_native as A
from sleeping_machines.shadow_lanes import shadow_losses


def credit(pi, utility):
    return pi * (utility - (pi * utility).sum(-1, keepdim=True))


def finite_sampling_contract():
    pi = torch.tensor([[.2, .8], [.7, .3], [.4, .6], [.6, .4]], dtype=torch.float64)
    g = credit(pi, torch.arange(8, dtype=torch.float64).reshape(4, 2))
    q = credit(pi, torch.tensor([[3., -1.], [2., 7.], [5., 1.], [-2., 4.]], dtype=torch.float64))
    for k in (1, 2, 4):
        errors = []
        for selected in itertools.combinations(range(4), k):
            estimate = q.clone(); estimate[list(selected)] += (4 / k) * (g - q)[list(selected)]
            errors.append(estimate - g)
        errors = torch.stack(errors)
        torch.testing.assert_close(errors.mean(0), torch.zeros_like(g), atol=1e-14, rtol=0)
        torch.testing.assert_close(errors.square().sum((1, 2)).mean(), (4 / k - 1) * (g-q).square().sum())
    return 'exhaustive fixed-critic unbiasedness and exact score-space MSE pass'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    torch.set_num_threads(1); started = time.perf_counter()
    args = CR.parser().parse_args(['--tag', a.tag, '--data', 'experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json',
        '--controls', 'experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json', '--fit', '64', '--dev', '1', '--seed', '7'])
    args.bins = 4; args.clock_step = .25
    rows, _, provenance = A.load(args)
    results = []; banks = {}
    for seed in (7, 8):
        args.seed = seed; model = LE.make_model(args, fast=True)
        checkpoint = ROOT / f'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.progress.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['best_state']); model.train()
        examples = []; replay_wall = time.perf_counter()
        for row in rows:
            loss, races, state, scores, values, _ = CR.run_with_values(model, row, 100000 + seed, False)
            with torch.no_grad():
                forces = [(r, i) for r in range(races) for i in range(model.pool)]
                losses = shadow_losses(model, row, 100000 + seed, forces).reshape(races, model.pool)
                label = torch.nn.functional.one_hot(torch.tensor(row['target']), 11).float().expand(model.pool, -1)
                feats = torch.stack([torch.cat([CR.features(scores[r], values[r], r, races, (r // model.heads) % model.depth, model.depth), values[r].detach(), values[r].detach()-values[r].detach().mean(0), label], -1) for r in range(races)])
                pi = torch.stack([s.detach().double().softmax(0) for s in scores])
                # Full lane/fork comparisons are separately guarded prerequisites.
                examples.append((feats, pi, losses - losses.mean(-1, keepdim=True)))
        replay_s = time.perf_counter() - replay_wall
        torch.manual_seed(seed + 99)
        critic = nn.Sequential(nn.Linear(5 + model.depth + 2*model.payload + 11, 32), nn.GELU(), nn.Linear(32, 1))
        optimizer = torch.optim.Adam(critic.parameters(), lr=.003)
        x = torch.cat([e[0].reshape(-1, 5 + model.depth + 2*model.payload + 11) for e in examples[:32]])
        y = torch.cat([e[2].reshape(-1).float() for e in examples[:32]])
        start_fit = time.perf_counter()
        for _ in range(100):
            optimizer.zero_grad(); torch.nn.functional.mse_loss(critic(x)[:, 0], y).backward(); optimizer.step()
        critic_s = time.perf_counter() - start_fit
        holdout = []
        with torch.no_grad():
            for feats, pi, utility in examples[32:]:
                q = critic(feats.reshape(-1, 5 + model.depth + 2*model.payload + 11))[:, 0].double().reshape_as(utility)
                q -= q.mean(-1, keepdim=True)
                g = credit(pi, utility); predicted = credit(pi, q); R = len(pi)
                baseline = float(g.square().sum()); residual = float((g - predicted).square().sum())
                denom = float((utility - utility.mean()).square().sum())
                holdout.append(dict(races=R, critic_r2=1-float((utility-q).square().sum())/denom,
                    route_score_residual_ratio=residual / baseline,
                    score_coordinate_mse={f'critic_k{k}': (R/k-1)*residual for k in (1,2)},
                    no_critic_k4_mse=(R/4-1)*baseline))
        mse4 = sum(h['no_critic_k4_mse'] for h in holdout)
        ratios = {str(k):sum(h['score_coordinate_mse'][f'critic_k{k}'] for h in holdout)/mse4 for k in (1,2)}
        results.append(dict(seed=seed, checkpoint_sha256=N.sha(checkpoint), heldout_fit_rows=holdout, critic_vs_plain_k4_mse_ratio=ratios,
            variance_nomination_passed={k:v <= 1. for k,v in ratios.items()}, replay_wall_s=replay_s, critic_fit_wall_s=critic_s,
            replay_lanes=sum(len(e[1])*model.pool for e in examples), critic_updates=100))
        banks[str(seed)] = dict(critic=critic.state_dict(), examples=[tuple(t.cpu() for t in e) for e in examples])
    out = ROOT / 'experiments/results/diagnostics' / a.tag
    assert not out.with_suffix('.json').exists(), 'Unique output required'
    torch.save(banks, out.with_suffix('.pt'))
    sources = {**CR.sources(), 'experiments/aws_signed_replay_variance_probe.py': N.sha(Path(__file__)), 'experiments/aws_coarse_native.py': N.sha(ROOT / 'experiments/aws_coarse_native.py'),
        'sleeping_machines/factorized_race.py': N.sha(ROOT / 'sleeping_machines/factorized_race.py')}
    result = dict(status='completed', tag=a.tag, contract=finite_sampling_contract(), results=results,
        protocol='Saved fixed-pass trained native coarse4/.25-clock encoders seeds7/8; first32 FIT rows train signed-value/label-aware critic, next32 FIT rows holdout;100 fixed critic updates; no producer updates; no dev quality evaluated',
        scope='Exact conditional score-coordinate subsampling variance only; cross-site parameter-gradient covariance and learning quality not established',
        fitting_flops='unknown; diagnostic replay and critic training are not free', provenance=provenance, source_sha256=sources,
        artifact_sha256=N.sha(out.with_suffix('.pt')), wall_s=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['results'], indent=2))


if __name__ == '__main__':
    main()
