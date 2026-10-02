"""Training-only critic shrinkage and actual shared-parameter variance audit."""
import argparse
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import aws_coarse_native as A
import dvs_critic_le_benchmark as CR
import dvs_local_expectation_benchmark as LE
import dvs_native_benchmark as N


def credit(pi, utility):
    return pi * (utility - (pi * utility).sum(-1, keepdim=True))


def variance(vectors, k):
    r = len(vectors)
    centered = vectors - vectors.mean(0)
    return float(r*(r-k)/(k*(r-1)) * centered.square().sum())


def contract():
    v = torch.tensor([[1., 3.], [-2., 4.], [5., 7.], [2., -1.]], dtype=torch.float64)
    for k in (1, 2, 4):
        errors = torch.stack([(4/k)*v[list(s)].sum(0)-v.sum(0) for s in itertools.combinations(range(4), k)])
        torch.testing.assert_close(errors.mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        assert abs(float(errors.square().sum(1).mean())-variance(v, k)) < 1e-12
    return 'exhaustive shared-coordinate finite-population variance verified'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    torch.set_num_threads(1); started = time.perf_counter()
    parent = ROOT / 'experiments/results/diagnostics/aws_signed_replay_variance_20261002T230400Z.json'
    old = json.loads(parent.read_text())
    assert old['status'] == 'completed'
    for name, digest in old['source_sha256'].items():
        assert N.sha(ROOT/name) == digest, name
    artifact = parent.with_suffix('.pt'); assert N.sha(artifact) == old['artifact_sha256']
    banks = torch.load(artifact, weights_only=False, map_location='cpu')
    args = CR.parser().parse_args(['--tag', a.tag, '--data', old['provenance']['data_result'],
        '--controls', old['provenance']['controls'], '--fit', '64', '--dev', '1'])
    args.bins = 4; args.clock_step = .25
    rows, _, provenance = A.load(args)
    results = []
    for seed in (7, 8):
        bank = banks[str(seed)]
        net = nn.Sequential(nn.Linear(50, 32), nn.GELU(), nn.Linear(32, 1))
        net.load_state_dict(bank['critic']); net.eval()
        vectors = []
        with torch.no_grad():
            for feats, pi, utility in bank['examples']:
                q = net(feats.reshape(-1, 50))[:, 0].double().reshape_as(utility)
                vectors.append((credit(pi, utility), credit(pi, q)))
        train_g = torch.cat([g.flatten() for g, q in vectors[:32]])
        train_q = torch.cat([q.flatten() for g, q in vectors[:32]])
        alpha = float(((train_g*train_q).sum()/train_q.square().sum()).clamp(0, 1))
        score_results = {}
        for name, scale in [('raw', 1.), ('training_shrinkage', alpha), ('zero', 0.)]:
            mse4 = sum((len(g)/4-1)*float(g.square().sum()) for g,q in vectors[32:])
            ratios = {str(k):sum((len(g)/k-1)*float((g-scale*q).square().sum()) for g,q in vectors[32:])/mse4 for k in (1,2)}
            score_results[name] = dict(alpha=scale, heldout_score_mse_ratios=ratios)
        args.seed = seed; model = LE.make_model(args, fast=True)
        checkpoint = ROOT/f'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.progress.pt'
        expected = next(r['checkpoint_sha256'] for r in old['results'] if r['seed'] == seed)
        assert N.sha(checkpoint) == expected
        model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location='cpu')['best_state']); model.train()
        parameters = tuple(model.parameters()); parameter_results = []
        for index in (32, 33):
            _, races, _, scores, _, _ = CR.run_with_values(model, rows[index], 100000+seed, False)
            _, pi, _ = bank['examples'][index]
            torch.testing.assert_close(torch.stack([s.detach().double().softmax(0) for s in scores]), pi, rtol=0, atol=0)
            g, q = vectors[index]; full, predicted = [], []
            for r in range(races):
                for source, target in [(g, full), (q, predicted)]:
                    grads = torch.autograd.grad(scores[r], parameters, grad_outputs=source[r].to(scores[r].dtype), retain_graph=True, allow_unused=True)
                    target.append(torch.cat([(torch.zeros_like(p) if x is None else x).detach().double().flatten() for p,x in zip(parameters, grads)]))
            full = torch.stack(full); predicted = torch.stack(predicted)
            baseline = variance(full, 4)
            parameter_results.append(dict(fit_index=index, parameter_coordinates=full.shape[1], races=races,
                plain_k4_variance=baseline,
                variants={name:{str(k):variance(full-scale*predicted,k)/baseline for k in (1,2)}
                    for name,scale in [('raw',1.),('training_shrinkage',alpha),('zero',0.)]},
                cross_site_coherence=float(full.sum(0).square().sum()/full.square().sum())))
        results.append(dict(seed=seed, alpha_fit_only=alpha, score_results=score_results, parameter_results=parameter_results,
            score_nomination_passed={str(k):score_results['training_shrinkage']['heldout_score_mse_ratios'][str(k)]<=1 for k in (1,2)}))
    out = ROOT/'experiments/results/diagnostics'/f'{a.tag}.json'; assert not out.exists()
    result = dict(status='completed',tag=a.tag,contract=contract(),results=results,provenance=provenance,
        parent_sha256=N.sha(parent),artifact_sha256=N.sha(artifact),
        source_sha256={**old['source_sha256'],'experiments/aws_replay_calibration_probe.py':N.sha(Path(__file__))},
        scope='Training-only scalar calibration;32 heldout FIT prefixes for score variance;first2 for actual shared-parameter route variance. Conditional sampling variance only, no quality or whole-risk exactness claim.',
        diagnostic_flops='unknown; original producer fits and cached replay/critic work retained, all new vector-Jacobian work additional',
        wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
