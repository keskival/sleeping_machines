"""Check the actual protected-bank alias and terminal cold-start derivation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import joint_outcome_benchmark as J
from balanced_joint_protocol import joint_examples


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    a = p.parse_args(); out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    started = time.perf_counter(); torch.set_num_threads(1)
    model = J.make_model(J.parser().parse_args(['--tag', 'contract', '--payload', '4', '--depth', '2'])).double()
    rows = joint_examples(groups=4, seed=75103); banks = []; model.eval()
    for row in rows:
        with torch.no_grad(): _, state = model.forward_chunk(row['inputs'])
        banks.append(state.outcomes)
    for group in range(4):
        for first in (0, 1):
            pair = [i for i, row in enumerate(rows) if row['group'] == group and row['bits'][0] == first]
            hist = [Counter(banks[i].values()) for i in pair]
            assert len(pair) == 2 and hist[0] == hist[1]
            laws = [Counter({(x, y): nx * ny for x, nx in h.items() for y, ny in h.items()}) for h in hist]
            assert laws[0] == laws[1]
    # Fixed zero-logit context: the actual generic decoder has no balanced
    # relation gradient under uniform value reads. This excludes core credit.
    risk = 0.; score_grads = []
    for row, bank in zip(rows, banks):
        raw = torch.tensor(list(bank.values())); values = model.values(raw)
        pair = model.pair_logits(torch.zeros(8, dtype=torch.float64), torch.tensor(0., dtype=torch.float64), values)
        first = torch.zeros(len(raw), dtype=torch.float64, requires_grad=True)
        second = torch.zeros_like(first, requires_grad=True)
        loss = (first.softmax(0)[:, None] * second.softmax(0)[None, :] *
                F.softplus((1 - 2 * row['target']) * pair)).sum()
        score_grads.extend(torch.autograd.grad(loss, (first, second), retain_graph=True))
        risk = risk + loss / len(rows)
    risk.backward()
    max_decoder_gradient = max(float(x.grad.abs().max()) for x in
        [model.context_residual.weight, model.context_residual.bias, model.linear_values, model.interaction])
    max_score_gradient = max(float(x.abs().max()) for x in score_grads)
    assert max_decoder_gradient < 1e-12 and max_score_gradient < 1e-12
    # Target-independent marker policies and signed value map are an analytic
    # existence witness only; they are never used by the fitted benchmark.
    risks = []
    for row, bank in zip(rows, banks):
        addresses = list(bank); raw = torch.tensor(list(bank.values()))
        signed = torch.where(raw == 0, -1., torch.where(raw == 1, 1., 0.)).double()
        policies = []
        for marker in (24, 25):
            score = torch.full((len(raw),), -12., dtype=torch.float64)
            score[addresses.index(marker)] = 12.; policies.append(score.softmax(0))
        logits = -20 * signed[:, None] * signed[None, :]
        risks.append(float((policies[0][:, None] * policies[1][None, :] *
                            F.softplus((1 - 2 * row['target']) * logits)).sum()) / math.log(2))
    assert max(risks) < 1e-6
    names = [*J.sources(), 'experiments/uniform_address_contracts.py', 'experiments/theory/70_uniform_address_aliasing.md']
    result = dict(status='completed', args=vars(a), targets=len(rows), groups=4,
        actual_bank_uniform_pair_law_alias_verified=True, fixed_context_stationary_risk_bits=float(risk.detach()) / math.log(2),
        maximum_decoder_gradient=max_decoder_gradient, maximum_policy_gradient=max_score_gradient,
        analytic_marker_witness_max_expected_bits=max(risks), wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        scope='Actual bank distributional alias, fixed-uninformative-context terminal stationary point, analytic target-free marker witness. '
              'Not whole-core stationarity, a fitted score, semantic depth or a changed benchmark.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
