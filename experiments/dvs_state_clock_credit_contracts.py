"""Joint-time analytic witnesses, unchanged native forwards and driver recovery."""
import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
import dvs_state_clock_credit_benchmark as S
import dvs_native_contracts as C


def analytic_contracts():
    nodes, weights = np.polynomial.laguerre.laggauss(32)
    for raw in ([0., 0.], [-.8, .7], [.5, -.2]):
        scores = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        rates = scores.exp(); total = rates.sum(); pi = rates/total
        intercept = scores.new_tensor([.7, 1.3]); slope = scores.new_tensor([.2, .6])
        expected = (pi*(intercept+slope/total)).sum()
        exact = torch.autograd.grad(expected, scores, retain_graph=True)[0]
        times = torch.tensor(nodes, dtype=torch.float64)/total.detach()
        losses = intercept[None]+slope[None]*times[:, None]
        estimated = (torch.tensor(weights)[:, None]*S.joint_credit(
            scores.detach()[None].expand(len(nodes), -1), times, losses, times.new_full(times.shape, .9))).sum(0)
        torch.testing.assert_close(estimated, exact, rtol=1e-11, atol=1e-12)
        # Integrate a genuine downstream time jump using shifted exponential quadrature.
        boundary = .7
        jump_risk = pi[1]*torch.exp(-total*boundary)
        exact_jump = torch.autograd.grad(jump_risk, scores)[0]
        shifted = boundary+times
        jump_losses = shifted.new_tensor([0., 1.]).expand(len(nodes), -1)
        score = S.joint_credit(scores.detach()[None].expand(len(nodes), -1), shifted,
            jump_losses, shifted.new_zeros(shifted.shape))
        estimated_jump = torch.exp(-total.detach()*boundary)*(torch.tensor(weights)[:, None]*score).sum(0)
        torch.testing.assert_close(estimated_jump, exact_jump, rtol=1e-11, atol=1e-12)
        if raw == [0., 0.]:
            assert exact_jump[1] < 0
            assert (pi[1]*(1-pi[1])*torch.exp(-total*boundary)) > 0


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    p.add_argument('--data', required=True); p.add_argument('--controls', required=True)
    a = p.parse_args(); started = time.perf_counter(); torch.set_num_threads(1)
    out = ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name != a.tag or out.exists():raise ValueError('Unused plain tag required')
    analytic_contracts()
    common = ['--data', a.data, '--controls', a.controls, '--fit', '4', '--dev', '2',
        '--epochs', '2', '--update-targets', '2']
    config = S.N.parser().parse_args(['--tag', 'contract']+common)
    fitting, _, metadata = S.N.load(config); rows = fitting[:4]
    model = S.N.make_model(config).double(); seed = 1237
    with torch.no_grad():base, original, _ = S.K.forward(model, rows, seed)
    loss, state, detail = S.corrected_loss(model, rows, seed, 1)
    torch.testing.assert_close(base, detail['logits'], rtol=0, atol=0)
    for key in ('memories', 'arrivals', 'seen', 'context', 'context_times', 'winners'):C.equal(original[key], state[key])
    factual, shadow = detail['record'], detail['shadow']
    torch.testing.assert_close(factual['scores'], shadow['scores'], rtol=0, atol=0)
    torch.testing.assert_close(factual['values'], shadow['values'], rtol=0, atol=0)
    torch.testing.assert_close(factual['first'], shadow['first'], rtol=0, atol=0)
    factual['scores'].retain_grad()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # The replacement is the independent sum over legal branch likelihood scores,
    # including a nonzero common-clock component, not a conserved choice gradient.
    scores = factual['scores'].detach()[:, detail['head']]; rates = scores.exp(); pi = rates/rates.sum(-1, keepdim=True)
    independent = torch.zeros_like(scores)
    for option in range(2):
        indicator = F.one_hot(torch.full((len(rows),), option), 2).to(scores.dtype)
        independent += pi[:, option, None]*(detail['outcome_losses'][:, option]-detail['baseline'])[:, None]*(indicator-rates*factual['first'][:, None])
    torch.testing.assert_close(detail['credit'], independent, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(factual['scores'].grad[:, detail['head']], independent, rtol=1e-12, atol=1e-12)
    with tempfile.TemporaryDirectory(prefix='dvs-state-clock-') as temp:
        continuous = S.N.parser().parse_args(['--tag', 'continuous']+common)
        first = S.run(continuous, temp)
        for sample in first['work_samples']:
            for stage in sample['stages'].values():
                assert stage['formula_coverage_complete'], stage['unsupported_floating_operators']
        recovered = S.N.parser().parse_args(['--tag', 'recovered']+common)
        recovered.stop_after_updates = 1; S.run(recovered, temp)
        recovered.stop_after_updates = None; recovered.resume = True; second = S.run(recovered, temp)
        x = torch.load(Path(temp)/'continuous.progress.pt', weights_only=False)
        y = torch.load(Path(temp)/'recovered.progress.pt', weights_only=False)
        for name in ('online_model', 'optimizer', 'best_state', 'best', 'cursor', 'torch_rng'):C.equal(x[name], y[name])
        for name in ('final', 'activity', 'work', 'work_samples', 'selected_epoch', 'credit_protocol'):C.equal(first[name], second[name])
    result = dict(status='completed', args=vars(a), contracts_passed=4, data=metadata,
        smooth_joint_clock_integral_matches_analytic_gradient=True,
        discontinuous_timing_jump_matches_analytic_gradient=True,
        factual_forward_and_every_state_unchanged=True, actual_alternative_prefix_and_first_time_match=True,
        actual_model_adam_cursor_rng_recovery=True, formula_coverage_complete=True,
        source_sha256={**S.sources(), 'experiments/dvs_native_contracts.py':S.N.sha(ROOT/'experiments/dvs_native_contracts.py')},
        wall_s=time.perf_counter()-started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='One isolated node joint likelihood component is exact in expectation under fixed entering-prefix assumptions. Full native model retains other local teachers. No measured quality, variance or advantage claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':main()
