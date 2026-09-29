"""Exact joint clock/mark credit contracts; no dataset or training claims."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--samples', type=int, default=300000)
    args = ap.parse_args()
    if Path(args.tag).name != args.tag or args.samples < 10000:
        raise ValueError('Invalid contract settings')
    out = Path('experiments/results/e132') / (args.tag + '.json')
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    start = time.perf_counter()
    logits = np.array([.4, -.3, .1])
    p = softmax(logits)
    rate, deadline, nonresponse = 1.3, .9, 2.1
    cost = np.array([.2, 1.7, .8])
    a, survival = rate * deadline, np.exp(-rate * deadline)
    mean_cost = p @ cost
    risk = (1 - survival) * mean_cost + survival * nonresponse
    exact = np.r_[(1 - survival) * p * (cost - mean_cost),
                  a * survival * (mean_cost - nonresponse)]

    def objective(x):
        probs, r = softmax(x[:3]), np.exp(x[3])
        s = np.exp(-r * deadline)
        return (1 - s) * (probs @ cost) + s * nonresponse

    point = np.r_[logits, np.log(rate)]
    step = 1e-5
    finite = np.array([(objective(point + step * np.eye(4)[j]) -
                       objective(point - step * np.eye(4)[j])) / (2 * step)
                      for j in range(4)])
    assert np.max(np.abs(finite - exact)) < 1e-9

    # Integrate the fired density exactly enough for gradient/Fisher checks.
    nodes, weights = np.polynomial.legendre.leggauss(80)
    times, quadrature = (nodes + 1) * deadline / 2, weights * deadline / 2
    firing_weights = quadrature * rate * np.exp(-rate * times)
    scores = np.eye(3) - p
    gradient = np.zeros(4)
    fisher = np.zeros((4, 4))
    for k in range(3):
        joint_score = np.column_stack((np.tile(scores[k], (len(times), 1)), 1 - rate * times))
        probability = firing_weights * p[k]
        gradient += (probability * cost[k]) @ joint_score
        fisher += np.einsum('n,ni,nj->ij', probability, joint_score, joint_score)
    silent_score = np.r_[np.zeros(3), -a]
    gradient += survival * nonresponse * silent_score
    fisher += survival * np.outer(silent_score, silent_score)
    expected_fisher = np.zeros((4, 4))
    expected_fisher[:3, :3] = (1 - survival) * (np.diag(p) - np.outer(p, p))
    expected_fisher[3, 3] = 1 - survival
    assert np.max(np.abs(gradient - exact)) < 1e-12
    assert np.max(np.abs(fisher - expected_fisher)) < 1e-12
    assert objective(point - .1 * exact) < risk

    rng = np.random.default_rng(132)
    t = rng.exponential(1 / rate, args.samples)
    k = rng.choice(3, args.samples, p=p)
    fired = t < deadline
    observed_cost = np.where(fired, cost[k], nonresponse)
    local_score = np.zeros((args.samples, 4))
    local_score[fired, :3] = scores[k[fired]]
    local_score[:, 3] = np.where(fired, 1 - rate * t, -a)
    estimates = (observed_cost - risk)[:, None] * local_score
    mc_mean = estimates.mean(0)
    mc_se = estimates.std(0, ddof=1) / np.sqrt(args.samples)
    assert np.all(np.abs(mc_mean - exact) < 6 * mc_se + 1e-6)
    rb = np.zeros_like(estimates)
    rb[fired, :3] = p * (cost - mean_cost)
    rb[:, 3] = (np.where(fired, mean_cost, nonresponse) - risk) * local_score[:, 3]
    assert np.all(rb.var(0) <= estimates.var(0) + 1e-12)
    # A deadline step has zero ordinary sampled time derivative a.e., but not
    # zero expected derivative. Closed-form integral supplies the clock score.
    c = .7
    discontinuous_exact = -rate * c * np.exp(-rate * c)
    boundary_estimates = (t > c) * (1 - rate * t)
    boundary_se = boundary_estimates.std(ddof=1) / np.sqrt(args.samples)
    assert abs(boundary_estimates.mean() - discontinuous_exact) < 6 * boundary_se

    # A three-level hard-choice tree: all suffix probabilities change with
    # realized ancestors. Enumerate the actual conditional law, not a local
    # value tangent. All 27 leaves contribute through the same three logits.
    def tree(theta, credit=False):
        value, grad, mass = 0., np.zeros(3), 0.
        for path in itertools.product(range(3), repeat=3):
            history, probability, score = [], 1., np.zeros(3)
            for choice in path:
                feature = 1 + .2 * sum(history)
                probs = softmax(theta * feature + np.array([.0, .1 * len(history), -.2]))
                probability *= probs[choice]
                score += feature * (np.eye(3)[choice] - probs)
                history.append(choice)
            # Discrete interaction, intentionally not additive across layers.
            loss = .1 + 1.3 * (path[0] != path[2]) + .8 * ((sum(path) % 3) != 1)
            mass += probability
            value += probability * loss
            grad += probability * loss * score
        assert abs(mass - 1) < 1e-12
        return (value, grad) if credit else value

    tree_value, tree_gradient = tree(logits, True)
    tree_finite = np.array([(tree(logits + step * np.eye(3)[j]) -
                             tree(logits - step * np.eye(3)[j])) / (2 * step) for j in range(3)])
    assert np.max(np.abs(tree_gradient - tree_finite)) < 1e-9
    result = {'status': 'completed', 'args': vars(args), 'joint_risk': float(risk),
              'exact_gradient': exact.tolist(), 'finite_difference_gradient': finite.tolist(),
              'quadrature_gradient': gradient.tolist(),
              'gradient_max_error': float(np.max(np.abs(finite - exact))),
              'fisher_max_error': float(np.max(np.abs(fisher - expected_fisher))),
              'firing_probability': float(1 - survival), 'fisher': fisher.tolist(),
              'monte_carlo_gradient': mc_mean.tolist(), 'monte_carlo_standard_error': mc_se.tolist(),
              'sample_score_variance': estimates.var(0).tolist(),
              'counterfactual_conditioned_variance': rb.var(0).tolist(),
              'discontinuous_clock_exact': float(discontinuous_exact),
              'discontinuous_clock_sample_mean': float(boundary_estimates.mean()),
              'sample_pathwise_clock_derivative': 0.,
              'deep_tree': {'depth': 3, 'alternatives': 3, 'leaves': 27, 'risk': tree_value,
                            'gradient': tree_gradient.tolist(), 'finite_difference': tree_finite.tolist(),
                            'max_error': float(np.max(np.abs(tree_gradient - tree_finite)))},
              'hardware': {'platform': platform.platform(), 'numpy': np.__version__, 'device': 'cpu'},
              'wall_s': time.perf_counter() - start,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'energy_joules': None,
              'scope': 'Synthetic analytic/Monte Carlo contracts. No language/SHD model trained. Expected-gradient correctness is not a finite hard-loss or scaling guarantee.',
              'source_sha256': {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('gradient_max_error', 'fisher_max_error', 'firing_probability', 'deep_tree', 'wall_s', 'peak_rss_kib')}), flush=True)


if __name__ == '__main__':
    main()
