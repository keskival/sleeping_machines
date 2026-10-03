"""Exhaustive finite-dimensional contracts for residual replay control variates.
Pure standard library; no model, data, optimizer or training execution.
"""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify(targets, anchors, probabilities, costs):
    dimension = len(targets[0])
    residuals = [[x-y for x,y in zip(t,a)] for t,a in zip(targets,anchors)]
    anchor_sum = [sum(a[d] for a in anchors) for d in range(dimension)]
    truth = [sum(t[d] for t in targets) for d in range(dimension)]
    mean = [0.] * dimension
    second = [[0.] * dimension for _ in range(dimension)]
    expected_cost = 0.
    for bits in itertools.product((0,1), repeat=len(targets)):
        mass = 1.
        estimate = list(anchor_sum)
        cost = 0.
        for flag,p,residual,c in zip(bits, probabilities, residuals, costs):
            assert 0 < p <= 1
            mass *= p if flag else 1-p
            if flag:
                estimate = [x+r/p for x,r in zip(estimate,residual)]
                cost += c
        expected_cost += mass*cost
        for d in range(dimension):
            mean[d] += mass*estimate[d]
            for e in range(dimension):
                second[d][e] += mass*(estimate[d]-truth[d])*(estimate[e]-truth[e])
    formula = [[sum((1/p-1)*r[d]*r[e] for p,r in zip(probabilities,residuals))
                for e in range(dimension)] for d in range(dimension)]
    errors = [abs(x-y) for x,y in zip(mean,truth)]
    errors += [abs(second[d][e]-formula[d][e]) for d in range(dimension) for e in range(dimension)]
    errors += [abs(expected_cost-sum(p*c for p,c in zip(probabilities,costs)))]
    assert max(errors) < 1e-10
    return dict(mean=mean, target=truth, covariance=second,
                covariance_trace=sum(second[d][d] for d in range(dimension)),
                incremental_expected_cost=expected_cost, max_error=max(errors))


if __name__ == '__main__':
    targets = [[2.,1.],[-1.,3.],[.5,-2.]]
    probabilities = [.25,.5,.75]
    costs = [1.,2.,4.]
    zero = [[0.,0.]]*3
    good = [[1.8,.9],[-.9,2.7],[.45,-1.8]]
    bad = [[20.,10.],[-10.,30.],[5.,-20.]]
    output = dict(scope='Constructed exact conditional-gradient estimator contracts, not trained-model evidence. Anchor computation cost excluded here and must be added in admission.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  no_anchor=verify(targets,zero,probabilities,costs),
                  accurate_anchor=verify(targets,good,probabilities,costs),
                  poor_anchor=verify(targets,bad,probabilities,costs),
                  exact_anchor=verify(targets,targets,probabilities,costs))
    assert abs(output['accurate_anchor']['covariance_trace']/output['no_anchor']['covariance_trace']-.01)<1e-12
    assert abs(output['poor_anchor']['covariance_trace']/output['no_anchor']['covariance_trace']-81)<1e-12
    assert output['exact_anchor']['covariance_trace']==0
    path=ROOT/'experiments/results/diagnostics/aws_residual_credit_algebra_20261003.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))
