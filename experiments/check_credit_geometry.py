"""Static checks for theory139's identities; standard library, no model runtime."""
from fractions import Fraction as F
from itertools import combinations
import json
import math


def clip(g):
    radius = math.sqrt(sum(x*x for x in g))
    factor = min(1., 1./(radius+1e-6))
    return [factor*x for x in g]


def clip_jacobian(g):
    radius = math.sqrt(sum(x*x for x in g))
    if radius+1e-6 < 1:
        return [[float(i == j) for j in range(len(g))] for i in range(len(g))]
    return [[(float(i == j)-g[i]*g[j]/(radius*(radius+1e-6)))/(radius+1e-6)
             for j in range(len(g))] for i in range(len(g))]


def finite_difference(fn, g):
    step = 1e-5
    jacobian = [[0.]*len(g) for _ in g]
    for j in range(len(g)):
        plus, minus = list(g), list(g)
        plus[j] += step
        minus[j] -= step
        a, b = fn(plus), fn(minus)
        for i in range(len(g)):
            jacobian[i][j] = (a[i]-b[i])/(2*step)
    return jacobian


def check():
    errors = []

    def equal_jacobians(a, b):
        error = max(abs(x-y) for row, other in zip(a, b) for x, y in zip(row, other))
        assert error < 2e-8, error
        errors.append(error)

    for g in ([.1, -.2, .05], [3., -4., .5]):
        equal_jacobians(clip_jacobian(g), finite_difference(clip, g))

    # Heterogeneous coordinate histories, including a positive Adam sensitivity.
    moments = [.001, -.02, .004]
    second = [1e-6, .0005, .00008]
    steps = [13, 5, 24]

    def adam(h):
        return [-.003*((.9*m+.1*z)/(1-.9**t)) /
                (math.sqrt((.999*v+.001*z*z)/(1-.999**t))+1e-8)
                for m, v, z, t in zip(moments, second, h, steps)]

    def adam_diagonal(h):
        output = []
        for m, v, z, t in zip(moments, second, h, steps):
            a, b = .1/(1-.9**t), .001/(1-.999**t)
            numerator = (.9*m+.1*z)/(1-.9**t)
            denominator = math.sqrt((.999*v+.001*z*z)/(1-.999**t))
            output.append(-.003*(a/(denominator+1e-8) - numerator*b*z /
                                (denominator*(denominator+1e-8)**2)))
        return output

    g = [3., -4., .5]
    diagonal = adam_diagonal(clip(g))
    combined = [[diagonal[i]*x for x in row] for i, row in enumerate(clip_jacobian(g))]
    equal_jacobians(combined, finite_difference(lambda x: adam(clip(x)), g))
    assert adam_diagonal([.7, -.2, .1])[0] > 0

    # Rational arithmetic establishes the convex ascent example without rounding.
    floor, eps = F(1, 1000000), F(1, 100000000)
    probabilities, raw = [F(1, 10), F(9, 10)], [F(10), F(-1)]
    theta, rate = F(1, 10), F(3, 1000)
    assert sum(p*x for p, x in zip(probabilities, raw)) == theta
    clipped = [x*min(F(1), F(1)/(abs(x)+floor)) for x in raw]
    assert sum(p*x for p, x in zip(probabilities, clipped)) < 0
    for values in (clipped, raw):
        movements = [-rate*x/(abs(x)+eps) for x in values]
        assert sum(p*x for p, x in zip(probabilities, movements)) > 0
        assert sum(p*(theta*x+x*x/2) for p, x in zip(probabilities, movements)) > 0

    # Exhaust all subsets after an anisotropic linear map, independently of the formula.
    vectors = [(F(1), F(3)), (F(-2), F(1)), (F(4), F(-1))]
    count = len(vectors)
    mean = [sum(row[i] for row in vectors)/count for i in range(2)]
    centered = [[row[i]-mean[i] for i in range(2)] for row in vectors]
    jacobian = ((F(2), F(1)), (F(0), F(3)))
    transformed = [[sum(jacobian[i][j]*row[j] for j in range(2))
                    for i in range(2)] for row in centered]
    total = sum(x*x for row in transformed for x in row)
    for k in (1, 2, 3):
        subsets = list(combinations(range(count), k))
        actual = sum(sum((F(count, k)*sum(transformed[r][i] for r in subset))**2
                         for i in range(2)) for subset in subsets)/len(subsets)
        expected = F(count*(count-k), k*(count-1))*total
        assert actual == expected

    # Note140: identical messages can hide a consequential persistent write.
    probability = .3
    q0, q1 = math.log1p(math.exp(-1)), math.log(2)
    exact_choice = probability*(1-probability)*(q0-q1)
    assert exact_choice < 0
    message_only = probability*(1-probability)*(-.5)*(3.-3.)
    assert message_only == 0
    score = math.log(probability/(1-probability))
    def expected_loss(s):
        p = 1/(1+math.exp(-s))
        return p*q0+(1-p)*q1
    step = 1e-5
    choice_error = abs((expected_loss(score+step)-expected_loss(score-step))/(2*step)-exact_choice)
    assert choice_error < 2e-8
    for factual_memory in (0, 1):
        memory_linear = probability*(1-probability)*(-1/(1+math.exp(factual_memory)))
        assert memory_linear < 0
    # Linear terminal loss: check the state coefficient against finite differences.
    def expected_linear_loss(s):
        return -1/(1+math.exp(-s))
    linear_difference = (expected_linear_loss(score+step)-expected_linear_loss(score-step))/(2*step)
    assert abs(linear_difference-probability*(1-probability)*(-1)) < 2e-8
    return dict(status='passed', maximum_finite_difference_absolute_error=max(errors),
                private_write_choice_finite_difference_absolute_error=choice_error,
                scope='Static ideal-arithmetic equations; no native training, tensor gradients or benchmark')


if __name__ == '__main__':
    print(json.dumps(check()))
