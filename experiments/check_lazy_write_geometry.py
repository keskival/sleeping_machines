"""Static lazy-memory credit witnesses; standard library, no model execution.

These constructed conditional races keep messages/first times fixed and all
slots already seen. They isolate stored memory, timestamps and future reads;
they do not attribute a completed native fit's instability.
"""
import cmath
from dataclasses import dataclass, replace
import json
import math


@dataclass(frozen=True)
class Slot:
    memory: complex
    stamp: float
    rate: float
    frequency: float = 0.


def transport(slot, age, forget):
    return cmath.exp(complex(-slot.rate*forget*age, slot.frequency*age))


def alternatives(slots, arrival, forget, written):
    branches = []
    for i, slot in enumerate(slots):
        branch = list(slots)
        branch[i] = replace(slot, memory=slot.memory*transport(slot, arrival-slot.stamp, forget[i])+written[i],
                            stamp=arrival)
        branches.append(branch)
    return branches


def aged_read(slots, time, forget):
    return math.fsum((slot.memory*transport(slot, time-slot.stamp, forget[i])).real
                     for i, slot in enumerate(slots))


def pi(scores):
    rates = [math.exp(x-max(scores)) for x in scores]
    return [x/math.fsum(rates) for x in rates]


def categorical(scores, utilities):
    probabilities = pi(scores)
    mean = math.fsum(p*q for p, q in zip(probabilities, utilities))
    return [p*(q-mean) for p, q in zip(probabilities, utilities)]


def memory_surrogate(scores, slots, branches, adjoints, written=None):
    # The common unwritten background cancels when categorical credit centers.
    coefficients = [(g.conjugate()*(written[i] if written is not None else branches[i][i].memory-slot.memory)).real
                    for i, (slot, g) in enumerate(zip(slots, adjoints))]
    return categorical(scores, coefficients)


def check():
    checks, finite_errors = [], []
    scores = [0., 0.]
    slots = [Slot(1.+0j, 0., .5), Slot(.5+0j, 0., .25)]
    arrival, query_time = 2., 5.
    zero, constant = [0j, 0j], [1., 1.]

    def exact(utilities):
        result = categorical(scores, utilities)
        step = 1e-5
        for i in range(len(scores)):
            plus, minus = list(scores), list(scores)
            plus[i] += step
            minus[i] -= step
            risk = lambda s: math.fsum(p*q for p, q in zip(pi(s), utilities))
            error = abs((risk(plus)-risk(minus))/(2*step)-result[i])
            assert error < 5e-10, error
            finite_errors.append(error)
        assert abs(math.fsum(result)) < 1e-14
        return result

    def expected_stored(branches):
        teachers = []
        for factual in branches:
            cotangents = [transport(s, query_time-s.stamp, constant[i]).conjugate()
                          for i, s in enumerate(factual)]
            teachers.append(memory_surrogate(scores, slots, branches, cotangents))
        return [math.fsum(p*g[i] for p, g in zip(pi(scores), teachers)) for i in range(len(scores))]

    # A homogeneous constant-generator rebase changes coordinates, not the
    # common-time physical state, including a nonzero rotation frequency.
    rotating = Slot(1.+.7j, 0., .5, .37)
    assert abs(transport(rotating, 2., 1.)*transport(rotating, 3., 1.)-
               transport(rotating, 5., 1.)) < 1e-14
    checks.append('Constant-generator decay/rotation semigroup')
    branches = alternatives(slots, arrival, constant, zero)
    utilities = [aged_read(branch, query_time, constant) for branch in branches]
    assert max(utilities)-min(utilities) < 1e-14
    no_op_exact = exact(utilities)
    factual = branches[0]
    adjoints = [transport(s, query_time-s.stamp, constant[i]).conjugate()
                for i, s in enumerate(factual)]
    no_op_stored = memory_surrogate(scores, slots, branches, adjoints)
    no_op_written = memory_surrogate(scores, slots, branches, adjoints, zero)
    no_op_expected_stored = expected_stored(branches)
    assert abs(no_op_exact[0]) < 1e-14 and abs(no_op_stored[0]) > .01
    assert no_op_written == [0., 0.]
    assert abs(no_op_expected_stored[0]) > .001
    checks.append('Pure rebase: full utility zero, stored-only surrogate spurious, written-only zero')

    # Along the exact continuous coordinate change, stamp and memory tangents
    # cancel. A finite endpoint Taylor approximation is not an exact rebase.
    slot = slots[0]
    for alpha in [0., .2, .7, 1.]:
        memory = slot.memory.real*math.exp(-slot.rate*alpha*arrival)
        stamp = alpha*arrival
        g_m = math.exp(-slot.rate*(query_time-stamp))
        g_t = slot.rate*memory*g_m
        assert abs(g_m*(-slot.rate*arrival*memory)+g_t*arrival) < 1e-14
    finite_joint = math.exp(-slot.rate*query_time)*(branches[0][0].memory.real-slot.memory.real)+\
                   slot.rate*slot.memory.real*math.exp(-slot.rate*query_time)*arrival
    assert abs(finite_joint) > .01
    checks.append('Joint infinitesimal coordinate cancellation; finite endpoint linearization still biased')

    # Native keys read stored m before candidate decay. With an affine
    # conditional loss on that key feature, the stored delta is exact utility.
    raw_utilities = [math.fsum(s.memory.real for s in branch) for branch in branches]
    raw_exact = exact(raw_utilities)
    raw_stored = memory_surrogate(scores, slots, branches, [1.+0j, 1.+0j])
    raw_written = memory_surrogate(scores, slots, branches, [1.+0j, 1.+0j], zero)
    assert abs(raw_exact[0]) > .1
    assert max(abs(x-y) for x, y in zip(raw_exact, raw_stored)) < 1e-14
    assert raw_written == [0., 0.]
    checks.append('Raw stored-key read: stored delta exact, written-only omits real route utility')

    # Future input-dependent forget differs from the current write's gate.
    # Even a zero-input refresh then changes the later age-aware read.
    dynamic = alternatives(slots, arrival, [.2, .2], zero)
    dynamic_utilities = [aged_read(branch, query_time, constant) for branch in dynamic]
    dynamic_exact = exact(dynamic_utilities)
    dynamic_adjoints = [transport(s, query_time-s.stamp, constant[i]).conjugate()
                        for i, s in enumerate(dynamic[0])]
    dynamic_stored = memory_surrogate(scores, slots, dynamic, dynamic_adjoints)
    dynamic_written = memory_surrogate(scores, slots, dynamic, dynamic_adjoints, zero)
    dynamic_expected_stored = expected_stored(dynamic)
    assert dynamic_exact[0] > 0 and dynamic_stored[0] < 0
    assert dynamic_written == [0., 0.]
    assert dynamic_expected_stored[0] < 0
    risk_before = math.fsum(p*q for p, q in zip(pi(scores), dynamic_utilities))
    scaled_risk_increases = {}
    for scale in [.01, .1, 1.]:
        next_scores = [s-.001*scale*g for s, g in zip(scores, dynamic_expected_stored)]
        difference = math.fsum(p*q for p, q in zip(pi(next_scores), dynamic_utilities))-risk_before
        assert difference > 0
        scaled_risk_increases[str(scale)] = difference
    coefficients = [(g.conjugate()*(dynamic[i][i].memory-slot.memory)).real
                    for i, (slot, g) in enumerate(zip(slots, dynamic_adjoints))]
    residuals = [q-a for q, a in zip(dynamic_utilities, coefficients)]
    probabilities = pi(scores)
    residual_mean = math.fsum(p*e for p, e in zip(probabilities, residuals))
    residual_variance = math.fsum(p*(e-residual_mean)**2 for p, e in zip(probabilities, residuals))
    whitened_error = math.fsum((g-h)**2/p for g, h, p in zip(dynamic_exact, dynamic_stored, probabilities))
    assert math.isclose(whitened_error, residual_variance, rel_tol=1e-12, abs_tol=1e-15)
    checks.append('Changing forget: actual nonzero utility, stored-only opposite sign, written-only missing')

    for gap in [.001, 1., 10., 100.]:
        factor = transport(rotating, gap, 1.)
        assert abs(factor) <= 1+1e-14 and abs(factor-1) <= 2+1e-14
    # A bounded scalar write gate does not bound the unnormalized incoming
    # vector or trainable input map. Set control weights/bias to0: write=1.
    norms = [math.hypot(scale, -scale) for scale in [1., 1000.]]
    assert math.isclose(norms[1]/norms[0], 1000.)
    checks.append('Transport-delta bound2|m|; bounded gate does not bound written content')
    return dict(status='passed', contracts=checks,
                maximum_finite_difference_absolute_error=max(finite_errors),
                pure_rebase=dict(exact=no_op_exact, factual_winner=0, stored_only=no_op_stored,
                                 expected_stored_only=no_op_expected_stored, written_only=no_op_written),
                stored_key_read=dict(exact=raw_exact, stored_only=raw_stored, written_only=raw_written),
                changed_forget=dict(exact=dynamic_exact, factual_winner=0, stored_only=dynamic_stored,
                                   expected_stored_only=dynamic_expected_stored, written_only=dynamic_written),
                categorical_teacher_error=dict(pi_whitened_squared_error=whitened_error,
                                               conditional_residual_variance=residual_variance),
                scaled_biased_teacher_risk_increases=scaled_risk_increases,
                scope='Constructed fixed-message/fixed-first-time affine conditional losses; no native gradients, fitting, divergence attribution or benchmark')


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, allow_nan=False))
