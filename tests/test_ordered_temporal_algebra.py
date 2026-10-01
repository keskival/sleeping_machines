"""Read-only numerical check of theory §336, not an optimizer or model fit."""
import numpy as np


def ordered_recurrence(times, a, b, gamma1, gamma2, query):
    r = s = e = f = 0.
    previous = times[0]
    for time, first, second in zip(times, a, b):
        gap = time-previous
        decay1, decay2 = np.exp(-gamma1*gap), np.exp(-gamma2*gap)
        e = decay1*(e-gap*r)
        r *= decay1
        s = decay2*s+second*r
        f = decay2*f+second*e
        r += first
        previous = time
    final = np.exp(-gamma2*(query-previous))
    return final*s, final*f


def explicit_pairs(times, a, b, gamma1, gamma2, query):
    return sum(np.exp(-gamma2*(query-times[j]))*b[j]
               *np.exp(-gamma1*(times[j]-times[i]))*a[i]
               for j in range(len(times)) for i in range(j))


def test_ordered_state_and_exact_temporal_eligibility_against_pair_expansion():
    rng = np.random.default_rng(883)
    for length in (2, 7, 33):
        times = np.cumsum(rng.uniform(.001, 2., length))
        a, b = rng.normal(size=(2, length))
        query = times[-1]+.73
        gamma1, gamma2 = .17, .29
        value, eligibility = ordered_recurrence(times,a,b,gamma1,gamma2,query)
        expected = explicit_pairs(times,a,b,gamma1,gamma2,query)
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
        epsilon = 1e-5
        finite = (explicit_pairs(times,a,b,gamma1+epsilon,gamma2,query)
                  -explicit_pairs(times,a,b,gamma1-epsilon,gamma2,query))/(2*epsilon)
        np.testing.assert_allclose(eligibility, finite, rtol=2e-8, atol=1e-8)
    times = np.array([.1, .7])
    ab, _ = ordered_recurrence(times,[1.,0.],[0.,1.],.17,.29,1.)
    ba, _ = ordered_recurrence(times,[0.,1.],[1.,0.],.17,.29,1.)
    assert ab > 0 and ba == 0
