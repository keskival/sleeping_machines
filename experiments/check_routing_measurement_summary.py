"""Reproducible measurement-unit contracts; standard library, no model runtime."""
import math
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    helper = runpy.run_path(str(ROOT/'experiments/routing_measurement_summary.py'))
    Summary, mixture = helper['RoutingSummary'], helper['mixture_summary']
    summary = Summary(2, 2, 2)
    summary.add(0, 1, [2, 1], [1.5, 1.5], 2.4, 1.8, 1, 3,
                boundary_keys=2, clock_sensitivity_sum=.005, weak_clock_races=1)
    summary.add(0, 1, [0, 2], [1., 1.], 1.4, 1.6, 0, 2)
    summary.add(1, 0, [1, 0], [.1, .9], .9,
                -.1*math.log2(.1)-.9*math.log2(.9), 0, 1)
    first, second = summary.report()
    assert first['winner_counts'] == [2, 3] and first['races'] == 5
    assert first['observed_winner_fraction'] == [.4, .6]
    assert first['mean_pi_per_unit'] == [.5, .5]
    # Missing clock/key measurements do not dilute the measured denominator.
    assert first['fraction_keys_at_clamp_boundary'] == 1/3
    assert first['key_races_measured'] == first['clock_races_measured'] == 3
    assert first['mean_postclamp_common_clock_sensitivity'] == .005/3
    assert first['fraction_clock_sensitivity_below_one_percent_max'] == 1/3
    assert second['boundary_key_count'] is None
    assert second['clock_races_measured'] == 0
    assert second['mean_postclamp_common_clock_sensitivity'] is None
    # A mixture may lose to its best component while satisfying Jensen.
    result = mixture([-math.log2(.9), -math.log2(.1)], 1., 1)
    assert result['mixture_minus_first_seed_bpc'] > 0
    assert abs(result['jensen_gap_bpc']-.7369655941662061) < 1e-14
    assert mixture([2.], 2., 1)['jensen_gap_bpc'] == 0
    invalid = [([2, 0], [1., 1.], 1.5, 1., 0, 1),
               ([1, 0], [.5, .5], .5, 1., 0, 1, 3),
               ([1, 0], [.5, .5], .5, 1., 0, 1, 0, .0026, 0)]
    for args in invalid:
        try:
            Summary(1, 1, 2).add(0, 0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid measurement denominator/bound accepted')
    try:
        mixture([1., 2.], 2., 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid probability-mixture bound accepted')
    for clock in [.01, .1, 1., 10., 100.]:
        sensitivity = .010*clock/(1+clock)**2
        assert 0 < sensitivity <= .0025
        assert math.isclose(sensitivity, .010/clock/(1+1/clock)**2, rel_tol=1e-14)
    # Base-rate half-life is not an upper bound with input-controlled forget.
    rate, forget = math.log(2)/6., .01
    effective_half_life = math.log(2)/(rate*forget)
    assert math.isclose(effective_half_life, 600.)
    assert math.isclose(math.exp(-rate*forget*effective_half_life), .5)
    retention = math.exp(-rate*.1*12)*math.exp(-rate*.01*48)
    assert math.isclose(retention, math.exp(-rate*(.1*12+.01*48)))
    # Import/help inspection must not start NumPy or the tensor runtime.
    runpy.run_path(str(ROOT/'experiments/language_routing_measurements.py'))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(summary_contracts='pass', clock_identity='pass', retention_identity='pass', lazy_imports='pass',
               native_tensor_contracts='pending', forward_diagnostic='not run'))


if __name__ == '__main__':
    main()
