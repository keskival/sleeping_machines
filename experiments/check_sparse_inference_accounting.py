"""Read-only accounting/certificate fixtures; no tensor runtime or benchmark."""
import math
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    namespace = runpy.run_path(str(ROOT/'experiments/sparse_inference_accounting.py'))
    accounting, certificate = namespace['cache_accounting'], namespace['race_certificate']
    one = accounting(32, 4, 2, 1, 8)
    four = accounting(32, 4, 2, 4, 8)
    large = accounting(32, 4, 2, 4096, 8)
    assert four['available_slots'] == four['scored_keys_per_position'] == 32
    assert four['selected_writes_per_position'] == four['sparse_proposals_per_position'] == 8
    assert four['final_stacked_tensor_bytes_per_call'] == 612736
    assert four['parameter_payload_bytes_read_for_stacking'] == 610688
    assert four['allocated_unit_state_and_cache_bytes'] == 67840
    # Unit cost grows with U; query stacking has a fixed intercept. Lane
    # changes multiply state/cache, while parameter stacking is unchanged.
    two = accounting(32, 4, 2, 2, 8)
    assert four['final_stacked_tensor_bytes_per_call']-two['final_stacked_tensor_bytes_per_call'] == \
           2*(two['final_stacked_tensor_bytes_per_call']-one['final_stacked_tensor_bytes_per_call'])
    half_lanes = accounting(32, 4, 2, 4, 4)
    assert half_lanes['final_stacked_tensor_bytes_per_call'] == four['final_stacked_tensor_bytes_per_call']
    assert 2*half_lanes['allocated_unit_state_and_cache_bytes'] == four['allocated_unit_state_and_cache_bytes']
    assert large['final_stacked_tensor_bytes_per_call'] > 500*1024**2
    stable = certificate([1., math.exp(.1)], [math.exp(1e-7), math.exp(.1-1e-7)], 0, 0)
    assert stable['certified'] and stable['winner_agrees']
    flipped = certificate([1., math.exp(1e-8)], [math.exp(6e-9), math.exp(4e-9)], 0, 1)
    assert not flipped['certified'] and not flipped['winner_agrees']
    # Identical proposals hide a changed route from every output-only test.
    proposals = [4., 4.]
    assert proposals[0] == proposals[1] and not flipped['winner_agrees']
    tied = certificate([1., 1.], [1., 1.], 0, 0)
    assert tied['winner_agrees'] and not tied['certified']
    singleton = certificate([2.], [3.], 0, 0)
    assert singleton['certified'] and singleton['reference_log_margin'] is None
    for times, other, first, second in [([0., 1.], [1., 2.], 0, 0), ([1., 2.], [1.], 0, 0),
                                        ([1., 2.], [1., 2.], 1, 0)]:
        try:
            certificate(times, other, first, second)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid race input accepted')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(static_contracts='passed', pool4_stack_bytes=four['final_stacked_tensor_bytes_per_call'],
               pool4096_stack_bytes=large['final_stacked_tensor_bytes_per_call'],
               pool4_unit_cache_bytes=four['allocated_unit_state_and_cache_bytes'],
               near_tie_route_change_detected=True, native_contracts='pending'))


if __name__ == '__main__':
    main()
