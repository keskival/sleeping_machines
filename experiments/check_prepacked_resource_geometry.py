"""Preparation break-even witnesses, including ownership/setup overhead."""
from pathlib import Path
import runpy
import sys


def main():
    f = runpy.run_path(str(Path(__file__).with_name('prepacked_resource_geometry.py')))['preparation_payload']
    # Equality is not a positive gain; one new version pays setup again.
    assert f(100, 100, 100, 2)['payload_rw_difference_bytes'] == 0
    assert f(100, 100, 100, 2)['first_request_count_with_positive_payload_difference'] == 3
    assert f(100, 100, 101, 2)['payload_rw_difference_bytes'] == -2
    assert f(100, 100, 101, 3)['payload_rw_difference_bytes'] == 198
    assert f(100, 100, 0, 1)['payload_rw_difference_bytes'] == 0
    # Known p32/D4/U4 final stack shapes, native all-float32 parameter-count premise.
    r = f(610688, 612736, 177019*4, 10)
    assert r['ordinary_repeated_assembly_payload_rw_bytes'] == 12234240
    assert r['prepared_snapshot_and_once_assembly_payload_rw_bytes'] == 2639576
    assert r['payload_rw_difference_bytes'] == 9594664
    assert r['first_request_count_with_positive_payload_difference'] == 3
    for values in ((0, 0, 1, 1), (1, 1, 1, 0), (-1, 1, 1, 1), (1, 1, 1, True)):
        try:
            f(*values)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid ledger inputs admitted')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(stdlib_payload_contracts='passed', ownership_copy_charged=True,
               p32_pool4_ten_call_scenario=r, native_traffic='unmeasured'))


if __name__ == '__main__':
    main()
