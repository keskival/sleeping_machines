"""Exact expected target coverage for the source-frozen random-segment protocol.

Pure combinatorics: no model, training, dataset or accelerator execution.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def coverage(length, segment, windows, lanes):
    # np.integers(0, length - segment - 1): upper bound excluded.
    starts = length - segment - 1
    if starts < segment or windows < 0 or lanes < 1:
        raise ValueError('Require at least one full interior and nonnegative draws')
    draws = windows * lanes
    def touched(count):
        return -math.expm1(draws * math.log1p(-count / starts)) if count < starts else float(draws > 0)
    interior = starts - segment + 1
    expected = interior * touched(segment) + 2 * sum(touched(c) for c in range(1, segment))
    presentations = draws * segment
    return dict(length=length, segment=segment, windows=windows, lanes=lanes,
                independent_segment_draws=draws, target_presentations=presentations,
                possible_next_character_targets=length - 1,
                reachable_targets=length - 2,
                expected_unique_target_positions=expected,
                expected_unique_fraction=expected / (length - 1),
                expected_repeat_presentations=presentations - expected,
                interior_expected_exposure=draws * segment / starts,
                interior_probability_unseen=1 - touched(segment))


def fixtures():
    # Exact enumeration of all two-draw outcomes; no independence assumption
    # between neighboring target positions is needed for expected coverage.
    length, segment, draws = 9, 3, 2
    starts = length - segment - 1
    total = 0
    for a in range(starts):
        for b in range(starts):
            total += len(set(range(a + 1, a + segment + 1)) | set(range(b + 1, b + segment + 1)))
    actual = coverage(length, segment, draws, 1)['expected_unique_target_positions']
    assert abs(actual - total / starts ** draws) < 1e-12
    assert coverage(length, segment, 0, 1)['expected_unique_target_positions'] == 0
    assert abs(coverage(length, segment, 1, 1)['expected_unique_target_positions'] - segment) < 1e-12
    return dict(enumerated_expected_unique=total / starts ** draws, formula_expected_unique=actual, status='passed')


if __name__ == '__main__':
    driver = ROOT / 'experiments/language_batched_benchmark.py'
    text = driver.read_text()
    assert 'rng.integers(0, len(fit) - S - 1, B)' in text
    rows = []
    for length in (10_000_000, 90_000_000):
        for passes in (1, 2, 4, 6):
            rows.append(dict(nominal_passes=passes, **coverage(length, 128, int(passes * (length - 1) // (128 * 64)), 64)))
    result = dict(scope='Exact expectation over independent uniform segment draws; not measured realized coverage, generalization or quality. Neighboring positions are dependent.',
                  source_sha256={str(driver.relative_to(ROOT)): hashlib.sha256(driver.read_bytes()).hexdigest(),
                                 str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  fixtures=fixtures(), rows=rows)
    output = ROOT / 'experiments/results/diagnostics/aws_segment_exposure_20261003.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
