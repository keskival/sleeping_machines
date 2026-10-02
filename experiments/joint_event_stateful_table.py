"""Strong timestamp-aware calibration for the proposed joint event/text task.

One observed last timestamp per mark and full observed question text. Learn a
per-question one-dimensional split over any mark age, including its direction;
no word/mark mapping or generator threshold is supplied to the learner.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import joint_event_language_tasks as J


def features(row, work):
    text = []; last = [None] * 4
    for event in row:
        work['events'] += 1
        index = None
        for i, value in enumerate(event.mark):
            work['content_bit_inspections'] += 1
            if value:
                index = i; break
        if index is None: raise ValueError('Observed one-hot content required')
        if index < 27:
            text.append(' ' if index == 0 else chr(96 + index)); work['character_appends'] += 1
        elif index < 31:
            last[index - 27] = event.time; work['timestamp_writes'] += 1
        else:
            work['age_subtractions'] += sum(t is not None for t in last)
            return ''.join(text), tuple(0. if t is None else event.time - t for t in last)
    raise ValueError('Observed query flag required')


def ledger():
    return dict(events=0, content_bit_inspections=0, character_appends=0, timestamp_writes=0,
                age_subtractions=0, sorting_comparisons=0, threshold_candidates=0, probability_evaluations=0,
                logarithms=0, label_count_updates=0, prediction_comparisons=0)


def probability(counts, work):
    work['probability_evaluations'] += 1
    return (counts[1] + .5) / (sum(counts) + 1.)


def loss(counts, work):
    p = probability(counts, work); work['logarithms'] += 2
    return -counts[1] * math.log(p) - counts[0] * math.log1p(-p)


def fit(rows):
    work = ledger(); groups = defaultdict(list); total = [0, 0]
    for row in rows:
        text, ages = features(row, work); y = row[-1].target
        groups[text].append((ages, y)); total[y] += 1; work['label_count_updates'] += 1
    model = dict(fallback=probability(total, work), questions={})
    class CountedAge(float):
        def __lt__(self, other):
            work['sorting_comparisons'] += 1
            return float.__lt__(self, other)
    for text, group in sorted(groups.items()):
        counts = [sum(y == label for _, y in group) for label in (0, 1)]
        work['label_count_updates'] += len(group)
        best = dict(mark=None, threshold=None, below=probability(counts, work), above=None)
        best_loss = loss(counts, work)
        for mark in range(4):
            ordered = sorted(group, key=lambda xy: CountedAge(xy[0][mark])); left = [0, 0]
            for k, (ages, y) in enumerate(ordered[:-1]):
                left[y] += 1; work['label_count_updates'] += 1
                if ages[mark] == ordered[k + 1][0][mark]: continue
                work['threshold_candidates'] += 1
                right = [counts[i] - left[i] for i in (0, 1)]
                score = loss(left, work) + loss(right, work)
                if score < best_loss:
                    best_loss = score
                    best = dict(mark=mark, threshold=(ages[mark] + ordered[k + 1][0][mark]) / 2,
                                below=probability(left, work), above=probability(right, work))
        model['questions'][text] = best
    return model, work


def predict(model, feature, work):
    text, ages = feature; node = model['questions'].get(text)
    if node is None: return model['fallback']
    if node['mark'] is None: return node['below']
    work['prediction_comparisons'] += 1
    return node['below'] if ages[node['mark']] < node['threshold'] else node['above']


def score(model, rows):
    started = time.perf_counter(); work = ledger(); correct = 0; nll = 0.; values = []
    for row in rows:
        p = predict(model, features(row, work), work)
        y = row[-1].target; correct += int((p >= .5) == y)
        nll -= math.log(p if y else 1 - p); values.append(p)
    return dict(targets=len(rows), accuracy=correct / len(rows), nll=nll / len(rows),
                probabilities=values, logical_work=work, wall_s=time.perf_counter() - started)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    a = p.parse_args(); out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    started = time.perf_counter(); fitting = J.episodes(512, 1301)
    # Label mutation, future observations and a common clock shift cannot
    # enter the causal feature extractor or change the prediction inputs.
    row = fitting[0]; original = features(row, ledger())
    mutated = [J.Event(e.source, e.time, e.mark, None if e.target is None else 1 - e.target) for e in row]
    shifted = [J.Event(e.source, e.time + 128, e.mark, e.target) for e in row]
    assert features(mutated, ledger()) == original
    assert features(row + [J.Event(0, row[-1].time + 10, J.one_hot(27))], ledger()) == original
    new = features(shifted, ledger())
    assert new[0] == original[0]
    # Missing marks have an explicit zero sentinel; actual nonnegative ages
    # and this sentinel remain invariant under a common clock shift.
    for i in range(4):
        assert abs(new[1][i] - original[1][i]) < 1e-12
    fitting_started = time.perf_counter(); model, fitting_work = fit(fitting)
    fit_wall = time.perf_counter() - fitting_started
    final = {name: score(model, J.episodes(n, seed)) for name, n, seed in
             [('development', 256, 2301), ('confirmation', 1024, 3301)]}
    names = ['experiments/joint_event_stateful_table.py', 'experiments/joint_event_language_tasks.py',
             'experiments/native_event_tasks.py']
    result = dict(status='completed', args=vars(a), fitting_targets=512, fitting_passes=1,
        fitting_seed=1301, data_sha256=dict(fit=J.data_hash(fitting), development=J.data_hash(J.episodes(256, 2301)),
                                          confirmation=J.data_hash(J.episodes(1024, 3301))),
        fitting_wall_s=fit_wall, fitting_logical_work=fitting_work, model=model, final=final,
        causal_feature_contracts_passed=True, wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        scope='Timestamp-aware causal stateful table, full question lookup and one learned age split per question. '
              'No generator word/mark mapping or threshold supplied. Same distinct data, one fitting pass; '
              'logical counts/wall/state separate from neural FLOPs. Calibration control, not an integrated architecture '
              'or a general natural-language model. Confirmation does not tune the fixed learner.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
