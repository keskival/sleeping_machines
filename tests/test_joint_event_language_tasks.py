import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, 'experiments')
import joint_event_language_tasks as J  # noqa: E402


def _text(row):
    return ''.join(' ' if e.mark[0] else chr(96 + e.mark.index(1.)) for e in row if 1. in e.mark[:27])


def test_labels_follow_the_elapsed_time_rule_and_targets_never_enter_inputs():
    for row in J.episodes(400, 1):
        q = row[-1]; assert q.target is not None and q.mark[J.QUERY] == 1. and all(e.target is None for e in row[:-1])
        named = next(i for i, w in enumerate(J.WORDS) if f' {w} ' in f' {_text(row)} ')
        last = max(e.time for e in row if e.mark[J.MARK0 + named] == 1.)
        assert int(q.time - last < J.DELTA) == q.target
        assert all(row[i].time <= row[i + 1].time for i in range(len(row) - 1))
        assert all(len(e.mark) == J.WIDTH and sum(e.mark) == 1. for e in row)


def test_labels_are_balanced_and_text_precedes_the_decisive_event():
    rows = J.episodes(400, 2)
    assert sum(r[-1].target for r in rows) == 200
    for row in rows:
        named = next(i for i, w in enumerate(J.WORDS) if f' {w} ' in f' {_text(row)} ')
        last_char = max(e.time for e in row if 1. in e.mark[:27])
        last_named = max(e.time for e in row if e.mark[J.MARK0 + named] == 1.)
        assert last_char < last_named


def test_time_blind_tables_are_near_chance():
    fit, dev = J.episodes(2000, 3), J.episodes(2000, 4)
    table = defaultdict(Counter)
    for row in fit:
        table[J.blind_features(row)[:3]][row[-1].target] += 1   # word, last mark, events after named mark
    hits = []
    for row in dev:
        c = table.get(J.blind_features(row)[:3], Counter())
        hits.append((c[1] > c[0]) == bool(row[-1].target) if c else row[-1].target == 1)
    assert abs(np.mean(hits) - .5) < .04, np.mean(hits)
    # rank structure alone (number of events after the named mark) is independent of the label
    by_after = defaultdict(list)
    for row in fit:
        by_after[J.blind_features(row)[2]].append(row[-1].target)
    assert all(abs(np.mean(v) - .5) < .07 for v in by_after.values() if len(v) > 100)
