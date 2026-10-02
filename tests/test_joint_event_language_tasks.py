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


def test_native_joint_predict_is_causal_and_fast_path_matches_reference():
    import torch
    import joint_event_language_benchmark as B
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    row = J.episodes(4, 9)[0]
    torch.manual_seed(0)
    ref = AddressedEventHeads(sources=1, content_dim=J.WIDTH, classes=2, payload=4, depth=2, pool=2, heads=2).double()
    fast = fast_class(AddressedEventHeads)(sources=1, content_dim=J.WIDTH, classes=2, payload=4, depth=2, pool=2, heads=2).double()
    fast.load_state_dict(ref.state_dict())
    outs = []
    for m in (ref, fast):
        m.train(); m.zero_grad()
        with torch.random.fork_rng():
            torch.manual_seed(5); z, y, _ = B.predict(m, row)
        torch.nn.functional.cross_entropy(z, y).backward(); outs.append(z.detach())
        assert m._fast_layers is None if m is fast else True
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-11)
    for a, b in zip(ref.parameters(), fast.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-9, atol=1e-11)
    # the query output cannot depend on the label: flipping the target leaves inputs unchanged
    flipped = row[:-1] + [J.Event(0, row[-1].time, row[-1].mark, 1 - row[-1].target)]
    ref.eval()
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(5); z1, _, _ = B.predict(ref, row)
        torch.manual_seed(5); z2, _, _ = B.predict(ref, flipped)
    assert torch.equal(z1, z2)


def test_history_knob_keeps_default_data_and_lengthens_history():
    import hashlib, json
    default = J.episodes(64, 1301)
    assert J.data_hash(default) == J.data_hash(J.episodes(64, 1301, (2, 6)))
    long = J.episodes(64, 1301, (60, 70))
    assert np.mean([len(r) for r in long]) > np.mean([len(r) for r in default]) + 50
    for row in long:  # the rule and balance are unchanged by history length
        named = next(i for i, w in enumerate(J.WORDS) if f' {w} ' in f' {_text(row)} ')
        last = max(e.time for e in row if e.mark[J.MARK0 + named] == 1.)
        assert int(row[-1].time - last < J.DELTA) == row[-1].target
    assert sum(r[-1].target for r in long) == 32
