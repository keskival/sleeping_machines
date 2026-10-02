import sys

import numpy as np
import torch

sys.path.insert(0, 'experiments')
import long_range_core_benchmark as L  # noqa: E402
from sleeping_machines.count_carrying_language import eval_stream_counts  # noqa: E402


def test_lag_targets_copy_the_symbol_distance_back_from_the_cue():
    toks, mask = L.make_stream('lag', 2000, 12, 0)
    for p in np.flatnonzero(mask):
        assert toks[p - 1] == L.LAG_CUE and toks[p] == toks[p - 1 - 12]


def test_induction_targets_follow_the_most_recent_query_occurrence():
    toks, mask = L.make_stream('induction', 3000, 64, 1)
    for p in np.flatnonzero(mask):
        q = toks[p - 1]; assert toks[p - 2] == L.INDUCTION_CUE
        window = toks[max(0, p - 2 - 64):p - 2]
        last = max(i for i in range(len(window) - 1) if window[i] == q and window[i + 1] < L.FILLER)
        assert toks[p] == window[last + 1]


def test_count_tables_are_near_chance_on_targets():
    for task, d in (('lag', 48), ('induction', 128)):
        fit, _ = L.make_stream(task, 8192, d, 2)
        dev, mask = L.make_stream(task, 4096, d, 3)
        c = eval_stream_counts(fit, dev, 4)  # prequential order-1..4 counts, causal
        bits = []
        for p in np.flatnonzero(mask[1:]):
            best = np.inf
            for k in range(4):  # best single order, add-one smoothing: an optimistic count predictor
                v = c[k, p]; best = min(best, -np.log2((v[dev[p + 1]] + 1) / (v.sum() + 27)))
            bits.append(best)
        assert np.mean(bits) > 4.3, (task, np.mean(bits))


def test_native_core_runs_on_the_stream_and_scores_targets():
    toks, mask = L.make_stream('lag', 200, 4, 4)
    torch.manual_seed(0)
    m = L.NativeStreamLanguageModel(payload=4, depth=2, pool=2, heads=2)
    s = L.score(m, torch.tensor(toks), mask, 16)
    assert s['targets'] == int(mask[1:].sum()) and np.isfinite(s['target_bpc'])


def test_kv_model_runs_on_the_stream():
    toks, mask = L.make_stream('induction', 200, 32, 5)
    torch.manual_seed(0)
    m = L.ParallelHeadRaceLanguageModel(4, 2, 2, matching=2, recent=2, heads=2)
    s = L.score(m, torch.tensor(toks), mask, 16)
    assert np.isfinite(s['target_bpc'])


def test_tapped_model_runs_on_the_stream():
    from sleeping_machines.dilated_delay_taps import TappedNativeStreamLanguageModel
    toks, mask = L.make_stream('lag', 200, 8, 6)
    torch.manual_seed(0)
    s = L.score(TappedNativeStreamLanguageModel(4, 3, 2, 2), torch.tensor(toks), mask, 16)
    assert np.isfinite(s['target_bpc'])


def test_addressed_model_runs_on_the_stream():
    from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel
    toks, mask = L.make_stream('lag', 200, 8, 7)
    torch.manual_seed(0)
    s = L.score(ContextAddressedNativeModel(4, 2, 2, 2, order=2, buckets=64), torch.tensor(toks), mask, 16)
    assert np.isfinite(s['target_bpc'])
