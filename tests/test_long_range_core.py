import sys
import json

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


def test_local_count_controls_score_each_order_without_target_dependent_selection():
    for task, d in (('lag', 48), ('induction', 128)):
        fit, _ = L.make_stream(task, 8192, d, 2)
        dev, mask = L.make_stream(task, 4096, d, 3)
        c = eval_stream_counts(fit, dev, 4)  # prequential order-1..4 counts, causal
        positions = np.flatnonzero(mask[1:]); labels = dev[positions+1]
        for k in range(4):
            v = c[k, positions]; predictive = (v+1)/(v.sum(-1, keepdims=True)+27)
            np.testing.assert_allclose(predictive.sum(-1), 1., atol=2e-7)
            bits = -np.log2(predictive[np.arange(len(labels)), labels])
            assert np.isfinite(bits).all() and bits.mean() > 4.3, (task, k, bits.mean())


def test_lag_can_copy_cues_and_has_no_uniform_24_target_guarantee():
    tokens, mask = L.make_stream('lag', 8192, 48, 2)
    assert (tokens[mask] == L.LAG_CUE).any()


def test_missing_score_group_is_null_instead_of_nan():
    tokens=torch.tensor([1,2,3,4,5])
    model=L.NativeStreamLanguageModel(payload=2,depth=1,pool=2,heads=2)
    all_target=L.score(model,tokens,np.ones(5,bool),16)
    all_filler=L.score(model,tokens,np.zeros(5,bool),16)
    assert all_target['filler_bpc'] is None and all_target['targets']==4
    assert all_filler['target_bpc'] is None and all_filler['target_accuracy'] is None
    json.dumps([all_target,all_filler],allow_nan=False)


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


def test_driver_credit_comparison_preserves_optimizer_and_target_budgets(tmp_path, monkeypatch):
    # Execute the real CLI loop, including a partial final window; chunk lengths
    # can change gradients but must not change Adam's target budget or frequency.
    for name in ('sleeping_machines', 'experiments'):
        (tmp_path/name).symlink_to(L.ROOT/name, target_is_directory=True)
    # Results must be isolated while provenance reads actual frozen source.
    original_root = L.ROOT
    (tmp_path/'experiments').unlink()
    (tmp_path/'experiments').mkdir()
    for source in (original_root/'experiments').glob('*.py'):
        (tmp_path/'experiments'/source.name).symlink_to(source)
    monkeypatch.setattr(L, 'ROOT', tmp_path)
    rows = []
    for chunk in (16,64):
        tag = f'credit{chunk}'
        monkeypatch.setattr(sys, 'argv', ['long-range', '--tag', tag, '--task', 'lag', '--distance', '4',
            '--fit', '74', '--dev', '65', '--epochs', '2', '--payload', '2', '--depth', '1',
            '--chunk', str(chunk), '--update-targets', '64'])
        L.main()
        rows.append(json.loads((tmp_path/'experiments/results/long_range_core'/f'{tag}.json').read_text()))
    assert rows[0]['data_sha256'] == rows[1]['data_sha256']
    for row in rows:
        assert row['final']['optimizer_updates'] == 4
        assert row['final']['fitted_targets'] == 146
