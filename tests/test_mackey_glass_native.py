import sys
import types

import numpy as np
import torch

sys.path.insert(0, 'experiments'); sys.path.insert(0, 'experiments/vendor/neurobench_2_3_0')
import mackey_glass_native as M  # noqa: E402


def test_slices_match_official_neurobench_loader():
    sys.modules.setdefault('jitcdde', types.SimpleNamespace(y=lambda *a: .5, t=0., jitcdde_lyap=None))
    from neurobench.datasets import MackeyGlass
    for r in (0, 3, 17, 29):
        mg = MackeyGlass(str(M.DATA / 'mg_17.npy'), start_offset=float(torch.arange(0., 15., .5)[r] * 75), bin_window=1,
                         download=False)
        train, test = mg[mg.ind_train], mg[mg.ind_test]
        z = M.official_slice(17, r)
        assert np.array_equal(train[0][:, 0, 0].numpy(), z[:750]) and np.array_equal(train[1][:, 0].numpy(), z[1:751])
        assert np.array_equal(test[1][:, 0].numpy(), z[751:1501]) and len(test[1]) == 750


def test_smape_is_the_official_formula():
    p, y = np.array([1., 2., .5]), np.array([1.1, 1.9, .7])
    assert abs(M.smape(p, y) - 200 * np.mean(np.abs(p - y) / (np.abs(p) + np.abs(y)))) < 1e-12


def test_feedback_with_the_true_content_equals_teacher_forcing():
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.compiled_episodes import compiled_logits, layer_step
    from sleeping_machines.fast_native_core import fast_class
    torch.manual_seed(0)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=3, classes=1, payload=8, depth=2, heads=2, pool=2).double()
    z = np.sin(np.arange(40) / 3.)
    rows = [dict(events=[(float(i), M.taps_of(z, i, 3).astype(np.float64)) for i in range(s, s + 20)]) for s in (0, 7)]
    ref = compiled_logits(m, rows, 5, all_logits=True, step=layer_step)
    truth = torch.tensor(np.stack([[M.taps_of(z, i, 3) for i in range(s, s + 20)] for s in (0, 7)]), dtype=torch.float64)
    k = {'i': 4}

    def teacher(prev, logits):          # returns the row's own next content: identical inputs, identical outputs
        out = truth[:, k['i']]; k['i'] += 1
        return out
    got = compiled_logits(m, rows, 5, all_logits=True, step=layer_step, feedback=(4, teacher))
    assert torch.equal(ref, got)
    closed = compiled_logits(m, rows, 5, all_logits=True, step=layer_step,
                             feedback=(4, lambda prev, logits: torch.cat([logits[:, :1], prev[:, :-1]], -1)))
    assert torch.equal(closed[:, :4], ref[:, :4]) and not torch.equal(closed[:, 4:], ref[:, 4:])
