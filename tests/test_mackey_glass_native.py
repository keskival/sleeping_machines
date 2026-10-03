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
