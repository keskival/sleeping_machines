"""Read-only input and differentiation contracts; no fitting or optimizer."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from native_tabular_model import contracts
from native_tabular_data import load


def test_feature_identity_row_reset_and_gradient_contracts():
    checks=contracts();assert all(checks.values())


def test_split_duplicates_and_train_only_scale():
    for dataset in ('banknote','wine_red'):
        a=load(dataset,32,32);b=load(dataset,32,32)
        np.testing.assert_array_equal(a['x_fit'],b['x_fit'])
        assert set(a['protocol']['fit_indices']).isdisjoint(a['protocol']['dev_indices'])
        np.testing.assert_allclose(a['x_fit'].mean(0),0,atol=1e-12)
        np.testing.assert_allclose(a['x_fit'].std(0),1,atol=1e-12)
        assert not a['protocol']['test_labels_scored']
        # Invert train-only normalization and verify no feature row crosses splits.
        mean=np.array(a['protocol']['preprocessing_mean']);std=np.array(a['protocol']['preprocessing_std'])
        fit=a['x_fit']*std+mean;dev=a['x_dev']*std+mean
        assert not np.isclose(fit[:,None,:],dev[None,:,:],rtol=0,atol=1e-12).all(2).any()
