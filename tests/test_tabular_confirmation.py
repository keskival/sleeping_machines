"""Read-only leakage, probability and paired-uncertainty contracts; no fitting."""
import copy

import numpy as np
import pytest

from experiments.native_tabular_data import load
from experiments.tabular_confirmation_data import confirmation_rows,prediction_rows
from experiments.tabular_confirmation_analysis import paired_summary
from experiments.tabular_confirmation_controls import candidates


def test_reserved_features_use_fitting_scaler_and_disjoint_groups():
    data=load('banknote',128,128);p=data['protocol'];test=confirmation_rows(p)
    assert set(test['indices']).isdisjoint(p['fit_indices']+p['dev_indices'])
    # Verify all three folds remain feature-disjoint, including duplicate records.
    assert not np.isclose(test['x'][:,None,:],data['x_fit'][None,:,:],rtol=0,atol=1e-12).all(2).any()
    assert not np.isclose(test['x'][:,None,:],data['x_dev'][None,:,:],rtol=0,atol=1e-12).all(2).any()
    assert len(test['groups'])==len(test['indices'])


def records(probabilities):
    rows=prediction_rows(np.log(probabilities),[0,1,0,1],[11,12,13,14]);rows['groups']=[1,2,3,4]
    return [dict(status='completed',args=dict(seed=seed),protocol=dict(test_labels_scored=True,
        weights_frozen_before_test=True,fit_sha256='fit',dev_sha256='dev',test_sha256='test'),
        final=dict(test=dict(rows=copy.deepcopy(rows)))) for seed in (6,7,8)]


def test_paired_identical_scores_have_zero_gap_and_no_false_precision():
    rows=records([[.8,.2],[.2,.8],[.7,.3],[.3,.7]])
    summary=paired_summary(rows,rows,draws=100)
    assert summary['mean_control_minus_ours_nll']==0
    assert summary['nll_interval']==[0.,0.]
    assert summary['test_rows']==4 and summary['seeds']==3
    with pytest.raises(ValueError,match='three'):paired_summary(rows[:1],rows[:1])


def test_pairing_rejects_identity_and_prediction_corruption():
    ours=records([[.8,.2],[.2,.8],[.7,.3],[.3,.7]])
    control=records([[.6,.4],[.4,.6],[.5,.5],[.5,.5]])
    assert paired_summary(ours,control,draws=100)['mean_control_minus_ours_nll']>0
    bad=copy.deepcopy(control);bad[0]['final']['test']['rows']['indices'][0]=999
    with pytest.raises(ValueError,match='identity'):paired_summary(ours,bad)
    bad=copy.deepcopy(control);bad[0]['final']['test']['rows']['losses'][0]+=1
    with pytest.raises(ValueError,match='losses'):paired_summary(ours,bad)


def test_control_candidates_are_fixed_and_catboost_is_single_thread_cpu():
    for family in ('trees','catboost','logistic'):
        assert len(candidates(family,6))==4
    for _,model in candidates('catboost',6):
        p=model.get_params();assert p['thread_count']==1 and p['task_type']=='CPU'
        assert p['use_best_model'] is False and p['allow_writing_files'] is False


def test_confirmation_evaluation_preserves_parent_values_weights_and_rng():
    import torch
    from experiments.native_tabular_benchmark import evaluate as parent_evaluate
    from experiments.tabular_confirmation_benchmark import evaluate
    from experiments.native_tabular_model import NativeTabularModel
    torch.manual_seed(6);model=NativeTabularModel(4,2)
    x=torch.tensor([[.2,-.3,.4,-.1],[-.7,.3,.2,.8]])
    y=np.asarray([0,1]);before={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    original=parent_evaluate(model,x,y,True)
    confirmed=evaluate(model,x,y,True,[10,20])
    for k in ('n','accuracy','nll'):assert original[k]==confirmed[k]
    assert confirmed['rows']['indices']==[10,20]
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
