"""Read-only checkpoint/probability-population checks; never optimizer steps."""
import copy

import pytest
import torch

from experiments.split_confirmation_analysis import comparison
from experiments.split_event_confirmation import restore,weight_hash
from experiments.split_event_benchmark import episodes,evaluate,sources
from experiments.native_event_tasks import data_hash
from sleeping_machines.split_event_heads import SplitEventHeads


def records(scores):
    scores=scores*(256//len(scores))
    return [dict(status='completed',args=dict(seed=s,task='paired_timing',sources=4),data_sha256=dict(fit='fit',dev='dev'),
        protocol=dict(confirmation_data_seed=3201,synthetic_holdout_read=True),
        final=dict(confirmation=dict(n=1024,accuracy=sum(scores)/len(scores),episode_accuracy=scores)),
        work=dict(total_training_unit_special_flops=100)) for s in (6,7,8)]


def test_same_seed_examples_are_not_pooled_as_independent():
    a=records([1.,0.,1.,0.]);b=records([.5,.5,.5,.5])
    summary=comparison(a,b,paired_timing=True,draws=100)
    assert summary['independent_populations']==128
    assert summary['adjusted_97_5_interval_pp']==[0.,0.]
    with pytest.raises(ValueError,match='All prespecified'):comparison(a[:1],b[:1])
    bad=copy.deepcopy(b);bad[1]['data_sha256']['fit']='other'
    with pytest.raises(ValueError,match='data differ'):comparison(a,bad)


def test_restore_uses_selected_weights_and_rejects_protocol_or_score_mismatch():
    with torch.random.fork_rng():
        torch.manual_seed(6);model=SplitEventHeads(sources=4,classes=4,payload=4,depth=2,pool=2,heads=2)
        a=dict(task='order',sources=4,payload=4,depth=2,pool=2,heads=2,credit='counterfactual',
            shared_maps=False,protected_pairs=0,fit_targets=4,dev_targets=4,time_input='observed',seed=6)
        dev=evaluate(model,episodes('order',4,4,2201))
        row=dict(status='completed',args=a,source_sha256=sources(),selected_epoch=1,
            data_sha256={key:data_hash(episodes('order',4,4,seed)) for key,seed in (('fit',1201),('dev',2201))},
            parameters=sum(p.numel() for p in model.parameters()),work={},final=dict(dev=dev))
        saved=dict(result=row,best_state=copy.deepcopy(model.state_dict()))
        restored,score=restore(saved,row)
        assert weight_hash(restored)==weight_hash(model)
        assert score['nll']==dev['nll']
        bad=copy.deepcopy(row);bad['final']['dev']['nll']+=.01
        with pytest.raises(ValueError,match='score differs'):restore(saved,bad)
        bad=copy.deepcopy(row);bad['selected_epoch']=2
        with pytest.raises(ValueError,match='lineage differs'):restore(saved,bad)
