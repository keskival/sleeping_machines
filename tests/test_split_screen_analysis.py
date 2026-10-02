import pytest
from experiments.split_screen_analysis import paired_accuracy


def test_pair_members_remain_one_independent_cluster():
    a={'episode_accuracy':[1.,0.,1.,0.]}
    b={'episode_accuracy':[.5,.5,.5,.5]}
    result=paired_accuracy(a,b,paired_timing=True,draws=100)
    assert result['independent_clusters']==2
    assert result['accuracy_gain_pp']==0
    assert result['cluster_bootstrap_95_pp']==[0.,0.]


def test_does_not_invent_uncertainty_from_one_population_or_unmatched_arrays():
    with pytest.raises(ValueError,match='two independent'):
        paired_accuracy({'episode_accuracy':[1.]},{'episode_accuracy':[0.]})
    with pytest.raises(ValueError,match='Identical'):
        paired_accuracy({'episode_accuracy':[1.,1.]},{'episode_accuracy':[0.]})
