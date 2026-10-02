"""Read-only algebra and replay integrity; no optimizer or fitting."""
import torch

from experiments.native_event_tasks import episodes
from experiments.route_write_decomposition import replay,components
from experiments.frozen_native_route_audit import weight_hash
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def test_factorial_residual_identity_and_complementary_effects():
    c=components(1.,2.,4.,8.,.5)
    assert c['delivered_value_effect']==1.
    assert c['persistent_commit_effect']==3.
    assert c['delivery_commit_interaction']==3.
    assert c['teacher_residual']==c['residual_reconstruction']==6.5


def test_hybrid_replays_preserve_noise_time_and_realized_route():
    with torch.random.fork_rng():
        torch.manual_seed(117)
        model=AddressedEventHeads(sources=4,payload=8,depth=8,heads=2,pool=2)
        row=episodes('order',4,4,2201)[0];before=weight_hash(model)
        base=replay(model,row,0,619,gradients=True)
        realized=replay(model,row,0,619,delivery=base['winner'],commit=base['winner'])
        assert realized['loss']==base['loss']
        for delivery,commit in ((1-base['winner'],base['winner']),(base['winner'],1-base['winner'])):
            branch=replay(model,row,0,619,delivery=delivery,commit=commit)
            assert branch['delay']==base['delay']
            assert torch.equal(branch['rng'],base['rng'])
            assert branch['races']==base['races'] and branch['commits']==base['commits']
        assert before==weight_hash(model)
