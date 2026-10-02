import torch
from sleeping_machines.split_event_heads import SplitEventHeads, SharedSourcePools
from sleeping_machines.rule_seed_event_heads import RuleSeedEventHeads


def test_diagonals_preserve_original_weights_and_rng_exactly():
    for shared in (False,True):
        torch.manual_seed(19)
        old=SplitEventHeads(sources=4,payload=4,depth=2,heads=2,shared_maps=shared)
        rng=torch.get_rng_state()
        torch.manual_seed(19)
        new=RuleSeedEventHeads(sources=4,payload=4,depth=2,heads=2,shared_maps=shared,common_source_seed=shared)
        assert torch.equal(rng,torch.get_rng_state())
        assert old.state_dict().keys()==new.state_dict().keys()
        for name,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[name])


def test_off_diagonals_separate_maps_and_seed_without_sharing_state():
    for maps,common in ((True,False),(False,True)):
        model=RuleSeedEventHeads(sources=4,payload=4,depth=2,heads=2,shared_maps=maps,common_source_seed=common)
        assert isinstance(model.units[0][0],SharedSourcePools)==maps
        assert torch.equal(model.embedding.weight[0],model.embedding.weight[1])==common
        state=model.new_state()
        model.consume_event(0,0.,(1.,0.),state)
        assert all(address[2]==0 for address in state.memories)
        model.consume_event(1,.1,(-1.,0.),state)
        assert set(state.contexts)=={0,1}
