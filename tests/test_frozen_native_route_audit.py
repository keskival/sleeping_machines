import torch

from experiments.frozen_native_route_audit import replay, weight_hash
from experiments.native_event_tasks import episodes
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def test_frozen_replay_preserves_checkpoint_and_common_noise():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(31)
        model = AddressedEventHeads(sources=4, payload=4, depth=2, heads=1, pool=2)
        row = episodes('order', 4, 4, 2201)[0]
        before = weight_hash(model)
        baseline = replay(model, row, 0, 619, gradients=True)
        branches = [replay(model, row, 0, 619, forced=i) for i in (0, 1)]
        assert baseline['loss'] == branches[baseline['winner']]['loss']
        for branch in branches:
            assert baseline['delay'] == branch['delay']
            assert baseline['races'] == branch['races']
            assert baseline['commits'] == branch['commits']
            assert torch.equal(baseline['rng'], branch['rng'])
        assert before == weight_hash(model)
        assert model.race.__func__ is AddressedEventHeads.race
