import copy

import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.split_event_heads import SplitEventHeads, split_evolve
from experiments.native_event_tasks import episodes
from experiments.paired_timing_tasks import paired_timing_episodes, pair_bootstrap


def forward(model, rows):
    state = model.new_state()
    outputs = []
    for e in rows:
        z, _ = model.consume_event(e.source, e.time, e.mark, state)
        if e.target is not None:
            outputs.append(z)
    return torch.stack(outputs), state


def test_zero_split_exact_parent_forward_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    parent = AddressedEventHeads(sources=2, payload=4, depth=2)
    torch.manual_seed(41)
    candidate = SplitEventHeads(sources=2, payload=4, depth=2)
    assert parent.state_dict().keys() == candidate.state_dict().keys()
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    row = episodes('order', 2, 2, 911)[0]
    for model in (parent, candidate):
        torch.manual_seed(22)
        z, _ = forward(model, row)
        z.square().sum().backward()
    torch.manual_seed(42)
    a, _ = forward(parent.eval(), row)
    torch.manual_seed(42)
    b, _ = forward(candidate.eval(), row)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for p, q in zip(parent.parameters(), candidate.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_silent_identity_semigroup_and_smooth_derivatives():
    torch.set_num_threads(1)
    value = torch.randn(8, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor([-.5, -2.], dtype=torch.float64, requires_grad=True)
    frequency = torch.tensor([.31, -.42], dtype=torch.float64, requires_grad=True)
    age = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    evolve = lambda x, t, r, w: split_evolve(x, t, r, w, 2)
    assert torch.autograd.gradcheck(evolve, (value, age, rate, frequency))
    result = evolve(value, age, rate, frequency)
    torch.testing.assert_close(result[:4], value[:4], rtol=0, atol=0)
    torch.testing.assert_close(evolve(result, age, rate, frequency),
                               evolve(value, 2*age, rate, frequency), rtol=1e-12, atol=1e-12)
    assert result.norm() <= value.norm()+1e-12
    derivative, = torch.autograd.grad(result[:4].sum(), age)
    assert derivative == 0


def test_shared_rules_have_private_states_and_address_equivariance():
    torch.set_num_threads(1)
    model = SplitEventHeads(sources=2, payload=4, depth=2, protected_pairs=1, shared_maps=True).eval()
    assert model.units[0][0][0] is model.units[0][0][1]
    row = episodes('order', 2, 2, 211)[0]
    renamed = [type(e)(1-e.source, e.time, e.mark, e.target) for e in row]
    torch.manual_seed(31)
    z, state = forward(model, row)
    torch.manual_seed(31)
    other, renamed_state = forward(model, renamed)
    torch.testing.assert_close(z, other, rtol=0, atol=0)
    for (d, h, s, i), tensor in state.memories.items():
        torch.testing.assert_close(tensor, renamed_state.memories[(d, h, 1-s, i)], rtol=0, atol=0)
    before = {k:v.clone() for k,v in state.memories.items() if k[2] == 1}
    contexts = copy.deepcopy(state.contexts[1][0].detach())
    model.consume_event(0, row[-1].time+1, (.1, 0.), state)
    for k, v in before.items():
        torch.testing.assert_close(v, state.memories[k], rtol=0, atol=0)
    torch.testing.assert_close(contexts, state.contexts[1][0], rtol=0, atol=0)
    small = SplitEventHeads(sources=1, payload=4, depth=2, shared_maps=True)
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in
        SplitEventHeads(sources=2, payload=4, depth=2).parameters())
    same = SplitEventHeads(sources=2, payload=4, depth=2, shared_maps=True)
    assert sum(p.numel() for p in small.parameters()) == sum(p.numel() for p in same.parameters())


def test_paired_timing_has_identical_rank_observation_and_opposite_labels():
    rows = paired_timing_episodes(4, 32, 918)
    assert rows == paired_timing_episodes(4, 32, 918)
    assert rows != paired_timing_episodes(4, 32, 919)
    for short, long in zip(rows[::2], rows[1::2]):
        assert [(e.source, e.mark) for e in short] == [(e.source, e.mark) for e in long]
        assert all(e.time < f.time for e, f in zip(short, long) if e.target is not None)
        assert all(e.target != f.target for e, f in zip(short, long) if e.target is not None)
    assert pair_bootstrap([.25, .75]) is None
    assert pair_bootstrap([.25, .75, .5, .5]) == [.5, .5]


def test_paired_rank_control_exact_chance_with_coupled_noise():
    from experiments.split_event_benchmark import evaluate
    torch.set_num_threads(1)
    model = SplitEventHeads(sources=4, payload=4, depth=2, classes=2,
                            shared_maps=True, protected_pairs=1)
    rows = paired_timing_episodes(4, 32, 918)
    score = evaluate(model, rows, 'rank', paired=True)
    assert score['accuracy'] == .5
    assert score['independent_clusters'] == 4
    assert score['episode_bootstrap_95_percent_interval'] == [.5, .5]
    # A population interval cannot quantify independent model-seed variation.


def test_integrated_split_forward_backward_operator_coverage():
    from experiments.race_language_screen import capture
    from experiments.split_event_contracts import predict
    torch.set_num_threads(1)
    model = SplitEventHeads(sources=4, payload=8, depth=8, classes=2,
                            shared_maps=True, protected_pairs=2).train()
    row = paired_timing_episodes(4, 8, 918)[0]
    box = {}
    def run():
        logits, labels, state, _ = predict(model, row)
        box['loss'] = torch.nn.functional.cross_entropy(logits, labels)
        box['state'] = state
    forward_trace = capture(run)
    backward_trace = capture(lambda:box['loss'].backward())
    for trace in (forward_trace, backward_trace):
        assert trace['formula_coverage_complete']
    assert torch.isfinite(box['loss'])
    assert box['state'].counterfactual_values == box['state'].candidate_scores == 512
    assert box['state'].selected_updates == 256
    for depth in model.queries:
        for head in depth:
            assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
