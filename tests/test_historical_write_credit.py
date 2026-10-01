"""Read-only numerical contracts for delayed write credit; no optimizer fits."""
import copy
import io

import torch

from sleeping_machines.historical_write_credit import (
    HistoricalKeyMatch, HistoricalValueCredit, PackedFeatureBank,
)
from sleeping_machines.historical_write_race_language import HistoricalWriteRaceLanguageModel
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel
from sleeping_machines.sparse_race_language import TemporalRoute


def test_key_factorization_is_exact_conditional_common_write_perturbation():
    torch.manual_seed(991)
    q = torch.randn(6, dtype=torch.float64, requires_grad=True)
    keys = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    features = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 2, 4]); error = torch.randn(5, dtype=torch.float64)
    result = HistoricalKeyMatch.apply(q, keys, features, weight, ids, .25)
    (result @ error).backward()
    torch.testing.assert_close(result, keys @ q, rtol=0, atol=0)
    delta = torch.zeros_like(weight, requires_grad=True)
    objective = ((features.detach() @ delta.T) @ q.detach()) @ error[ids] * .25
    expected, = torch.autograd.grad(objective, delta)
    torch.testing.assert_close(weight.grad, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(q.grad, keys.detach().T @ error, rtol=0, atol=0)
    torch.testing.assert_close(keys.grad, error[:, None]*q.detach()[None, :], rtol=0, atol=0)
    assert features.grad is None
    direction = torch.randn_like(delta); eps = 1e-5
    def local(t):
        return (((features.detach() @ (t*direction).T) @ q.detach()) @ error[ids])*.25
    finite = (local(eps)-local(-eps))/(2*eps)
    torch.testing.assert_close(finite, (weight.grad*direction).sum(), rtol=1e-12, atol=1e-12)


def test_value_credit_preserves_primal_and_winner_adjoints():
    torch.manual_seed(992)
    value = torch.randn(8, requires_grad=True)
    feature = torch.randn(8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)
    error = torch.randn(8)
    result = HistoricalValueCredit.apply(value, feature, weight, .25)
    (result @ error).backward()
    torch.testing.assert_close(result, value, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, error, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.outer(error, feature.detach())*.25, rtol=0, atol=0)
    assert feature.grad is None


def coupled_credit(alpha, detach=True):
    torch.manual_seed(993)
    parent = ParallelHeadRaceLanguageModel(8, 2, 2, heads=2, block_size=7)
    child = HistoricalWriteRaceLanguageModel(8, 2, 2, heads=2, block_size=7,
                                             historical_credit=alpha)
    child.load_state_dict(parent.state_dict())
    tokens = torch.tensor([1, 2, 1, 3, 1, 2, 4, 1])
    rng = torch.get_rng_state()
    _, a = parent.forward_chunk(tokens[:-1])
    if detach:
        a.detach()
    z = parent.consume(tokens[-1], a); z.square().sum().backward()
    end_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    _, b = child.forward_chunk(tokens[:-1])
    if detach:
        b.detach()
    w = child.consume(tokens[-1], b); w.square().sum().backward()
    torch.testing.assert_close(z, w, rtol=0, atol=0)
    assert torch.equal(end_rng, torch.get_rng_state())
    for (name, p), (_, q) in zip(parent.named_parameters(), child.named_parameters()):
        if alpha > 0 and detach and name.startswith(('kv_key.', 'kv_value.')):
            assert p.grad is None
            assert q.grad is not None and bool(q.grad.abs().sum() > 0)
        else:
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
    return child, b


def test_disabled_credit_exactly_nests_parent_after_detach():
    coupled_credit(0.)


def test_enabled_credit_changes_only_old_write_map_gradients():
    _, state = coupled_credit(1.)
    assert state.historical_key_queries == 4
    assert state.historical_value_deliveries == 4


def test_live_writes_are_not_taught_twice():
    _, state = coupled_credit(1., detach=False)
    assert state.historical_key_candidates == 0
    assert state.historical_value_deliveries == 0


def test_feature_bank_lossless_graph_free_checkpoint_and_storage():
    bank = PackedFeatureBank(3)
    for i in range(7):
        bank.append(torch.full((4,), float(i), requires_grad=True))
    saved = io.BytesIO(); torch.save(bank, saved); saved.seek(0)
    restored = torch.load(saved, weights_only=False)
    for i in range(7):
        assert not restored[i].requires_grad
        torch.testing.assert_close(restored[i], torch.full((4,), float(i)), rtol=0, atol=0)
    first = restored[0].clone(); restored.append(torch.ones(4))
    torch.testing.assert_close(restored[0], first, rtol=0, atol=0)
    _, state = coupled_credit(0.)
    stats = state.packed_storage()
    assert stats['eligibility_entries'] == stats['entries']
    assert stats['allocated_eligibility_bytes']*2 == stats['allocated_key_value_bytes']
    clone = copy.deepcopy(state.detach())
    assert clone.detached_until == 8
    assert all(not row.requires_grad for b in clone.write_features for row in b.slabs)


def test_common_score_mode_preserves_choice_but_teaches_arrival_time():
    scores = torch.tensor([-.4, .1, .7], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[1., -2.], [3., 4.], [-1., 2.]], dtype=torch.float64)
    error = torch.tensor([.3, -.7], dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(1117)
        noise = torch.empty_like(scores).exponential_()
        raw, expected_winner = (noise/scores.detach().exp()).min(0)
        torch.manual_seed(1117)
        value, delay, winner = TemporalRoute.apply(scores, values)
        torch.manual_seed(1117)
        shifted_value, shifted_delay, shifted_winner = TemporalRoute.apply(scores+.3, values)
        assert int(winner) == int(expected_winner) == int(shifted_winner)
        torch.testing.assert_close(value, shifted_value, rtol=0, atol=0)
        shifted_raw = raw*torch.exp(torch.tensor(-.3, dtype=torch.float64))
        torch.testing.assert_close(shifted_delay, .001+.010*shifted_raw/(1+shifted_raw), rtol=1e-14, atol=1e-15)
        gradient, = torch.autograd.grad(value @ error+7*delay, scores)
        expected = -7*.010*raw/(1+raw).square()
        torch.testing.assert_close(gradient.sum(), expected, rtol=1e-12, atol=1e-14)
        eps = 1e-5
        def conditional_time(offset):
            torch.manual_seed(1117)
            return TemporalRoute.apply(scores+offset, values)[1]
        finite = (conditional_time(eps)-conditional_time(-eps))/(2*eps)
        torch.testing.assert_close(7*finite, expected, rtol=1e-9, atol=1e-12)
