import numpy as np
import torch
from torch.nn import functional as F

from sleeping_machines import count_carrying_language as M
from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel
from sleeping_machines.statistic_race_memory import StatisticRaceNativeModel


def _pair(writes=True):
    torch.manual_seed(0)
    plain = GatedCountCarryingNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2, count_message=False).double()
    torch.manual_seed(0)
    race = StatisticRaceNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2, addresses=8, key_dim=4,
                                    writes=writes).double()
    race.load_state_dict({**race.state_dict(), **plain.state_dict()})
    rng = np.random.default_rng(4)
    tokens = torch.tensor(rng.integers(0, 5, 40))
    counts = M.eval_stream_counts(rng.integers(0, 5, 200), tokens.numpy(), 2)
    for m in (plain, race):
        m.register_stream('s', counts); m.use_stream('s')
    return plain, race, tokens


def _run(model, tokens, chunks=(40,)):
    with torch.random.fork_rng():
        torch.manual_seed(1)
        state, out, s = model.new_state(), [], 0
        for size in chunks:
            z, state = model.forward_chunk(tokens[s:s + size], state); out.append(z); s += size
            state = state.detach()
    return torch.cat(out), state


def test_empty_receivers_nest_the_gated_count_model():
    plain, race, tokens = _pair(writes=False)
    with torch.no_grad():
        plain.escape_gate.weight.normal_(0, .3); race.escape_gate.weight.copy_(plain.escape_gate.weight)
    assert torch.allclose(_run(plain, tokens)[0], _run(race, tokens)[0], atol=1e-12, rtol=0)


def test_pooled_level_is_the_exact_race_expectation():
    _, race, _ = _pair()
    rng = np.random.default_rng(1)
    counts = torch.tensor(rng.integers(0, 3, (8, 27)), dtype=torch.float64); counts[3] = 0
    log_q = F.log_softmax(torch.randn(1, 27, dtype=torch.float64), -1)
    pi = torch.softmax(torch.randn(1, 8, dtype=torch.float64), -1)
    D, th = torch.sigmoid(race.raw_pool_discount), F.softplus(race.raw_pool_theta)
    q = log_q.exp()[0]; expect = torch.zeros(27, dtype=torch.float64)
    for a in range(8):
        c = counts[a]; n = c.sum(); T = (c > 0).sum()
        pa = q if n == 0 else (torch.clamp(c - D, min=0) + (th + D * T) * q) / (n + th)
        expect = expect + pi[0, a] * pa
    got = race.pooled(log_q, pi, counts)[0]
    assert torch.allclose(got, expect, atol=1e-12) and abs(float(got.sum()) - 1) < 1e-12


def test_writes_are_causal_chunk_invariant_and_train_keys():
    _, race, tokens = _pair()
    whole, state = _run(race, tokens)
    assert state.pool_writes == 39 and float(state.pool_counts.sum()) == 39
    assert torch.allclose(whole, _run(race, tokens, (16, 16, 8))[0], atol=1e-10)
    changed = tokens.clone(); changed[30:] = (changed[30:] + 1) % 5
    assert torch.allclose(whole[:30], _run(race, changed)[0][:30], atol=1e-12)
    assert torch.allclose(whole.exp().sum(-1), torch.ones(40, dtype=torch.float64), atol=1e-10)
    race.train()
    z, _ = _run(race, tokens)
    F.nll_loss(z[:-1], tokens[1:]).backward()
    for p in (race.pool_keys, race.pool_query.weight, race.pool_log_temperature, race.embedding.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_seeded_state_starts_from_saved_fit_counts():
    _, race, _ = _pair()
    race.pool_seed.fill_(1.)
    assert float(race.new_state().pool_counts.sum()) == 0
    race.seed_pool = True
    assert float(race.new_state().pool_counts.sum()) == 8 * 27
