import numpy as np
import torch
from torch.nn import functional as F

from sleeping_machines.statistic_race_top import TopStatisticRaceNativeModel
from tests.test_statistic_race_memory import _pair, _run


def _top():
    plain, race, tokens = _pair()
    torch.manual_seed(0)
    top = TopStatisticRaceNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2, addresses=8, key_dim=4).double()
    top.load_state_dict(race.state_dict()); top.streams, top.active = race.streams, race.active
    return plain, top, tokens


def test_empty_top_receivers_nest_the_gated_count_model():
    plain, top, tokens = _top()
    top.writes = False
    with torch.no_grad():
        plain.escape_gate.weight.normal_(0, .3); top.escape_gate.weight.copy_(plain.escape_gate.weight)
    assert torch.allclose(_run(plain, tokens)[0], _run(top, tokens)[0], atol=1e-12, rtol=0)


def test_top_level_is_causal_normalized_chunk_invariant_and_trains_router():
    _, top, tokens = _top()
    whole, state = _run(top, tokens)
    assert state.pool_writes == 39
    assert torch.allclose(whole.exp().sum(-1), torch.ones(40, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(whole, _run(top, tokens, (16, 16, 8))[0], atol=1e-10)
    changed = tokens.clone(); changed[30:] = (changed[30:] + 1) % 5
    assert torch.allclose(whole[:30], _run(top, changed)[0][:30], atol=1e-12)
    top.train()
    z, _ = _run(top, tokens)
    F.nll_loss(z[:-1], tokens[1:]).backward()
    for p in (top.pool_keys, top.pool_query.weight, top.embedding.weight, top.escape_gate.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_receiver_with_evidence_claims_mass_directly():
    _, top, _ = _top()
    exact = F.log_softmax(torch.zeros(1, 27, dtype=torch.float64), -1)  # uniform exact cascade
    counts = torch.zeros(8, 27, dtype=torch.float64); counts[0, 3] = 50
    pi = torch.zeros(1, 8, dtype=torch.float64); pi[0, 0] = 1
    p = top.pooled(exact, pi, counts)[0]
    assert p[3] > .95  # first-order: no product of exact-order escapes in between
