"""Finite-difference audit of every parameter downstream of the core's races (count composition models).

Given fixed race noise (same seed), the head, count message, escape gate, escape discount/concentration and the
statistic-race router (query, keys, temperature, pool discount/concentration) enter the loss smoothly, so
autograd must match central differences.  Core race parameters use the declared counterfactual surrogate and
are only checked for nonzero reach.  Discrete pooled writes can switch under a perturbation; coordinates whose
two-sided differences disagree in curvature are skipped and counted.
"""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sleeping_machines import count_carrying_language as M
from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel
from sleeping_machines.statistic_race_memory import StatisticRaceNativeModel
from sleeping_machines.statistic_race_sampled import SampledStatisticRaceNativeModel
from sleeping_machines.statistic_race_top import TopStatisticRaceNativeModel

KW = dict(payload=4, depth=2, pool=2, heads=2, orders=2)
DOWNSTREAM = ('head.', 'count_message.', 'escape_gate.', 'raw_discount', 'raw_theta', 'pool_query.', 'pool_keys',
              'pool_log_temperature', 'raw_pool_discount', 'raw_pool_theta')


def _build(kind):
    torch.manual_seed(0)
    if kind == 'gated':
        m = GatedCountCarryingNativeModel(**KW, escape_gate=True, count_message=True)
    else:
        cls = dict(bottom=StatisticRaceNativeModel, sampled=SampledStatisticRaceNativeModel,
                   top=TopStatisticRaceNativeModel)[kind]
        m = cls(**KW, addresses=6, key_dim=4, count_message=True)
    m = m.double()
    with torch.no_grad():  # move every zero-initialised repair off its nesting point
        m.escape_gate.weight.normal_(0, .3); m.count_message.weight.normal_(0, .1)
        if hasattr(m, 'pool_keys'):
            m.pool_keys.mul_(4)
    rng = np.random.default_rng(4)
    tokens = torch.tensor(rng.integers(0, 6, 33))
    m.register_stream('s', M.eval_stream_counts(rng.integers(0, 6, 300), tokens.numpy(), 2)); m.use_stream('s')
    m.train()
    return m, tokens


def _loss(m, tokens, chunk=16):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        state, total = m.new_state(), 0.
        for s in range(0, 32, chunk):
            z, state = m.forward_chunk(tokens[s:s + chunk], state)
            total = total + F.nll_loss(z, tokens[s + 1:s + chunk + 1], reduction='sum')
            state = state.detach()
    return total


@pytest.mark.parametrize('kind', ['gated', 'bottom', 'sampled', 'top'])
def test_downstream_gradients_match_central_differences(kind):
    m, tokens = _build(kind)
    m.zero_grad(); _loss(m, tokens).backward()
    rng = np.random.default_rng(1)
    checked = skipped = 0
    for name, p in m.named_parameters():
        if not name.startswith(DOWNSTREAM):
            continue
        flat = p.data.view(-1)
        for i in rng.choice(flat.numel(), size=min(4, flat.numel()), replace=False):
            old, eps = float(flat[i]), 1e-5
            with torch.no_grad():
                flat[i] = old + eps; up = float(_loss(m, tokens))
                flat[i] = old - eps; down = float(_loss(m, tokens))
                flat[i] = old + eps / 2; up2 = float(_loss(m, tokens))
                flat[i] = old
            fd, fd2 = (up - down) / (2 * eps), (up2 - float(_loss(m, tokens))) / (eps / 2)
            if abs(fd - fd2) > 1e-3 * max(1., abs(fd)):  # a discrete write switched inside the bracket
                skipped += 1; continue
            g = float(p.grad.view(-1)[i])
            assert abs(g - fd) <= 1e-5 * max(1., abs(fd)), (kind, name, int(i), g, fd)
            checked += 1
    assert checked >= 20 and skipped <= checked // 4, (checked, skipped)


@pytest.mark.parametrize('kind', ['gated', 'top'])
def test_core_race_parameters_receive_credit(kind):
    m, tokens = _build(kind)
    m.zero_grad(); _loss(m, tokens).backward()
    reached = {n.split('.')[0] for n, p in m.named_parameters()
               if not n.startswith(DOWNSTREAM) and p.grad is not None and p.grad.abs().sum() > 0}
    assert {'embedding', 'content', 'channel_mix', 'queries', 'units'} <= reached, reached


def test_whole_core_pathwise_gradients_match_central_differences():
    """With the diagnostic pathwise race (winner fixed by the noise, exact interior clock/value derivatives), the
    forward is piecewise smooth in every parameter; one undetached 32-target chunk (truncated credit at chunk
    boundaries is by design and is excluded), so all of the core's plumbing (embeddings, channel mixes,
    queries, units' memories/rotations/gates, transport, context gates, layer norms) must match finite differences."""
    m, tokens = _build('gated')
    m.credit = 'pathwise'
    m.zero_grad(); _loss(m, tokens, 32).backward()
    rng = np.random.default_rng(2)
    checked = skipped = 0
    groups = {}
    for name, p in m.named_parameters():
        groups.setdefault(name.split('.')[0], []).append((name, p))
    for group, params in groups.items():
        for name, p in params[:3]:
            flat = p.data.view(-1)
            for i in rng.choice(flat.numel(), size=min(2, flat.numel()), replace=False):
                old, eps = float(flat[i]), 1e-6
                with torch.no_grad():
                    flat[i] = old + eps; up = float(_loss(m, tokens, 32))
                    flat[i] = old - eps; down = float(_loss(m, tokens, 32))
                    flat[i] = old + eps / 2; up2 = float(_loss(m, tokens, 32))
                    flat[i] = old; base = float(_loss(m, tokens, 32))
                fd, fd2 = (up - down) / (2 * eps), (up2 - base) / (eps / 2)
                if abs(fd - fd2) > 1e-3 * max(1., abs(fd)):  # race winner switched inside the bracket
                    skipped += 1; continue
                g = float(p.grad.view(-1)[i])
                assert abs(g - fd) <= 1e-4 * max(1., abs(fd)), (name, int(i), g, fd)
                checked += 1
    assert checked >= 30 and skipped <= checked // 4, (checked, skipped)
