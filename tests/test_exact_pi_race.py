import torch

from sleeping_machines.exact_pi_race import ExactPiRoute
from sleeping_machines.sparse_race_language import TemporalRoute


def test_forward_matches_temporal_route_exactly_with_the_same_noise():
    s = torch.randn(4, dtype=torch.float64); v = torch.randn(4, 6, dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(1); a = TemporalRoute.apply(s, v)
        torch.manual_seed(1); b = ExactPiRoute.apply(s, v)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and int(a[2]) == int(b[2])


def test_score_credit_equals_the_expectation_of_the_original_teacher():
    torch.manual_seed(0)
    s = torch.randn(3, dtype=torch.float64, requires_grad=True); v = torch.randn(3, 5, dtype=torch.float64)
    g = torch.randn(5, dtype=torch.float64)
    out, _, _ = ExactPiRoute.apply(s, v); (out @ g).backward(); exact = s.grad.clone()
    pi = torch.softmax(s.detach(), 0); d = v @ g
    torch.testing.assert_close(exact, pi * (d - (pi * d).sum()), rtol=0, atol=1e-12)
    total = torch.zeros(3, dtype=torch.float64)
    for k in range(20000):                       # Monte-Carlo mean of the original single-sample teacher
        s.grad = None
        with torch.random.fork_rng():
            torch.manual_seed(10 + k); o, _, _ = TemporalRoute.apply(s, v)
        (o @ g).backward(); total += s.grad
    assert (total / 20000 - exact).abs().max() < .03 * exact.abs().max() + 1e-3


def test_exact_pi_model_fast_path_matches_reference_and_scores_learn():
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, 'experiments')
    import dvs_exact_pi_benchmark as X
    a = SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6)
    ref, fast = X.make_model(a, fast=False).double(), X.make_model(a, fast=True).double()
    fast.load_state_dict(ref.state_dict())
    outs = []
    for m in (ref, fast):
        m.train(); m.zero_grad()
        if hasattr(m, '_fast_layers'):
            m._fast_layers = {}
        with torch.random.fork_rng():
            torch.manual_seed(4); state = m.new_state()
            for k in range(10):
                z, _ = m.consume_event(0, (k + 1) * .05, torch.randn(33, generator=torch.Generator().manual_seed(k)).double(), state)
        if hasattr(m, '_fast_layers'):
            m._fast_layers = None
        torch.nn.functional.cross_entropy(z[None], torch.tensor([2])).backward(); outs.append(z.detach())
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-11)
    for (name, p), q in zip(ref.named_parameters(), fast.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, rtol=1e-9, atol=1e-11, msg=name)
    assert ref.units[0][0][0][0].key.grad.abs().sum() > 0
