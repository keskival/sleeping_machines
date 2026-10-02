import torch

from sleeping_machines.factorized_race import FactorizedRace, delay_of, force_at_first_time


def test_time_only_witness_has_the_true_expected_score_derivative():
    """note 92 witness: rates 1 and 3, loss = T (here through d(delay)/dT, undone exactly).  True d E[T] / d s_i =
    -pi_i / Lambda = [-1/16, -3/16]; choice credit is zero because the loss does not depend on the winner."""
    s = torch.log(torch.tensor([1., 3.], dtype=torch.float64)).requires_grad_()
    v = torch.zeros(2, 1, dtype=torch.float64)
    total = torch.zeros(2, dtype=torch.float64); n = 40000
    for k in range(n):
        with torch.random.fork_rng():
            torch.manual_seed(k)
            _, delay, _ = FactorizedRace.apply(s, v)
        time = delay.detach()  # recover T from delay: delay = .001 + .010 T/(1+T)
        T = (delay.detach() - .001) / (.010 - (delay.detach() - .001))
        weight = (1 + T) ** 2 / .010                       # d T / d delay, so the loss T has gradient weight * d delay
        s.grad = None; (weight * delay).backward(); total += s.grad
    estimate = total / n
    torch.testing.assert_close(estimate, torch.tensor([-1 / 16, -3 / 16], dtype=torch.float64), rtol=0, atol=.006)


def test_forced_alternative_keeps_the_factual_first_time_and_rng_consumption():
    s = torch.randn(3, dtype=torch.float64); v = torch.randn(3, 4, dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(5); _, d_fact, w = FactorizedRace.apply(s, v); after = torch.rand(1)
    for i in range(3):
        with torch.random.fork_rng():
            torch.manual_seed(5); val, d_i, idx = force_at_first_time(s, v, i); after_i = torch.rand(1)
        assert torch.equal(d_i, d_fact) and int(idx) == i and torch.equal(val, v[i]) and torch.equal(after_i, after)


def test_choice_credit_from_first_time_replays_is_exact_for_winner_dependent_losses():
    torch.manual_seed(0)
    s = torch.randn(3, dtype=torch.float64, requires_grad=True); c = torch.randn(3, dtype=torch.float64)
    route = (torch.softmax(s, 0) * c).sum()               # E_W[c_W] marginal, F_i = c_i at the factual first time
    route.backward()
    pi = torch.softmax(s.detach(), 0)
    torch.testing.assert_close(s.grad, pi * (c - (pi * c).sum()), rtol=0, atol=1e-12)
