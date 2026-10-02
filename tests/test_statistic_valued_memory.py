"""Numerical contracts for THEORY §§382-383 (statistic-valued race memory)."""
from math import lgamma

import numpy as np

A, ALPHA = 27, .5


def codelen(c):
    n = c.sum()
    return -(sum(lgamma(v + ALPHA) - lgamma(ALPHA) for v in c) + lgamma(A * ALPHA) - lgamma(n + A * ALPHA))


def sequential_codelen(seq):
    c = np.zeros(A)
    total = 0.0
    for y in seq:
        total -= np.log((c[y] + ALPHA) / (c.sum() + A * ALPHA))
        c[y] += 1
    return total


def test_dirichlet_code_length_is_order_invariant_and_matches_sequential_prediction():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, A, 200)
    c = np.bincount(seq, minlength=A)
    assert abs(sequential_codelen(seq) - codelen(c)) < 1e-9
    assert abs(sequential_codelen(rng.permutation(seq)) - codelen(c)) < 1e-9


def test_pooling_sign_follows_divergence_and_evidence():
    rng = np.random.default_rng(1)
    q1 = rng.dirichlet(np.ones(A) * .3)
    q2 = .7 * q1 + .3 * rng.dirichlet(np.ones(A) * .3)
    gains = {}
    for n in (20, 20000):
        g = []
        for _ in range(20):
            c1 = np.bincount(rng.choice(A, n, p=q1), minlength=A)
            c2 = np.bincount(rng.choice(A, n, p=q2), minlength=A)
            g.append(codelen(c1) + codelen(c2) - codelen(c1 + c2))
        gains[n] = np.mean(g)
    assert gains[20] > 0 > gains[20000]


def test_closed_form_expected_write_credit_is_exact():
    rng = np.random.default_rng(2)
    for _ in range(200):
        q = rng.dirichlet(np.ones(A) * .4)
        n = int(rng.integers(5, 300))
        c = np.bincount(rng.choice(A, n, p=q), minlength=A).astype(float)
        y = int(rng.integers(A))
        s = (c + ALPHA) / (n + A * ALPHA)
        w = c.copy()
        w[y] += 1
        s2 = (w + ALPHA) / (n + 1 + A * ALPHA)
        exact = float(np.sum(q * (np.log(s2) - np.log(s))))
        closed = q[y] * np.log1p(1 / (c[y] + ALPHA)) - np.log1p(1 / (n + A * ALPHA))
        assert abs(exact - closed) < 1e-12


def test_expected_route_loss_gradient_is_exact_over_all_candidates():
    torch = __import__('torch')
    rng = np.random.default_rng(3)
    ell = torch.tensor(rng.exponential(size=6), dtype=torch.float64)
    z = torch.tensor(rng.normal(size=6), dtype=torch.float64, requires_grad=True)
    pi = torch.softmax(z, 0)
    (pi * ell).sum().backward()
    p = pi.detach()
    assert torch.allclose(z.grad, p * (ell - (p * ell).sum()), atol=1e-12)
