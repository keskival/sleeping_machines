import numpy as np

import experiments.count_reference_language as C


def test_smoothed_count_predictors_are_normalized_and_positive():
    rng = np.random.default_rng(0)
    fit = rng.integers(0, C.A, 600).tolist()
    for method in ('wb', 'ad', 'kn'):
        for K in (0, 1, 3):
            t = C.Counts(K)
            t.add_sequence(fit)
            kn = C.continuation_tables(fit, K) if method == 'kn' else None
            for hist in ([], fit[:1], fit[10:13], [5, 5, 5]):
                p = C.predict(t, hist[-K:] if K else [], K, method, 0.75, kn)
                assert np.all(p > 0) and abs(p.sum() - 1) < 1e-12


def test_vectorized_counts_match_incremental_counts():
    rng = np.random.default_rng(1)
    x = rng.integers(0, C.A, 300).tolist()
    a, b = C.Counts(3), C.Counts(3)
    a.add_sequence(x)
    for i in range(len(x)):
        b.add(x[max(0, i - 3):i], x[i])
    for k in range(4):
        assert set(a.t[k]) == set(b.t[k])
        for h in a.t[k]:
            assert np.array_equal(a.t[k][h], b.t[k][h])


def _cascade_race_sample(levels, rng):
    """levels: list of (symbol_rates, escape_rate) from longest context to shortest; uniform after the last escape."""
    for rates, esc in levels:
        clocks = np.where(rates > 0, rng.exponential(1.0, rates.shape) / np.maximum(rates, 1e-300), np.inf)
        e = rng.exponential(1.0) / esc if esc > 0 else np.inf
        j = int(np.argmin(clocks))
        if clocks[j] < e:
            return j
    return int(rng.integers(0, C.A))


def test_interpolated_kneser_ney_is_a_cascade_of_escape_races():
    """THEORY §377: symbol clocks max(c-D,0), escape clock D*T; escape defers to the shorter context."""
    rng = np.random.default_rng(3)
    fit = rng.integers(0, 4, 400).tolist()  # skewed small support, so escapes matter
    K, D = 2, 0.75
    t = C.Counts(K)
    t.add_sequence(fit)
    kn = C.continuation_tables(fit, K)
    hist = fit[-K:]
    p = C.predict(t, hist, K, 'kn', D, kn)
    levels = []
    for k in range(K, -1, -1):
        h = C.ctx_code(hist, k)
        c = t.t[k][h] if k == K else kn[k][h]
        T = float((c > 0).sum())
        levels.append((np.maximum(c - D, 0.0), D * T))
    draws = np.bincount([_cascade_race_sample(levels, rng) for _ in range(40000)], minlength=C.A) / 40000
    assert np.max(np.abs(draws - p)) < 0.01


def test_neural_base_gradient_is_escape_responsibility_times_cross_entropy_gradient():
    """THEORY §378: p(y)=a_y+e*softmax(z)_y gives d log p/dz = r_y (onehot_y - q), r_y = e q_y / p(y)."""
    torch = __import__('torch')
    rng = np.random.default_rng(4)
    a = torch.tensor(rng.dirichlet(np.ones(C.A)) * 0.6, dtype=torch.float64)
    e = 0.4
    z = torch.tensor(rng.normal(size=C.A), dtype=torch.float64, requires_grad=True)
    y = 5
    q = torch.softmax(z, 0)
    p = a[y] + e * q[y]
    torch.log(p).backward()
    r = float(e * q[y] / p)
    expected = r * (torch.nn.functional.one_hot(torch.tensor(y), C.A).double() - q.detach())
    assert torch.allclose(z.grad, expected, atol=1e-12)
