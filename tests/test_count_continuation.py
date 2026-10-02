import numpy as np

from sleeping_machines import count_continuation as C
from sleeping_machines.count_carrying_language import A


def _cont(seq_pairs, h, k):
    """brute force N1+(. h y): distinct preceding symbols of (h, y) over (sequence, end) pairs."""
    v = np.zeros(A)
    for y in range(A):
        pre = set()
        for x, end in seq_pairs:
            for p in range(k, end):
                if tuple(x[p - k + 1:p + 1]) == h and x[p + 1] == y:
                    pre.add(x[p - k])
        v[y] = len(pre)
    return v


def test_fit_stream_leave_one_out_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 120)
    out = C.fit_stream_continuation_counts(x, 3)
    for k in (1, 2):
        for p in range(k, 119):
            h = tuple(x[p - k + 1:p + 1])
            full = _cont([(x, 119)], h, k)
            # leave-one-out removes position p's transition; the triple disappears only if it was unique
            others = [q for q in range(k, 119) if q != p and tuple(x[q - k:q + 2]) == tuple(x[p - k:p + 2])]
            if not others:
                full[x[p + 1]] -= 1
            assert np.allclose(out[k - 1, p], full), (k, p)


def test_eval_stream_is_prequential_and_matches_brute_force():
    rng = np.random.default_rng(1)
    f, s = rng.integers(0, 4, 80), rng.integers(0, 4, 60)
    out = C.eval_stream_continuation_counts(f, s, 3)
    for k in (1, 2):
        for p in range(k, 59):
            h = tuple(s[p - k + 1:p + 1])
            pairs = [(f, 79), (s, p)]  # fit transitions plus stream transitions strictly before p
            assert np.allclose(out[k - 1, p], _cont(pairs, h, k)), (k, p)
