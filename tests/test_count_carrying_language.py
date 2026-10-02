"""Contracts for count-carrying receivers (THEORY §380): normalization, causality, credit, chunk alignment."""
import numpy as np
import torch
from torch.nn import functional as F

import experiments.count_reference_language as R
from sleeping_machines import count_carrying_language as M


def _counts_from(rng, K, L):
    return torch.tensor(rng.integers(0, 3, (K, L, M.A)) * (rng.random((K, L, M.A)) < .2), dtype=torch.float64)


def test_cascade_is_normalized_and_positive():
    rng = np.random.default_rng(0)
    c = _counts_from(rng, 3, 40)
    log_q = F.log_softmax(torch.randn(40, M.A, dtype=torch.float64), -1)
    lp = M.compose(log_q, c, torch.tensor([.3, .75, .9], dtype=torch.float64), torch.tensor([.1, 1., 5.], dtype=torch.float64))
    assert torch.allclose(lp.exp().sum(-1), torch.ones(40, dtype=torch.float64), atol=1e-12)
    assert torch.isfinite(lp).all()


def test_cascade_matches_reference_absolute_discounting_with_zero_concentration():
    rng = np.random.default_rng(1)
    fit = rng.integers(0, M.A, 500).tolist()
    K = 2
    t = R.Counts(K)
    t.add_sequence(fit)
    hist = fit[-K:]
    counts = torch.tensor(np.stack([t.t[k][R.ctx_code(hist, k)] for k in range(1, K + 1)])[:, None, :])
    # reference cascade includes order 0 over a uniform floor; use that order-0 predictive as our base q
    q0 = R.predict(t, [], 0, 'ad', .75)
    lp = M.compose(torch.log(torch.tensor(q0))[None], counts, torch.tensor([.75, .75], dtype=torch.float64),
                   torch.zeros(2, dtype=torch.float64))
    assert np.allclose(lp.exp()[0].numpy(), R.predict(t, hist, K, 'ad', .75), atol=1e-12)


def test_fit_counts_leave_out_their_own_transition_and_eval_counts_ignore_the_future():
    rng = np.random.default_rng(2)
    fit = rng.integers(0, 5, 300)
    K = 3
    loo = M.fit_stream_counts(fit, K)
    tables = M.fit_tables(fit, K)
    for p in (10, 100, 298):
        for k in range(1, K + 1):
            code = M.context_codes(fit, k)[p]
            full = tables[k - 1][code]
            assert loo[k - 1, p].sum() == full.sum() - 1 and loo[k - 1, p, fit[p + 1]] == full[fit[p + 1]] - 1
    dev = rng.integers(0, 5, 120)
    a = M.eval_stream_counts(fit, dev, K)
    changed = dev.copy()
    changed[61:] = (changed[61:] + 1) % 5  # alter target 61 and everything after it
    b = M.eval_stream_counts(fit, changed, K)
    assert np.array_equal(a[:, :61], b[:, :61])


def test_base_logit_gradient_is_escape_responsibility_times_cross_entropy_gradient():
    rng = np.random.default_rng(3)
    c = _counts_from(rng, 1, 1)
    z = torch.randn(1, M.A, dtype=torch.float64, requires_grad=True)
    D, th = torch.tensor([.6], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64)
    y = int(torch.argmax(c[0, 0]))
    lp = M.compose(F.log_softmax(z, -1), c, D, th)
    lp[0, y].backward()
    q = torch.softmax(z.detach(), -1)[0]
    n, T = c[0, 0].sum(), (c[0, 0] > 0).sum()
    e = (th + D * T) / (n + th)
    r = e * q[y] / lp.detach().exp()[0, y]
    assert torch.allclose(z.grad[0], r * (F.one_hot(torch.tensor(y), M.A).double() - q), atol=1e-12)


def test_chunked_forward_uses_the_same_stream_positions_as_one_pass():
    torch.manual_seed(0)
    model = M.CountCarryingNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2).double()
    rng = np.random.default_rng(4)
    tokens = rng.integers(0, M.A, 40)
    model.register_stream('s', M.eval_stream_counts(rng.integers(0, M.A, 200), tokens, 2))
    model.use_stream('s')
    with torch.random.fork_rng():
        torch.manual_seed(1)
        whole, _ = model.forward_chunk(torch.tensor(tokens))
    with torch.random.fork_rng():
        torch.manual_seed(1)
        state = model.new_state()
        parts = []
        for s in range(0, 40, 16):
            out, state = model.forward_chunk(torch.tensor(tokens[s:s + 16]), state)
            parts.append(out)
    assert torch.allclose(whole, torch.cat(parts), atol=1e-10)


def _tiny_checkpoint(tmp_path, model):
    import sys
    sys.path.insert(0, 'experiments')
    import e64_lm_baselines as E
    torch.manual_seed(0)
    net = E.LSTMLM(16) if model == 'lstm' else E.TfLM(16, 1, 32)
    args = dict(model=model, D=20000, size=16, layers=1, ctx=32, dropout=0.0)
    path = tmp_path / f'{model}.pt'
    torch.save({'args': args, 'state': net.state_dict()}, path)
    return path, net, args


def test_frozen_composition_reproduces_e64_base_scores_and_runs(tmp_path):
    """THEORY §386 frozen-composition plumbing: base-only bpc equals e64 score() on the same positions."""
    import math
    import sys
    sys.path.insert(0, 'experiments')
    import e64_lm_baselines as E
    import count_composition_frozen as Z
    import count_reference_scale as S
    x = S.load()
    seg = x[95_000_000:95_000_000 + 3000]
    for model in ('lstm', 'tf'):
        path, net, args = _tiny_checkpoint(tmp_path, model)
        t, lq = Z.base_logprobs(net, model, seg, 32)
        ours = float(-lq.gather(1, torch.as_tensor(seg[t].astype(np.int64))[:, None]).sum()) / len(t) / math.log(2)
        ref = E.score(net, torch.as_tensor(seg.astype(np.int64)), type('a', (), dict(model=model, batch_size=16)), 32)
        assert abs(ours - ref) < 1e-4
    out = tmp_path / 'z.json'
    sys.argv = ['z', '--checkpoint', str(path), '--K', '3', '--valid', '2000', '--test', '3000', '--fit', '20000',
                '--out', str(out)]
    Z.main()
    import json
    r = json.loads(out.read_text())['results']
    assert r['composed']['test']['bpc'] < r['base_only_test_bpc']  # random base: counts must help
    assert 0 <= r['composed']['test']['mean_responsibility'] <= 1


def test_generic_wrapper_keeps_positions_across_chunks_and_proxies_state():
    from sleeping_machines.count_composed_stream import CountComposedModel
    from sleeping_machines.selective_stream_language import SelectiveEventLanguageModel
    torch.manual_seed(0)
    model = CountComposedModel(SelectiveEventLanguageModel(width=8, modes=4, depth=2), 2).double()
    rng = np.random.default_rng(7)
    tokens = torch.tensor(rng.integers(0, M.A, 50))
    model.register_stream('s', M.eval_stream_counts(rng.integers(0, M.A, 300), tokens.numpy(), 2))
    model.use_stream('s')
    whole, s1 = model.forward_chunk(tokens)
    state, parts = model.new_state(), []
    for s in range(0, 50, 16):
        out, state = model.forward_chunk(tokens[s:s + 16], state)
        state = state.detach()
        parts.append(out)
    assert torch.allclose(whole, torch.cat(parts), atol=1e-10)
    assert state.position == 50 and state.deliveries == s1.deliveries
    assert torch.allclose(whole.exp().sum(-1), torch.ones(50, dtype=torch.float64))
