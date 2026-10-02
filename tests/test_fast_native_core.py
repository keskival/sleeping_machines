import torch
from torch.nn import functional as F

from sleeping_machines.fast_native_core import FastNativeStreamLanguageModel
from sleeping_machines.native_stream_language import NativeStreamLanguageModel


def _pair(**kw):
    torch.manual_seed(0)
    ref = NativeStreamLanguageModel(**kw).double()
    fast = FastNativeStreamLanguageModel(**kw).double()
    fast.load_state_dict(ref.state_dict())
    with torch.no_grad():  # move zero-initialised controls off zero so every path is exercised
        for a, b in zip(ref.parameters(), fast.parameters()):
            noise = torch.randn_like(a) * .05; a.add_(noise); b.add_(noise)
    return ref, fast


def _train(model, tokens, chunks):
    model.train(); model.zero_grad()
    outs, s = [], 0
    with torch.random.fork_rng():
        torch.manual_seed(3)
        state = model.new_state()
        for size in chunks:
            z, state = model.forward_chunk(tokens[s:s + size], state)
            F.cross_entropy(z, tokens[s + 1:s + size + 1], reduction='sum').backward()
            outs.append(z.detach()); state = state.detach(); s += size
    return torch.cat(outs), state


def test_fast_training_path_equals_reference_outputs_gradients_and_state():
    for kw in (dict(payload=4, depth=3, pool=2, heads=2), dict(payload=6, depth=2, pool=3, heads=3)):
        ref, fast = _pair(**kw)
        tokens = torch.randint(0, 27, (41,), generator=torch.Generator().manual_seed(5))
        zr, sr = _train(ref, tokens, (16, 16, 8))
        zf, sf = _train(fast, tokens, (16, 16, 8))
        torch.testing.assert_close(zf, zr, rtol=0, atol=1e-11)
        for (name, a), b in zip(ref.named_parameters(), fast.parameters()):
            if a.grad is None:
                assert b.grad is None or not b.grad.abs().any(), name
            else:
                torch.testing.assert_close(b.grad, a.grad, rtol=1e-9, atol=1e-11, msg=name)
        assert sr.memories.keys() == sf.memories.keys() and sr.visited_units == sf.visited_units
        for k in sr.memories:
            torch.testing.assert_close(sf.memories[k], sr.memories[k], rtol=0, atol=1e-11)
            torch.testing.assert_close(sf.arrivals[k], sr.arrivals[k], rtol=0, atol=1e-12)
        assert (sr.counterfactual_values, sr.candidate_scores, sr.selected_updates, sr.events) == \
               (sf.counterfactual_values, sf.candidate_scores, sf.selected_updates, sf.events)


def test_fast_eval_path_is_the_reference_path():
    ref, fast = _pair(payload=4, depth=2, pool=2, heads=2)
    tokens = torch.randint(0, 27, (20,), generator=torch.Generator().manual_seed(6))
    ref.eval(); fast.eval()
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(9); a, _ = ref.forward_chunk(tokens)
        torch.manual_seed(9); b, _ = fast.forward_chunk(tokens)
    assert torch.equal(a, b)


def test_fast_mixin_accelerates_pooled_and_gated_compositions_identically():
    import numpy as np
    from sleeping_machines import count_carrying_language as M
    from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.statistic_race_top import TopStatisticRaceNativeModel
    rng = np.random.default_rng(4)
    tokens = torch.tensor(rng.integers(0, 6, 33))
    counts = M.eval_stream_counts(rng.integers(0, 6, 300), tokens.numpy(), 2)
    for cls, kw in ((GatedCountCarryingNativeModel, dict(escape_gate=True, count_message=True)),
                    (TopStatisticRaceNativeModel, dict(addresses=6, key_dim=4))):
        torch.manual_seed(0)
        ref = cls(payload=4, depth=2, pool=2, heads=2, orders=2, **kw).double()
        fast = fast_class(cls)(payload=4, depth=2, pool=2, heads=2, orders=2, **kw).double()
        fast.load_state_dict(ref.state_dict())
        for m in (ref, fast):
            m.register_stream('s', counts); m.use_stream('s')
        zr, _ = _train(ref, tokens, (16, 16))
        zf, _ = _train(fast, tokens, (16, 16))
        torch.testing.assert_close(zf, zr, rtol=0, atol=1e-11)
        for (name, a), b in zip(ref.named_parameters(), fast.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(b.grad, a.grad, rtol=1e-9, atol=1e-11, msg=name)


def test_class_swap_used_by_the_driver_preserves_isinstance_and_state():
    from sleeping_machines.fast_native_core import fast_class, FastNativeCoreMixin
    from sleeping_machines.statistic_race_top import TopStatisticRaceNativeModel
    from sleeping_machines.statistic_race_memory import StatisticRaceNativeModel
    m = TopStatisticRaceNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2, addresses=4, key_dim=4)
    keys = list(m.state_dict())
    m.__class__ = fast_class(type(m))
    assert isinstance(m, StatisticRaceNativeModel) and isinstance(m, FastNativeCoreMixin)
    assert list(m.state_dict()) == keys
