import torch

from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel
from sleeping_machines.native_stream_language import NativeStreamLanguageModel


def _pair():
    torch.manual_seed(0)
    plain = NativeStreamLanguageModel(payload=4, depth=2, pool=2, heads=2).double()
    torch.manual_seed(0)
    mem = ContextAddressedNativeModel(payload=4, depth=2, pool=2, heads=2, order=2, buckets=64).double()
    mem.load_state_dict({**mem.state_dict(), **plain.state_dict()})
    return plain, mem


def _run(model, tokens, chunks):
    with torch.random.fork_rng():
        torch.manual_seed(1)
        state, out, s = model.new_state(), [], 0
        for size in chunks:
            z, state = model.forward_chunk(tokens[s:s + size], state); out.append(z); s += size
            state = state.detach()
    return torch.cat(out)


def test_zero_read_map_nests_the_native_core_exactly():
    plain, mem = _pair()
    tokens = torch.randint(0, 27, (30,), generator=torch.Generator().manual_seed(2))
    assert torch.equal(_run(plain, tokens, (30,)), _run(mem, tokens, (30,)))


def test_memory_is_causal_chunk_invariant_and_learns():
    _, mem = _pair()
    with torch.no_grad():
        mem.memory_read_map.weight.normal_(0, .3)
    g = torch.Generator().manual_seed(3)
    tokens = torch.randint(0, 5, (40,), generator=g)  # small alphabet: contexts recur
    whole = _run(mem, tokens, (40,))
    assert torch.allclose(whole, _run(mem, tokens, (16, 16, 8)), atol=1e-10)
    changed = tokens.clone(); changed[30:] = (changed[30:] + 1) % 5
    assert torch.allclose(whole[:30], _run(mem, changed, (40,))[:30], atol=1e-12)  # future never leaks
    mem.train()
    with torch.random.fork_rng():
        torch.manual_seed(1)
        z, state = mem.forward_chunk(tokens[:-1])
    torch.nn.functional.cross_entropy(z, tokens[1:]).backward()
    assert mem.memory_read_map.weight.grad.abs().sum() > 0
    assert mem.memory_write_map.weight.grad.abs().sum() > 0
    assert state.memory_writes == 38 and len(state.slots) > 1


def test_slot_holds_mean_of_what_followed_the_context():
    _, mem = _pair()
    with torch.no_grad():
        mem.memory_write_map.weight.copy_(torch.eye(8))
    tokens = torch.tensor([1, 2, 3, 1, 2, 4])
    with torch.random.fork_rng():
        torch.manual_seed(1)
        _, state = mem.forward_chunk(tokens)
    a = mem.address([1, 2])  # followed by 3, then by 4
    assert state.counts[a] == 2
