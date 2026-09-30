"""Shared causal updates and guarded checks for new integrated experiments."""
import copy
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from sparse_language_contracts import hashes
from sleeping_machines.sparse_race_language import SparseRaceLanguageModel

ROOT = Path(__file__).resolve().parents[1]


def sources(driver):
    paths = [Path(__file__), Path(driver)]
    return {**hashes(), **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths}}


def learn_chunk(model, optimizer, inputs, targets, state):
    """All causal predictions precede this chunk's parameter update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, state = model.forward_chunk(inputs, state)
    predictions = logits.detach().clone()
    loss = F.cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite predict-before-update loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()) * len(inputs), state.detach(), predictions


def numerical_contracts(payload, depth, pool):
    """No data fitting: causality, teacher equality and exact update recovery."""
    with torch.random.fork_rng():
        torch.manual_seed(431)
        model = SparseRaceLanguageModel(payload, depth, pool)
        tokens = torch.tensor([1, 2, 1, 3, 1, 2, 4, 1])
        def forward(training, sequence=tokens, origin=0):
            torch.manual_seed(37)
            model.train(training)
            state = model.new_state(); state.position = origin
            logits, state = model.forward_chunk(sequence, state)
            return logits, state
        with torch.no_grad():
            inference, state = forward(False)
            teacher, trained = forward(True)
            torch.testing.assert_close(inference, teacher, rtol=0, atol=0)
            shifted, _ = forward(False, origin=10_000_000)
            torch.testing.assert_close(inference, shifted, rtol=5e-5, atol=5e-6)
            altered = tokens.clone(); altered[4:] = (altered[4:] + 7) % 27
            future, _ = forward(False, altered)
            torch.testing.assert_close(inference[:4], future[:4], rtol=0, atol=0)
        assert state.selected_updates == depth * len(tokens)
        assert trained.counterfactual_values == depth * pool * len(tokens)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        _, state, _ = learn_chunk(model, opt, tokens[:4], tokens[1:5], model.new_state())
        # A saved model, optimizer, detached event state and RNG must recover
        # the very next prediction and parameter update, not only a cold fit.
        recovered = copy.deepcopy(model)
        other = torch.optim.Adam(recovered.parameters(), lr=.001)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        event_state = copy.deepcopy(state)
        rng = torch.get_rng_state()
        loss, _, logits = learn_chunk(model, opt, tokens[4:7], tokens[5:8], state)
        torch.set_rng_state(rng)
        resumed_loss, _, resumed_logits = learn_chunk(recovered, other, tokens[4:7],
                                                    tokens[5:8], event_state)
        assert loss == resumed_loss
        torch.testing.assert_close(logits, resumed_logits, rtol=0, atol=0)
        for p, q in zip(model.parameters(), recovered.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        # The current target may change the update, never its own prediction.
        left, right = copy.deepcopy(recovered), copy.deepcopy(recovered)
        first = torch.optim.Adam(left.parameters(), lr=.001)
        second = torch.optim.Adam(right.parameters(), lr=.001)
        torch.set_rng_state(rng)
        _, _, a = learn_chunk(left, first, tokens[:4], tokens[1:5], left.new_state())
        torch.set_rng_state(rng)
        _, _, b = learn_chunk(right, second, tokens[:4], (tokens[1:5] + 1) % 27, right.new_state())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert any(not torch.equal(p, q) for p, q in zip(left.parameters(), right.parameters()))
        return dict(payload=payload, depth=depth, pool=pool, causal=True,
                    teacher_equals_inference=True, large_origin=True,
                    current_target_cannot_change_prediction=True,
                    exact_next_update_recovery=True)
