"""Stable signed temporal modes for addressed winning event updates.

This primitive is not yet part of the common SHD core. It supplies the explicit
affine temporal operator and paired route states/values/clocks from THEORY §219.
Only the winner updates memory and emits. Counterfactual candidates are exposed
when requested; exact discrete route teaching needs continuation utility.
"""
import torch
from torch import nn
from torch.nn import functional as F


def mode_flow(state, elapsed, rates, frequencies):
    """Real two-coordinate representation of diagonal complex exponential flow."""
    if state.shape[-1] != 2 or state.shape[-2:] != (*rates.shape[-1:], 2):
        raise ValueError("Expected [..., modes, 2] state and matching mode rates")
    angle = frequencies*elapsed[..., None]
    damping = torch.exp(-rates*elapsed[..., None])
    a, b = state.unbind(-1)
    cos, sin = angle.cos(), angle.sin()
    return damping[..., None]*torch.stack((cos*a-sin*b, sin*a+cos*b), -1)


class WinningTemporalModes(nn.Module):
    """One receiver update; source ids are categorical message origins.

    The caller owns addressed state and the last arrival timestamp. This avoids
    a dense node/time grid and permits independent receivers. Equal initialized
    route programs can contain an existing linear SSM; distinct programs extend
    its temporal computation. This reference primitive is sequential, not a
    high-throughput scan kernel or a trained classifier.
    """
    def __init__(self, sources, modes, output_dim, options=3):
        super().__init__()
        if min(sources, modes, output_dim, options) < 1:
            raise ValueError("Positive dimensions required")
        self.sources, self.modes, self.output_dim, self.options = sources, modes, output_dim, options
        self.raw_rate = nn.Parameter(torch.zeros(options, modes))
        self.frequency = nn.Parameter(torch.zeros(options, modes))
        self.input = nn.Parameter(torch.randn(options, sources, modes, 2)*.01)
        self.output = nn.Parameter(torch.randn(options, output_dim, 2*modes)*.01)
        self.direct = nn.Parameter(torch.zeros(options, sources, output_dim))
        self.route = nn.Parameter(torch.zeros(sources, options))

    def rates(self):
        return F.softplus(self.raw_rate)+1e-6

    def event(self, source, time, previous_time, state, count=1., counterfactuals=False):
        if source.ndim != 0 or state.shape != (self.modes, 2):
            raise ValueError("One scalar source and one receiver state per event")
        elapsed = time-previous_time
        if float(elapsed.detach()) < 0:
            raise ValueError("Receiver events must be in causal arrival order")
        delays = .001+.01*torch.sigmoid(-self.route[source])
        winner = delays.argmin()
        choices = torch.arange(self.options, device=state.device) if counterfactuals else winner[None]
        rates = self.rates()[choices]
        frequencies = self.frequency[choices]
        candidate_states = mode_flow(state.expand(len(choices), -1, -1),
            elapsed.expand(len(choices)), rates, frequencies)+count*self.input[choices, source]
        candidate_values = torch.einsum("kdm,km->kd", self.output[choices],
            candidate_states.flatten(1))+count*self.direct[choices, source]
        selected = winner if counterfactuals else 0
        new_state, value = candidate_states[selected], candidate_values[selected]
        emission_time = time+delays[winner]
        details = {"winner": winner, "delays": delays,
            "state_updates_evaluated": len(choices), "emitted_vectors": 1}
        if counterfactuals:
            details.update(candidate_states=candidate_states, candidate_values=candidate_values,
                candidate_times=time+delays, candidate_routes=choices)
        return new_state, value, emission_time, details
