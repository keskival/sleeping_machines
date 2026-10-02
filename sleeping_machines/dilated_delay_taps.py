"""Learned dilated delay taps for the native temporal core (THEORY §391).

WaveNet-style dilation as literal event delay: at layer d, the channel mix also reads that layer's own
input from tau_d events earlier,

    mixed_d(t) = W_d x_d(t) + U_d x_d(t - tau_d),     tau_d = softplus(r_d), initialised 2**d,

read by linear interpolation between buffered events so tau_d receives a gradient.  Depth L reaches
2**L - 1 events back along a path of at most L taps, instead of one depth-L traversal per intervening event,
which shortens credit paths for long dependencies.  U_d starts at zero, so the tapped model equals the
unchanged native core exactly (nesting contract).  Races, addressed units, persistent memories,
counterfactual route credit and the per-event context path are retained.  Taps add one d_model x d_model
product per layer per event and a bounded per-layer buffer of recent layer inputs.  Credit through a tap
reaches only what the current credit chunk retains; buffered entries are detached at chunk boundaries.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .native_stream_language import NativeLanguageState, NativeStreamLanguageModel


class TappedNativeState(NativeLanguageState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.taps = {}

    def detach(self):
        super().detach()
        self.taps = {d: [v.detach() for v in buf] for d, buf in self.taps.items()}
        return self


class TappedMix(nn.Linear):
    def __init__(self, base, depth, init_delay, capacity, owner):
        super().__init__(base.in_features, base.out_features, bias=False)
        with torch.no_grad():
            self.weight.copy_(base.weight)
        self.depth, self.capacity = depth, capacity
        self.tap = nn.Linear(base.in_features, base.out_features, bias=False)
        nn.init.zeros_(self.tap.weight)
        self.raw_delay = nn.Parameter(torch.tensor(math.log(math.expm1(float(init_delay)))))
        self._owner = [owner]  # plain list: no submodule registration cycle

    def delay(self):
        return F.softplus(self.raw_delay).clamp(1., float(self.capacity - 1))

    def forward(self, x):
        out = F.linear(x, self.weight)
        state = self._owner[0]._tap_state
        buf = state.taps.setdefault(self.depth, [])
        if buf:
            tau = self.delay()
            lo = int(tau.detach().floor())
            frac = tau - lo
            older = buf[-min(lo + 1, len(buf))]
            newer = buf[-min(lo, len(buf))]
            out = out + self.tap((1 - frac) * newer + frac * older)
        buf.append(x)
        if len(buf) > self.capacity:
            del buf[0]
        return out


class TappedNativeStreamLanguageModel(NativeStreamLanguageModel):
    def __init__(self, payload=16, depth=8, pool=2, heads=2, vocabulary=27, max_delay=None):
        super().__init__(payload, depth, pool, heads, vocabulary)
        delays = [2 ** d for d in range(depth)]
        capacity = int(max_delay or 2 * delays[-1]) + 2
        self.channel_mix = nn.ModuleList([TappedMix(mix, d, delays[d], capacity, self)
                                          for d, mix in enumerate(self.channel_mix)])
        self._tap_state = None

    def new_state(self):
        return TappedNativeState()

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        self._tap_state = state
        try:
            return super().forward_chunk(tokens, state)
        finally:
            self._tap_state = None

    def tap_delays(self):
        return [float(m.delay()) for m in self.channel_mix]
