"""A learned periodic event state and a hard race of class clocks.

Each observed symbol applies phi <- reflection*phi + phase[symbol] modulo P.
The scalar is a phase of a unit-circle payload, not an encoded arithmetic
answer. Symbol offsets and class clock phases start randomly and learn from
the teacher's timing error. The current local update is CPU-only.
"""
import math
import numpy as np
import torch
from torch import nn


class PhaseMemory(nn.Module):
    def __init__(self, symbols, outcomes, period, seed=6, reflection=-1, sigma=2.21, cool=20000.):
        super().__init__()
        if symbols < 1 or outcomes < 2 or period <= 0 or reflection not in (-1, 1):
            raise ValueError("Invalid phase state configuration")
        self.period, self.reflection = float(period), reflection
        self.sigma, self.cool = float(sigma), float(cool)
        rng = np.random.default_rng(seed)
        self.register_buffer("phase", torch.from_numpy(rng.uniform(0, period, symbols)))
        self.register_buffer("centers", torch.from_numpy(rng.uniform(0, period, outcomes)))
        self.register_buffer("offset", torch.tensor(float(rng.uniform(-period, 0)), dtype=torch.float64))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def forward(self, symbols, ids, size):
        """Query scores are minus waiting time; maximum is the earliest clock.

        Grouped input events are in their observed order. This reduction is
        the exact affine phase composition; it processes E symbols, not E²
        pairs or a dense time grid. One phase message is carried per event.
        """
        lengths = torch.bincount(ids, minlength=size)
        if (lengths < 1).any():
            raise ValueError("Phase queries must be nonempty")
        starts = lengths.cumsum(0)-lengths
        positions = torch.arange(len(symbols), device=symbols.device)-starts[ids]
        if self.reflection == -1:
            signs = 1-2*((lengths[ids]-1-positions)%2)
        else:
            signs = torch.ones_like(positions)
        accumulated = self.phase.new_zeros(size).index_add(0, ids, self.phase[symbols]*signs)
        phi = (accumulated+self.offset).remainder(self.period)
        waits = (self.centers[None]-phi[:, None]).remainder(self.period)
        return -waits, phi

    def serial(self, symbols, rng=None):
        """Reference event execution; optional timing exploration during teaching."""
        phases = self.phase.numpy()
        phi = 0.
        noise = self.sigma*math.exp(-int(self.updates)/self.cool) if rng is not None else 0.
        for j, symbol in enumerate(symbols):
            perturbation = noise*rng.standard_normal() if rng is not None and j > 0 else 0.
            phi = (self.reflection*phi+float(phases[symbol])+perturbation)%self.period
        phi = (phi+float(self.offset))%self.period
        waits = (self.centers.numpy()-phi)%self.period
        return int(waits.argmin()), phi

    @torch.no_grad()
    def teach(self, symbols, label, rng, eta=.3, margin=.5):
        """Mistake-only causal phase credit. No modular target formula is used.

        With unit-magnitude phase Jacobians, each occurrence gets its signed
        share of the circular timing error; repeated symbols accumulate credit.
        This is a local timing surrogate, not categorical cross-entropy SGD.
        """
        if self.phase.device.type != "cpu":
            raise ValueError("Local phase teaching currently requires CPU")
        prediction, _ = self.serial(symbols)
        if prediction == label:
            return False
        _, phi = self.serial(symbols, rng)
        centers, phases = self.centers.numpy(), self.phase.numpy()
        gap = (centers[label]-phi-margin+self.period/2)%self.period-self.period/2
        step = eta*np.sign(gap)/(len(symbols)+3)
        signs = np.power(self.reflection, np.arange(len(symbols)-1, -1, -1))
        np.add.at(phases, symbols, step*signs)
        self.offset.add_(float(step))
        centers[label] -= step
        self.updates.add_(1)
        return True
