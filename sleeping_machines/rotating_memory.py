"""Addressed signed temporal memory, with the real mean as its zero-phase case.

Conjugating source vectors by their event times permits the existing linear-work
affine scan to evaluate damped coordinate-pair rotations. Elapsed time changes
the payload, without extra emitted events or a dense event-pair matrix.
"""
import torch
from .event_memory import affine_prefix


def rotate_pairs(x, angle):
    pair = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pair.unbind(-1)
    cos, sin = angle.cos(), angle.sin()
    return torch.stack((cos*a-sin*b, sin*a+cos*b), -1).flatten(-2)


def rotating_memory(x, times, counts, keys, taus, phase, sequential=False):
    if not len(x) or x.shape[-1] % 2 or phase.shape != (len(taus), x.shape[-1]//2):
        raise ValueError("Nonempty even-dimensional messages and one phase per mode required")
    order = torch.argsort(keys, stable=True)
    k, t, c, h = keys[order], times[order], counts[order], x[order]
    first = torch.cat((torch.ones(1, dtype=torch.bool, device=k.device), k[1:] != k[:-1]))
    dt = torch.diff(t, prepend=t[:1]).clamp_min(0)
    decay = torch.exp(-dt[:, None]/taus[None, :])*(~first[:, None])
    frequency = phase/taus[:, None]
    angle = t[:, None, None]*frequency[None]
    source = rotate_pairs(h[:, None, :].expand(-1, len(taus), -1), -angle)
    values = torch.cat((source, source.new_ones((*source.shape[:-1], 1))), -1)*c[:, None, None]
    if sequential:
        state, rows = torch.zeros_like(values[0]), []
        for i in range(len(values)):
            state = decay[i, :, None]*state+values[i]
            rows.append(state)
        z, work = torch.stack(rows), len(rows)
    else:
        z, work = affine_prefix(decay, values)
    mass = z[:, :, -1]
    mean = z[:, :, :-1]/(mass[:, :, None]+1e-4)
    memory = rotate_pairs(mean, angle)
    inverse = torch.argsort(order)
    return memory[inverse], mass[inverse], work
