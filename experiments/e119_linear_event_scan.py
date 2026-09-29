"""Linear-work affine scan over actual arrivals; no empty-time updates.

Pair reduction and prefix reconstruction implement the same segmented
recurrence as E117. PyTorch differentiates the O(E) operation graph, including
decays, time constants and arrival times. Sorting remains O(E log E).
"""
import torch


def affine_prefix(a, z):
    """Inclusive z[i] + a[i]*previous, starting from zero, in <2E combines."""
    n = len(z)
    if n <= 1:
        return z, 0
    pairs = n // 2
    paired_a = a[1:2*pairs:2] * a[:2*pairs:2]
    paired_z = z[1:2*pairs:2] + a[1:2*pairs:2, :, None] * z[:2*pairs:2]
    odd, work = affine_prefix(paired_a, paired_z)
    evens = (n-1)//2
    rest_even = z[2::2] + a[2::2, :, None] * odd[:evens]
    even = torch.cat((z[:1], rest_even), 0)
    interleaved = torch.stack((even[:pairs], odd), 1).flatten(0, 1)
    if n % 2:
        interleaved = torch.cat((interleaved, even[-1:]), 0)
    return interleaved, work + pairs + evens


def segmented_memory(x, times, counts, keys, taus, sequential=False):
    if sequential:
        from e117_serial_event_shd import segmented_memory as reference
        return reference(x, times, counts, keys, taus, sequential=True)
    if not len(x):
        raise ValueError("Expected at least one event")
    order = torch.argsort(keys, stable=True)
    k, t, c, h = keys[order], times[order], counts[order], x[order]
    first = torch.cat((torch.ones(1, dtype=torch.bool, device=k.device), k[1:] != k[:-1]))
    dt = torch.diff(t, prepend=t[:1]).clamp_min(0)
    decay = torch.exp(-dt[:, None]/taus[None, :]) * (~first[:, None])
    values = torch.cat((h, h.new_ones((len(h), 1))), -1) * c[:, None]
    z, work = affine_prefix(decay, values[:, None, :].expand(-1, len(taus), -1))
    mass = z[:, :, -1]
    memory = z[:, :, :-1] / (mass[:, :, None]+1e-4)
    inverse = torch.argsort(order)
    return memory[inverse], mass[inverse], work
