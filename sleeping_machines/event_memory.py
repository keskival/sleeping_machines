"""Segmented affine event memory: reference and linear-work implementations."""
import torch


def segmented_memory(x, times, counts, keys, taus, sequential=False):
    """Exact exponential numerator/mass recurrence, reset at each receiver.

    Stable key sort retains deterministic (time, band) order at tied times.
    Online execution must apply that same tie policy. A packet depends only
    on arrivals already processed in its receiver, including itself.
    """
    order = torch.argsort(keys, stable=True)
    k, t, c, h = keys[order], times[order], counts[order], x[order]
    first = torch.cat((torch.ones(1, dtype=torch.bool, device=k.device), k[1:] != k[:-1]))
    dt = torch.diff(t, prepend=t[:1]).clamp_min(0)
    decay = torch.exp(-dt[:, None] / taus[None, :]) * (~first[:, None])
    values = torch.cat((h, h.new_ones((len(h), 1))), -1) * c[:, None]
    z = values[:, None, :].expand(-1, len(taus), -1)
    work = 0
    if sequential:
        state = z.new_zeros(z.shape[1:])
        rows = []
        for j in range(len(z)):
            state = decay[j, :, None] * state + z[j]
            rows.append(state)
        z = torch.stack(rows)
        work = len(z)
    else:
        # A segmented affine monoid: a zero at a receiver boundary prevents
        # any contribution from a different receiver at every scan scale.
        a = decay
        step = 1
        while step < len(z):
            z = torch.cat((z[:step], z[step:] + a[step:, :, None] * z[:-step]), 0)
            a = torch.cat((a[:step], a[step:] * a[:-step]), 0)
            work += len(z) - step
            step *= 2
    mass = z[:, :, -1]
    memory = z[:, :, :-1] / (mass[:, :, None] + 1e-4)
    inverse = torch.argsort(order)
    return memory[inverse], mass[inverse], work



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


def linear_memory(x, times, counts, keys, taus, sequential=False):
    if sequential:
        return segmented_memory(x, times, counts, keys, taus, sequential=True)
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
