"""Exact event-driven leaky threshold trains for piecewise constant currents.

Multiple spikes per interval, reset to zero after each. No silent-time steps.
Autograd gives exact derivatives conditional on the realized spike count;
creation/deletion at interval endpoints needs separate boundary credit.
"""
import torch


def threshold_spike_train(knots, currents, threshold, leak, initial, max_spikes=8192):
    """Solve dv/dt=I-leak*v, emit at v=threshold and reset v=0.

    Currents [segments], knots [segments+1], other inputs scalar tensors.
    Return realized spike times and terminal voltage. Positive leak/threshold,
    nonnegative bounded currents and finite horizon prevent Zeno behavior.
    This reference uses actual spike counts, not a dense candidate train.
    """
    if knots.ndim != 1 or currents.ndim != 1 or len(knots) != len(currents)+1:
        raise ValueError('One current per strictly ordered interval required')
    for tensor in (knots,threshold,leak,initial):
        if tensor.dtype != currents.dtype or tensor.device != currents.device:
            raise ValueError('Matching floating dtype/device required')
    if not currents.is_floating_point() or any(not bool(torch.isfinite(t).all())
            for t in (knots,currents,threshold,leak,initial)):
        raise ValueError('Finite floating inputs required')
    if any(t.ndim for t in (threshold,leak,initial)) or not bool(threshold>0) or not bool(leak>0):
        raise ValueError('Positive scalar threshold and leak required')
    if bool((knots[1:]<=knots[:-1]).any()) or bool((currents<0).any()) or not bool(0<=initial<threshold):
        raise ValueError('Strictly ordered intervals, nonnegative currents and subthreshold initial state required')
    voltage = initial
    trains = []
    total = 0
    for start,end,current in zip(knots[:-1],knots[1:],currents):
        equilibrium = current/leak
        gap = end-start
        crosses = bool(current>leak*threshold)
        first = torch.log((current-leak*voltage)/(current-leak*threshold))/leak if crosses else None
        if crosses and bool(first<=gap):
            period = -torch.log1p(-leak*threshold/current)/leak
            count = 1+int(torch.floor(((gap-first)/period).detach()))
            total += count
            if total>max_spikes:raise ValueError('Realized train exceeds declared event budget')
            times = start+first+torch.arange(count,dtype=currents.dtype,device=currents.device)*period
            trains.append(times)
            voltage = equilibrium*(-torch.expm1(-leak*(end-times[-1])))
        else:
            voltage = voltage*torch.exp(-leak*gap)+equilibrium*(-torch.expm1(-leak*gap))
    zero = (knots.sum()+currents.sum()+threshold+leak+initial)*0
    return torch.cat(trains) if trains else zero.expand(0),voltage
