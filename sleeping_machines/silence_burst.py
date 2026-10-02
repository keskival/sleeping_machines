"""Causal event-only silence-timeout accumulation, a scoped reference primitive.

This does not replace the integrated native model. Gradients are exact inside
a fixed burst partition; hard merge/split boundary credit is a separate task.
"""
import torch


def silence_bursts(times, values, timeout, horizon, decay):
    """Accumulate decaying messages; emit timeout after the last arrival.

    An input exactly at the deadline extends the burst. A deadline at the
    observation horizon fires if no input at that time replaces it. Never use
    a later input to close a burst early or flush an unfinished burst at EOF.
    Return emitted times/values and causal pending state at the horizon.
    Scalar timeout/decay and all tensors share dtype/device. Fixed-partition
    autograd includes input/value/time/timeout/decay and downstream layers.
    """
    if times.ndim!=1 or values.ndim!=2 or len(times)!=len(values):
        raise ValueError('Sorted scalar times and event-by-payload values required')
    for x in (times,values,timeout,horizon,decay):
        if x.dtype!=values.dtype or x.device!=values.device or not torch.isfinite(x).all():
            raise ValueError('Finite floating tensors with common dtype/device required')
    if not values.is_floating_point() or any(x.ndim for x in (timeout,horizon,decay)):
        raise ValueError('Floating values and scalar timing parameters required')
    if not timeout>0 or not decay>=0 or bool((times[1:]<times[:-1]).any()):
        raise ValueError('Positive timeout, nonnegative decay and sorted inputs required')
    zero=values.sum(0)*0+(times.sum()+timeout+horizon+decay)*0
    state=zero;last=None;deadline=None;output_times=[];outputs=[];accepted=0
    for timestamp,value in zip(times,values):
        if timestamp>horizon:break
        if deadline is not None and deadline<timestamp:
            output_times.append(deadline)
            outputs.append(state*torch.exp(-decay*timeout))
            state=zero;last=None;deadline=None
        if last is not None:state=state*torch.exp(-decay*(timestamp-last))
        state=state+value;last=timestamp;deadline=timestamp+timeout;accepted+=1
    if deadline is not None and deadline<=horizon:
        output_times.append(deadline);outputs.append(state*torch.exp(-decay*timeout))
        state=zero;last=None;deadline=None
    pending=state if last is None else state*torch.exp(-decay*(horizon-last))
    ts=torch.stack(output_times) if output_times else times[:0]
    ys=torch.stack(outputs) if outputs else zero.expand(0,values.shape[1])
    return ts,ys,dict(pending=pending,deadline=deadline,active=last is not None,
        accepted_events=accepted,emitted_events=len(outputs),state_tensor_elements=values.shape[1])


def finite_timeout_risk(scores, timeouts, outcome_loss):
    """Exact categorical option risk; every hard timed outcome remains paid.

    Fixed positive timeout candidates, learned score probabilities. The callback
    evaluates a complete causal outcome including its actual emission times.
    No soft averaging of event times or values enters hard inference.
    """
    if scores.ndim!=1 or timeouts.shape!=scores.shape or not len(scores):
        raise ValueError('One score per timeout candidate required')
    if not torch.isfinite(scores).all() or not torch.isfinite(timeouts).all() or not (timeouts>0).all():
        raise ValueError('Finite scores and positive timeout candidates required')
    if timeouts.requires_grad:
        raise ValueError('Fixed timeout bank: continuous boundary gradients are not claimed')
    losses=torch.stack([outcome_loss(timeout) for timeout in timeouts])
    return (scores.softmax(-1)*losses).sum(),losses
