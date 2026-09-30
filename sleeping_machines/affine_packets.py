"""Affine state and weighted state-pooling summaries in linear event work.

The modal transition may depend on the incoming message. Composition requires
its coefficients to be available before the scan, not to depend on recurrent
state. Complex diagonal modes represent real decay/rotation coordinate pairs.
"""
import torch


def compose(first,second):
    """Apply first, then second, including a resettable pooled-state query."""
    a,b,r,d,e,n=first
    aa,bb,rr,dd,ee,nn=second
    return (aa*a,aa*b+bb,rr*a+ee[:,None]*r,
        rr*b+ee[:,None]*d+dd,ee*e,ee*n+nn)


def prefix(descriptors):
    """Inclusive associative prefix with fewer than 2N descriptor combines."""
    count=len(descriptors[0])
    if count<=1:return descriptors,0
    pairs=count//2
    paired=compose(tuple(x[:2*pairs:2] for x in descriptors),
        tuple(x[1:2*pairs:2] for x in descriptors))
    odd,work=prefix(paired)
    rest=compose(tuple(x[:(count-1)//2] for x in odd),
        tuple(x[2::2] for x in descriptors))
    even=tuple(torch.cat((x[:1],y),0) for x,y in zip(descriptors,rest))
    result=[]
    for x,y in zip(even,odd):
        z=torch.stack((x[:pairs],y),1).flatten(0,1)
        if count%2:z=torch.cat((z,x[-1:]),0)
        result.append(z)
    return tuple(result),work+pairs+(count-1)//2


def pooled_state_scan(transitions,drives,weights,packet_first,receiver_first):
    """Return state, packet state-sum, mass and descriptor work at every event.

    Events must already be ordered within each receiver and consecutive packet.
    First-receiver events reset the carried state. First-packet events reset
    the accumulated state query/mass, while keeping the receiver state.
    Only the packet's last row needs a dense output map/gate.
    """
    count=len(transitions)
    if count<1 or transitions.shape!=drives.shape or transitions.ndim!=2:
        raise ValueError("Nonempty matching event/mode transitions and drives required")
    if any(x.shape!=(count,) for x in (weights,packet_first,receiver_first)):
        raise ValueError("One weight/reset marker per event required")
    if torch.any(weights<=0) or torch.any(receiver_first & ~packet_first):
        raise ValueError("Positive weights and a packet reset at each new receiver required")
    a=transitions*(~receiver_first[:,None])
    r=weights[:,None]*a
    d=weights[:,None]*drives
    e=(~packet_first).to(weights.dtype)
    result,work=prefix((a,drives,r,d,e,weights))
    return result[1],result[3],result[5],work
