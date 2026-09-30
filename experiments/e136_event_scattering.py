"""Reversible packet/memory exchange conditioned on realized event choices."""
import torch
from sleeping_machines.event_memory import affine_prefix


def scatter(packets, angles, receivers, initial, sequential=False):
    """Inputs are in chronological arrival order; ties retain that order.

    One receiver state is updated and one packet is emitted per event. Angles
    and addresses are declared key-policy outputs, independent of value inputs
    for the conditional transport claim. Initial/final state is part of it.
    """
    if packets.ndim!=2 or initial.ndim!=2 or initial.shape[1]!=packets.shape[1]:
        raise ValueError('Invalid payload/state shapes')
    if len(packets)==0 or angles.shape!=(len(packets),) or receivers.shape!=angles.shape:
        raise ValueError('Expected a nonempty stream with one angle/address per event')
    if int(receivers.min())<0 or int(receivers.max())>=len(initial):raise ValueError('Receiver out of bounds')
    a,b=angles.cos(),angles.sin()
    if sequential:
        states=list(initial.unbind());outputs=[]
        for i in range(len(packets)):
            k=int(receivers[i]);previous=states[k]
            states[k]=a[i]*previous+b[i]*packets[i]
            outputs.append(-b[i]*previous+a[i]*packets[i])
        return torch.stack(outputs),torch.stack(states),len(packets)
    order=torch.argsort(receivers,stable=True);inverse=torch.argsort(order)
    k,x,c,s=receivers[order],packets[order],a[order],b[order]
    first=torch.cat((torch.ones(1,dtype=torch.bool,device=k.device),k[1:]!=k[:-1]))
    decay=c*(~first)
    marked=s[:,None]*x+first[:,None]*c[:,None]*initial[k]
    inclusive,work=affine_prefix(decay[:,None],marked[:,None,:]);inclusive=inclusive[:,0]
    previous=torch.cat((torch.zeros_like(inclusive[:1]),inclusive[:-1]),0)
    previous=torch.where(first[:,None],initial[k],previous)
    out=-s[:,None]*previous+c[:,None]*x
    last=torch.cat((k[1:]!=k[:-1],torch.ones(1,dtype=torch.bool,device=k.device)))
    final=initial.index_copy(0,k[last],inclusive[last])
    return out[inverse],final,work
