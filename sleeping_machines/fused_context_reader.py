"""Inference-only compilation of a late-projected addressed reader.

R_v(W mean_h)+R_s seen = (R_v W)mean_h+R_s seen. Fold once with fixed
weights; raw-feature slot schema and native temporal computation are preserved.
Training the fused reader would be a different optimizer parameterization.
"""
import torch
from torch import nn
from .late_projected_context_memory import LateProjectedContextModel


class FusedContextReader(LateProjectedContextModel):
    def train(self, mode=True):
        if mode:
            raise ValueError('Compiled reader is inference-only; train the unfused model')
        return super().train(False)

    def read_value(self, state, address):
        n=state.counts.get(address,0)
        return state.slots[address]/n if n else self.embedding.weight.new_zeros(self.total_payload)

    def forward_chunk(self, tokens, state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Frozen fused reader requires eval and no_grad')
        return super().forward_chunk(tokens,state)


def compile_reader(model):
    if not isinstance(model,LateProjectedContextModel):
        raise TypeError('Compilation requires raw-feature late-projection memory')
    with torch.random.fork_rng():
        fused=FusedContextReader(payload=model.payload,depth=model.depth,pool=model.pool,heads=model.heads,
            vocabulary=model.vocabulary,order=model.order,buckets=model.buckets)
    fused.to(device=model.embedding.weight.device,dtype=model.embedding.weight.dtype)
    fused.load_state_dict(model.state_dict())
    d=model.total_payload
    with torch.no_grad():
        fused.memory_read_map.weight[:,:d].copy_(model.memory_read_map.weight[:,:d]@model.memory_write_map.weight)
    fused.memory_write_map=nn.Identity()
    fused.eval()
    return fused
