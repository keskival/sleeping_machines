"""Matched chronological teacher/factorized accumulators with replay-compatible counters."""
import torch
from torch.nn import functional as F
from causal_language_replay_accumulator import ReplayAccumulator
from causal_language_replay_helpers_rng import row
from sleeping_machines.causal_language_shadow_rng import batched_chunks
from parallel_head_gradient_accumulation import GradientAccumulator

class TeacherAccumulator(ReplayAccumulator):
    def accumulate(self,inputs,targets,state):
        result=GradientAccumulator.accumulate(self,inputs,targets,state)
        self.credit_chunks+=1
        return result

class FactorizedAccumulator(ReplayAccumulator):
    def accumulate(self,inputs,targets,state):
        self.model.train()
        if self.pending_targets==0:self.optimizer.zero_grad(set_to_none=True)
        seed=torch.get_rng_state().clone()
        logits,states,end_rng=batched_chunks(self.model,[row(self.model,inputs,state)],seed,[state])
        loss=F.cross_entropy(logits[0],targets,reduction='sum')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite factorized objective')
        loss.backward();torch.set_rng_state(end_rng)
        self.pending_targets+=len(targets);self.total_targets+=len(targets);self.credit_chunks+=1
        return float(loss.detach()),states[0].detach(),logits[0].detach().clone()
