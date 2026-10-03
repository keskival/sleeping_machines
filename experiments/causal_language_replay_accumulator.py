"""Chronological RNG and target-weighted updates for actual-write language credit."""
import torch
from parallel_head_gradient_accumulation import GradientAccumulator
from causal_language_replay_helpers_rng import batched_objective


class ReplayAccumulator(GradientAccumulator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.shadow_lanes=0;self.shadow_events=0;self.credit_chunks=0

    def accumulate(self,inputs,targets,state):
        self.model.train()
        if self.pending_targets==0:self.optimizer.zero_grad(set_to_none=True)
        # Shadow lanes fork this entering stream state; only the factual end advances the stream.
        entering_rng=torch.get_rng_state().clone()
        objective,logits,state,activity=batched_objective(self.model,inputs,targets,state,entering_rng)
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite actual-write replay objective')
        objective.backward();torch.set_rng_state(activity['factual_end_rng'])
        self.pending_targets+=len(targets);self.total_targets+=len(targets);self.credit_chunks+=1
        self.shadow_lanes+=activity['shadow_lanes'];self.shadow_events+=activity['shadow_events']
        return activity['factual_loss_sum'],state.detach(),logits.detach().clone()

    def counters(self):
        return {name:getattr(self,name) for name in ('pending_targets','total_targets','updates',
                                                   'shadow_lanes','shadow_events','credit_chunks')}

    def restore_counters(self,values):
        if values.keys()!=self.counters().keys():raise ValueError('Exact replay accumulator counters required')
        for name,value in values.items():
            if not isinstance(value,int) or value<0:raise ValueError('Nonnegative integral counters required')
            setattr(self,name,value)
