"""Same chronological target-weighted replay using cached factual prefixes."""
import torch
from causal_language_replay_accumulator import ReplayAccumulator
from causal_language_replay_cached_prefix import batched_objective


class CachedPrefixAccumulator(ReplayAccumulator):
    def accumulate(self,inputs,targets,state):
        self.model.train()
        if self.pending_targets==0:self.optimizer.zero_grad(set_to_none=True)
        entering_rng=torch.get_rng_state().clone()
        objective,logits,state,activity=batched_objective(self.model,inputs,targets,state,entering_rng)
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite cached-prefix replay objective')
        objective.backward();torch.set_rng_state(activity['factual_end_rng'])
        self.pending_targets+=len(targets);self.total_targets+=len(targets);self.credit_chunks+=1
        self.shadow_lanes+=activity['shadow_lanes'];self.shadow_events+=activity['shadow_events']
        return activity['factual_loss_sum'],state.detach(),logits.detach().clone()
