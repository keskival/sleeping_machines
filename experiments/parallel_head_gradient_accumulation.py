"""Decouple truncated temporal credit from the optimizer's update interval."""
import copy
import io

import torch
from torch.nn import functional as F

from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel


class GradientAccumulator:
    def __init__(self, model, optimizer, learning_rate, warmup_targets=0):
        self.model, self.optimizer = model, optimizer
        self.learning_rate, self.warmup_targets = learning_rate, warmup_targets
        self.pending_targets = 0
        self.total_targets = 0
        self.updates = 0

    def accumulate(self, inputs, targets, state):
        self.model.train()
        if self.pending_targets == 0:
            self.optimizer.zero_grad(set_to_none=True)
        logits, state = self.model.forward_chunk(inputs, state)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite accumulated training loss')
        predictions = logits.detach().clone()
        loss.backward()
        self.pending_targets += len(targets)
        self.total_targets += len(targets)
        # Credit still stops here, even while gradients await an Adam update.
        return float(loss.detach()), state.detach(), predictions

    def normalize(self):
        if self.pending_targets < 1:
            raise ValueError('No accumulated targets')
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.pending_targets)

    def step(self):
        multiplier = min(1., self.total_targets/self.warmup_targets) if self.warmup_targets else 1.
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate*multiplier
        self.optimizer.step()
        self.pending_targets = 0
        self.updates += 1

    def update(self):
        self.normalize()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.step()


def accumulation_contracts(payload, depth, pool, matching, recent, heads):
    """Explicit summed gradients, partial windows, causality and exact recovery."""
    with torch.random.fork_rng():
        torch.manual_seed(117)
        model = ParallelHeadRaceLanguageModel(payload,depth,pool,matching=matching,recent=recent,heads=heads)
        reference = copy.deepcopy(model)
        opt = torch.optim.Adam(model.parameters(),lr=.002)
        ropt = torch.optim.Adam(reference.parameters(),lr=.002)
        learner = GradientAccumulator(model,opt,.002)
        tokens = torch.tensor(([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3]*5)[:65])
        event, other = model.new_state(), reference.new_state()
        summed = {}
        for start,end in ((0,16),(16,32),(32,48),(48,63)):
            rng = torch.get_rng_state()
            _,event,predictions = learner.accumulate(tokens[start:end],tokens[start+1:end+1],event)
            assert event.packed_storage()['differentiable_entries']==0
            torch.set_rng_state(rng); reference.train(); ropt.zero_grad(set_to_none=True)
            z,other = reference.forward_chunk(tokens[start:end],other)
            F.cross_entropy(z,tokens[start+1:end+1],reduction='sum').backward()
            other = other.detach()
            torch.testing.assert_close(predictions,z.detach(),rtol=0,atol=0)
            for name,p in reference.named_parameters():
                if p.grad is not None:
                    if name not in summed:summed[name]=p.grad.detach().clone()
                    else:summed[name].add_(p.grad)
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
        assert learner.pending_targets==63
        # Saving a partly filled window must retain gradients, not just Adam.
        saved=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=event,
            gradients={n:p.grad for n,p in model.named_parameters() if p.grad is not None},
            pending=learner.pending_targets,total=learner.total_targets,rng=torch.get_rng_state()),saved)
        saved.seek(0); checkpoint=torch.load(saved,weights_only=False)
        restored=copy.deepcopy(model); restored.load_state_dict(checkpoint['model'])
        restored_opt=torch.optim.Adam(restored.parameters(),lr=.002)
        restored_opt.load_state_dict(checkpoint['optimizer'])
        recovered=GradientAccumulator(restored,restored_opt,.002)
        recovered.pending_targets=checkpoint['pending']; recovered.total_targets=checkpoint['total']
        for name,p in restored.named_parameters():p.grad=checkpoint['gradients'].get(name)
        # First verify correct normalization of the partial 63-target window.
        learner.normalize()
        for name,p in model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad,summed[name]/63,rtol=2e-6,atol=2e-7)
        # Recover the original unnormalized gradients before the next microchunk.
        for name,p in model.named_parameters():p.grad=checkpoint['gradients'].get(name).clone() if name in checkpoint['gradients'] else None
        torch.set_rng_state(checkpoint['rng'])
        _,event,first=learner.accumulate(tokens[63:64],tokens[64:65],event)
        torch.set_rng_state(checkpoint['rng'])
        _,recovered_event,second=recovered.accumulate(tokens[63:64],tokens[64:65],checkpoint['state'])
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        learner.update(); recovered.update()
        for p,q in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        assert learner.updates==recovered.updates==1
        assert learner.pending_targets==0
        # No target, including a later target in this window, may affect the
        # predictions made before its delayed optimizer update.
        left,right=copy.deepcopy(model),copy.deepcopy(model)
        a=GradientAccumulator(left,torch.optim.Adam(left.parameters(),lr=.002),.002)
        b=GradientAccumulator(right,torch.optim.Adam(right.parameters(),lr=.002),.002)
        rng=torch.get_rng_state(); torch.set_rng_state(rng)
        _,_,p=a.accumulate(tokens[:16],tokens[1:17],left.new_state())
        torch.set_rng_state(rng)
        _,_,q=b.accumulate(tokens[:16],(tokens[1:17]+1)%27,right.new_state())
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        return dict(accumulated_gradient_matches_sum=True,partial_window_normalization=True,
            credit_detached_each_microchunk=True,prediction_before_delayed_update=True,
            serialized_partial_accumulation_recovery=True)
