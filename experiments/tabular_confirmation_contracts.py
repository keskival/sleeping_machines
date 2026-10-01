"""Optimizer contracts for guarded jobs only; not run beside a local trainer."""
import copy

import torch
from torch.nn import functional as F

from experiments.native_tabular_model import NativeTabularModel


def optimizer_recovery():
    with torch.random.fork_rng():
        torch.manual_seed(9017)
        model = NativeTabularModel(4,2)
        optimizer = torch.optim.Adam(model.parameters(),lr=.003)
        def update(network, opt, row, target):
            opt.zero_grad(set_to_none=True)
            logits,_=network(row)
            loss=F.cross_entropy(logits[None,:],torch.tensor([target]))
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            return loss.detach()
        update(model,optimizer,torch.tensor([.2,-.3,.7,.1]),1)
        other=copy.deepcopy(model)
        restored=torch.optim.Adam(other.parameters(),lr=.003)
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        rng=torch.get_rng_state()
        a=update(model,optimizer,torch.tensor([-.4,.2,.6,-.1]),0)
        torch.set_rng_state(rng)
        b=update(other,restored,torch.tensor([-.4,.2,.6,-.1]),0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for p,q in zip(model.parameters(),other.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        return dict(exact_next_optimizer_update_recovery=True,finite_full_native_gradients=True)
