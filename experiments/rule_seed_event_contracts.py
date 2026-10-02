"""Full-configuration gradient/update/recovery contracts, under host guard."""
import copy
import io

import torch
from torch.nn import functional as F

from native_event_tasks import episodes
from sleeping_machines.rule_seed_event_heads import RuleSeedEventHeads as SplitEventHeads


def predict(model, row, mode='observed', clear=False):
    state=model.new_state();outputs=[];targets=[];arrivals=[]
    for index,event in enumerate(row):
        if clear and event.target is not None:
            state.contexts.pop(event.source,None)
            state.memories={k:v for k,v in state.memories.items() if k[2]!=event.source}
            state.arrivals={k:v for k,v in state.arrivals.items() if k[2]!=event.source}
        timestamp=event.time if mode=='observed' else index/model.sources
        logits,arrival=model.consume_event(event.source,timestamp,event.mark,state)
        if event.target is not None:
            outputs.append(logits);targets.append(event.target);arrivals.append(arrival)
    return torch.stack(outputs),torch.tensor(targets),state,torch.stack(arrivals)


def contracts(sources,payload,depth,pool,heads,shared_maps=False,protected_pairs=0,common_source_seed=None):
    with torch.random.fork_rng():
        torch.manual_seed(931)
        model=SplitEventHeads(sources=sources,payload=payload,depth=depth,pool=pool,heads=heads,
                              shared_maps=shared_maps,protected_pairs=protected_pairs,common_source_seed=common_source_seed)
        row=episodes('order',sources,sources,123)[0]
        rng=torch.get_rng_state();model.eval()
        with torch.no_grad():z,y,state,_=predict(model,row)
        torch.set_rng_state(rng);model.train()
        taught,_,live,_=predict(model,row)
        torch.testing.assert_close(z,taught,rtol=0,atol=0)
        loss=F.cross_entropy(taught,y,reduction='sum');loss.backward()
        assert torch.isfinite(loss)
        for name in ('content.weight','source_gate.weight','transport_rate','transport_frequency'):
            grad=dict(model.named_parameters())[name].grad
            assert grad is not None and bool(grad.abs().sum()>0),name
        for d in range(depth):
            for h in range(heads):
                assert bool(model.queries[d][h].weight.grad.abs().sum()>0)
        assert live.counterfactual_values==live.candidate_scores==live.events*depth*heads*pool
        assert state.selected_updates==state.events*depth*heads
        assert state.storage()['occupied_sources']==sources
        live.detach()
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        for p in model.parameters():
            if p.grad is not None:p.grad.div_(len(y))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        saved=io.BytesIO();torch.save(dict(weights=model.state_dict(),optimizer=opt.state_dict(),
            rng=torch.get_rng_state(),stream=live),saved);saved.seek(0)
        snapshot=torch.load(saved,weights_only=False)
        restored=copy.deepcopy(model);restored.load_state_dict(snapshot['weights'])
        other=torch.optim.Adam(restored.parameters(),lr=.003);other.load_state_dict(snapshot['optimizer'])
        future=episodes('order',sources,sources,124)[0]
        for m,o in ((model,opt),(restored,other)):
            torch.set_rng_state(snapshot['rng']);o.zero_grad(set_to_none=True)
            p,t,_,_=predict(m,future);F.cross_entropy(p,t).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
        for p,q in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        # The diagnostic credit control must preserve the realized forward.
        control=copy.deepcopy(model);control.credit='pathwise'
        torch.set_rng_state(rng);a,_,_,_=predict(model,row)
        torch.set_rng_state(rng);b,_,_,_=predict(control,row)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        return dict(full_configuration_gradients=True,all_head_query_gradients=True,
            teacher_equals_inference=True,independent_observed_addresses=True,
            all_addressed_occupancy=True,exact_optimizer_checkpoint_recovery=True,
            pathwise_control_forward_equal=True,no_kv_attention=True,
            selected_updates_per_event=depth*heads,key_scores_per_event=depth*heads*pool,
            shared_rules_private_states=shared_maps,protected_pairs=protected_pairs)
