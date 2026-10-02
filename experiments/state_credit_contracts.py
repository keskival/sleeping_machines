"""Actual-task full-depth state-credit contracts; optimizer work requires guard."""
import copy
import io

import torch
from torch.nn import functional as F

from experiments.native_event_tasks import episodes
from experiments.paired_timing_tasks import paired_timing_episodes
from sleeping_machines.split_event_heads import SplitEventHeads
from sleeping_machines.state_write_credit import StateCreditEventHeads


def predict(model,row,mode='observed',clear=False):
    state=model.new_state();outputs=[];targets=[];arrivals=[]
    for index,event in enumerate(row):
        if clear and event.target is not None:
            if hasattr(state,'clear_source'):state.clear_source(event.source)
            else:
                state.contexts.pop(event.source,None)
                state.memories={k:v for k,v in state.memories.items() if k[2]!=event.source}
                state.arrivals={k:v for k,v in state.arrivals.items() if k[2]!=event.source}
        timestamp=event.time if mode=='observed' else index/model.sources
        z,arrival=model.consume_event(event.source,timestamp,event.mark,state)
        if event.target is not None:outputs.append(z);targets.append(event.target);arrivals.append(arrival)
    return torch.stack(outputs),torch.tensor(targets),state,torch.stack(arrivals)


def contracts(sources,payload,depth,pool,heads,shared_maps=False,protected_pairs=0,
              state_credit=1.,task='order',time_input='observed'):
    with torch.random.fork_rng():
        torch.manual_seed(931)
        kwargs=dict(sources=sources,payload=payload,depth=depth,pool=pool,heads=heads,
            shared_maps=shared_maps,protected_pairs=protected_pairs,classes=4 if task=='order' else 2)
        model=StateCreditEventHeads(**kwargs,state_credit=state_credit)
        zero=StateCreditEventHeads(**kwargs,state_credit=0.)
        parent=SplitEventHeads(**kwargs)
        for m in (zero,parent):m.load_state_dict(model.state_dict())
        rows=paired_timing_episodes(sources,2*sources,123) if task=='paired_timing' else episodes(task,sources,sources,123)
        row=rows[0];rng=torch.get_rng_state();parent_gradients=[]
        for m in (parent,zero):
            torch.set_rng_state(rng);m.train();z,y,_,_=predict(m,row,time_input)
            F.cross_entropy(z,y,reduction='sum').backward()
            parent_gradients.append([None if p.grad is None else p.grad.clone() for p in m.parameters()])
        for p,q in zip(*parent_gradients):
            if p is None:assert q is None
            else:torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.set_rng_state(rng);model.eval();z,y,state,_=predict(model,row,time_input);after_infer=torch.get_rng_state()
        torch.set_rng_state(rng);model.train();taught,_,live,_=predict(model,row,time_input)
        torch.testing.assert_close(z,taught,rtol=0,atol=0);assert torch.equal(after_infer,torch.get_rng_state())
        loss=F.cross_entropy(taught,y,reduction='sum');loss.backward();assert torch.isfinite(loss)
        for d in range(depth):
            for h in range(heads):
                grad=model.queries[d][h].weight.grad
                assert grad is not None and bool(grad.abs().sum()>0) and bool(torch.isfinite(grad).all())
        assert live.selected_updates==live.events*depth*heads==state.selected_updates
        assert live.candidate_scores==live.events*depth*heads*pool==state.candidate_scores
        if state_credit:
            assert len(live.credit_memories)==sources*depth*heads*pool
            assert not state.credit_memories
        live.detach();opt=torch.optim.Adam(model.parameters(),lr=.003)
        for p in model.parameters():
            if p.grad is not None:p.grad.div_(len(y))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        buffer=io.BytesIO();torch.save(dict(weights=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),buffer)
        buffer.seek(0);saved=torch.load(buffer,weights_only=False)
        restored=copy.deepcopy(model);restored.load_state_dict(saved['weights'])
        other=torch.optim.Adam(restored.parameters(),lr=.003);other.load_state_dict(saved['optimizer'])
        for m,o in ((model,opt),(restored,other)):
            torch.set_rng_state(saved['rng']);o.zero_grad(set_to_none=True)
            z,y,_,_=predict(m,row,time_input);F.cross_entropy(z,y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
        for p,q in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        return dict(actual_task_classes=kwargs['classes'],zero_added_credit_exact_parent_gradients=True,
            teacher_equals_inference=True,same_rng_consumption=True,all_head_query_gradients=True,
            exact_optimizer_checkpoint_recovery=True,winner_only_physical_commits=True,
            auxiliary_training_addresses=sources*depth*heads*pool if state_credit else 0,
            selected_updates_per_event=depth*heads,key_scores_per_event=depth*heads*pool,
            state_credit_strength=state_credit,
            scope='Full-state local surrogate; no arbitrary expected-gradient fidelity guarantee')
