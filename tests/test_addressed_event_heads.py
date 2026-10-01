"""Read-only contracts: native addressed state and causal task information."""
import copy
from dataclasses import replace
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from native_event_tasks import episodes, data_hash
from sleeping_machines.addressed_event_heads import AddressedEventHeads


def test_targets_do_not_enter_marks_and_future_changes_do_not_change_outputs():
    rows = episodes('order',4,8,123)
    assert all(e.mark==(0.,1.) for row in rows for e in row if e.target is not None)
    assert all(e.target is None for row in rows for e in row if e.mark[1]==0)
    assert data_hash(rows)==data_hash(episodes('order',4,8,123))
    torch.manual_seed(821);model=AddressedEventHeads(payload=8,depth=2).eval()
    rng=torch.get_rng_state()
    changed=[replace(e,mark=(-e.mark[0],e.mark[1]),target=0 if e.target is not None else None)
             if i>=5 else e for i,e in enumerate(rows[0])]
    with torch.no_grad():
        state=model.new_state();a=[]
        for e in rows[0]:a.append(model.consume_event(e.source,e.time,e.mark,state)[0])
        torch.set_rng_state(rng);state=model.new_state();b=[]
        for e in changed:b.append(model.consume_event(e.source,e.time,e.mark,state)[0])
    torch.testing.assert_close(torch.stack(a[:5]),torch.stack(b[:5]),rtol=0,atol=0)


def test_native_token_adapter_keeps_state_across_chunks_without_a_cache():
    from sleeping_machines.native_stream_language import NativeStreamLanguageModel
    torch.manual_seed(825);model=NativeStreamLanguageModel(payload=8,depth=2).eval()
    tokens=torch.tensor([1,2,3,1,6,4]);rng=torch.get_rng_state()
    with torch.no_grad():
        whole,full=model.forward_chunk(tokens)
        torch.set_rng_state(rng)
        first,state=model.forward_chunk(tokens[:2]);second,state=model.forward_chunk(tokens[2:],state)
    torch.testing.assert_close(whole,torch.cat((first,second)),rtol=0,atol=0)
    assert state.events==full.events==len(tokens)
    assert state.storage()['occupied_sources']==1
    assert state.storage()['occupied_receivers']<=model.depth*model.heads*model.pool
    assert not hasattr(state,'cache')


def test_unrelated_addresses_do_not_mutate_dormant_state_and_have_no_global_barrier():
    torch.manual_seed(822);model=AddressedEventHeads(payload=8,depth=8).eval()
    state=model.new_state()
    with torch.no_grad():
        _,finish=model.consume_event(0,0.,(1.,0.),state)
        saved={k:v.clone() for k,v in state.memories.items()}
        old_context=tuple(t.clone() for t in state.contexts[0])
        # This source starts before source0 finishes; admission is independent.
        _,other=model.consume_event(1,.0001,(-1.,0.),state)
        assert state.queue_wait_sum==0
        assert float(other)>0 and float(finish)>0
        for k,v in saved.items():torch.testing.assert_close(state.memories[k],v,rtol=0,atol=0)
        for a,b in zip(old_context,state.contexts[0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        _,_=model.consume_event(0,.0002,(0.,1.),state)
        assert state.queue_wait_sum>0


def test_teacher_inference_equality_timestamp_origin_and_detach():
    torch.manual_seed(823);model=AddressedEventHeads(payload=8,depth=2)
    row=episodes('order',4,4,123)[0]
    rng=torch.get_rng_state()
    def run(training,origin=0.):
        torch.set_rng_state(rng);model.train(training);state=model.new_state();out=[]
        with torch.no_grad():
            for e in row:out.append(model.consume_event(e.source,e.time+origin,e.mark,state)[0])
        return torch.stack(out),state
    a,state=run(False);b,taught=run(True);shift,_=run(False,1e7)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a,shift,rtol=5e-5,atol=5e-6)
    assert state.candidate_scores==state.events*model.depth*model.heads*model.pool
    assert taught.counterfactual_values==state.candidate_scores
    assert state.selected_updates==state.events*model.depth*model.heads
    assert state.storage()['occupied_sources']==4
    clone=copy.deepcopy(taught.detach())
    assert all(not t.requires_grad for pair in clone.contexts.values() for t in pair)


def test_silence_does_not_schedule_hidden_updates():
    torch.manual_seed(824);model=AddressedEventHeads(payload=8,depth=2).eval()
    with torch.no_grad():
        state=model.new_state();model.consume_event(0,.1,(1.,0.),state)
        before=state.selected_updates
        model.consume_event(0,1e5,(0.,1.),state)
    assert state.selected_updates-before==model.depth*model.heads
    assert state.events==2
