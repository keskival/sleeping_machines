"""Protocol helpers for the prepared repeated-arrival full-architecture trial."""
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from parallel_head_race_language_screen import activity as original_activity, source_hashes as original_sources
from integrated_language_protocol import learn_chunk
from sleeping_machines.repeated_arrival_race_language import RepeatedArrivalRaceLanguageModel as EpisodicRaceLanguageModel

ROOT=Path(__file__).resolve().parents[1]


def source_hashes():
    names=['experiments/repeated_arrival_language_helpers.py',
           'experiments/repeated_arrival_gradient_contracts.py',
           'experiments/parallel_head_gradient_accumulation.py',
           'sleeping_machines/repeated_arrival_race_language.py',
           'sleeping_machines/repeated_temporal_race.py']
    return {**original_sources(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def activity(state,model):
    result=original_activity(state,model)
    result.update(kv_emitter_renewals=state.kv_emitter_renewals,
        kv_minimum_comparisons=state.kv_minimum_comparisons,
        race_count=state.selected_updates+state.kv_values)
    return result


@torch.no_grad()
def evaluate(model,tokens,chunk):
    with torch.random.fork_rng():
        torch.manual_seed(314159);model.eval();state=model.new_state();total=0.
        for start in range(0,len(tokens)-1,chunk):
            end=min(start+chunk,len(tokens)-1)
            z,state=model.forward_chunk(tokens[start:end],state)
            total+=float(F.cross_entropy(z,tokens[start+1:end+1],reduction='sum'))
        return dict(n=len(tokens)-1,bpc=total/(len(tokens)-1)/math.log(2),
            activity=activity(state,model),packed_storage=state.packed_storage())


def contracts(payload, depth, pool, matching, recent, heads, arrivals):
    """Guarded timing, separate-channel, teacher and checkpoint contracts."""
    import io
    from sleeping_machines.repeated_temporal_race import poisson_clocks, temporal_arrivals
    with torch.random.fork_rng():
        torch.manual_seed(83)
        log_rates=torch.tensor([.25,.5,1.25],dtype=torch.float64).log()
        marks=[]; gaps=[]
        for _ in range(1024):
            _,times,winners=poisson_clocks(log_rates,16)
            marks.append(winners); gaps.append(torch.diff(times,prepend=times.new_zeros(1)))
        marks=torch.cat(marks); gaps=torch.cat(gaps)
        probabilities=log_rates.softmax(0)
        torch.testing.assert_close(torch.bincount(marks,minlength=3).to(probabilities)/len(marks),probabilities,rtol=0,atol=.013)
        joint=torch.bincount(marks[:-1]*3+marks[1:],minlength=9).reshape(3,3).to(probabilities)/(len(marks)-1)
        torch.testing.assert_close(joint,probabilities[:,None]*probabilities[None,:],rtol=0,atol=.015)
        assert abs(float(gaps.mean())-.5)<.02 and abs(float(gaps.var())-.25)<.03
        scores=torch.tensor([-.5,.2,1.],dtype=torch.float64,requires_grad=True)
        content=torch.tensor([[1.,2.],[-1.,.2],[.1,-2.]],dtype=torch.float64,requires_grad=True)
        errors=torch.tensor([[1.,-.3],[.4,.8],[-.1,2.],[.7,-.6]],dtype=torch.float64)
        torch.manual_seed(67);rates,times,winners=poisson_clocks(scores.detach(),4)
        torch.manual_seed(67);messages,_,_=temporal_arrivals(scores,content,4)
        credit,=torch.autograd.grad((messages*errors).sum(),scores)
        expected=torch.zeros_like(scores); previous=0.
        for time,winner,error in zip(times,winners,errors):
            part=rates*(time-previous)*((content.detach()-content.detach().mean(0))@error)
            part[winner]-=part.sum();expected+=part;previous=time
        torch.testing.assert_close(credit,expected,rtol=1e-12,atol=1e-12)
        assert abs(float(credit.sum()))<1e-12
    with torch.random.fork_rng():
        torch.manual_seed(431)
        model = EpisodicRaceLanguageModel(payload, depth, pool, matching=matching,
                                          recent=recent, heads=heads, block_size=7, arrivals_per_query=arrivals)
        for projections in (model.kv_query,model.kv_key,model.kv_value):
            assert len({p.weight.data_ptr() for p in projections})==heads*depth
        tokens = torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3])
        if arrivals==1:
            from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel
            original=ParallelHeadRaceLanguageModel(payload,depth,pool,matching=matching,
                recent=recent,heads=heads,block_size=7)
            original.load_state_dict(copy.deepcopy(model.state_dict()))
            torch.manual_seed(97); first,_=original.forward_chunk(tokens[:8]);first.square().sum().backward()
            rng=torch.get_rng_state()
            torch.manual_seed(97); second,_=model.forward_chunk(tokens[:8]);second.square().sum().backward()
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            assert torch.equal(rng,torch.get_rng_state())
            for p,q in zip(original.parameters(),model.parameters()):
                assert (p.grad is None)==(q.grad is None)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            model.zero_grad(set_to_none=True)
        def forward(training, sequence=tokens, origin=0):
            torch.manual_seed(37); model.train(training)
            state=model.new_state(); state.position=origin
            z,state=model.forward_chunk(sequence,state)
            return z,state
        with torch.no_grad():
            z,state=forward(False); taught,teacher=forward(True)
            torch.testing.assert_close(z,taught,rtol=0,atol=0)
            shifted,_=forward(False,origin=10_000_000)
            torch.testing.assert_close(z,shifted,rtol=5e-5,atol=5e-6)
            altered=tokens.clone(); altered[8:]=(altered[8:]+7)%27
            future,_=forward(False,altered)
            torch.testing.assert_close(z[:8],future[:8],rtol=0,atol=0)
            torch.manual_seed(37); model.eval(); split=model.new_state()
            first,split=model.forward_chunk(tokens[:8],split)
            second,split=model.forward_chunk(tokens[8:],split)
            torch.testing.assert_close(z,torch.cat((first,second)),rtol=0,atol=0)
            value=torch.linspace(-1,1,payload)
            a=torch.tensor(.007,dtype=torch.float64); b=torch.tensor(.021,dtype=torch.float64)
            once=model.transport(value,a+b,0,0)
            twice=model.transport(model.transport(value,a,0,0),b,0,0)
            torch.testing.assert_close(once,twice,rtol=2e-6,atol=2e-7)
        assert state.context.shape==(heads*payload,)
        assert state.context_arrivals.shape==(heads,)
        assert float(state.context_arrivals.max())<len(tokens)-1+.5
        assert float(state.context_arrivals.min())>=len(tokens)-1
        assert len(state.banks)==heads*depth
        assert all(len(bank)==len(tokens) for bank in state.banks)
        assert state.selected_updates==heads*depth*len(tokens)
        assert teacher.counterfactual_values==heads*depth*pool*len(tokens)
        assert state.kv_values==arrivals*heads*depth*(len(tokens)-1)
        assert state.kv_emitter_renewals==(arrivals-1)*state.kv_queries
        assert teacher.kv_teacher_reads==teacher.kv_scores
        assert state.kv_scores<=state.kv_queries*(matching+recent)
        assert state.kv_head_groups==depth*(len(tokens)-1)
        assert state.packed_storage()['differentiable_entries']==0
        for bank in state.banks:
            assert [entry[2] for entry in bank]==list(range(len(tokens)))
        # Old entries in an eligible bucket can still enter the shortlist.
        probe=model.new_state()
        for i in range(256):
            probe.banks[0].append((torch.zeros(payload),torch.zeros(payload),i))
        probe.semantic_buckets[0].setdefault(7,[]).extend(range(256))
        torch.manual_seed(37)
        assert any(i<200 for i in model.candidates(probe,0,1,torch.zeros(payload)))
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        _,event,_=learn_chunk(model,opt,tokens[:8],tokens[1:9],model.new_state())
        assert event.packed_storage()['differentiable_entries']==0
        assert not event.context_arrivals.requires_grad
        changed={name:p.grad is not None and bool(p.grad.abs().sum()>0)
                 for name,p in model.named_parameters()}
        for channel in range(heads*depth):
            for prefix in ('kv_query.','kv_key.','kv_value.','kv_gate.'):
                assert any(v for k,v in changed.items() if k.startswith(prefix+str(channel)+'.')), (prefix,channel)
        for layer in range(depth):
            for head in range(heads):
                assert changed[f'queries.{layer}.{head}.weight']
        if heads>1:
            for mix in model.channel_mix:
                assert bool(mix.weight.grad[:payload,payload:].abs().sum()>0)
        assert changed['transport_rate'] and changed['transport_frequency']
        saved=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                        state=event,rng=torch.get_rng_state()),saved)
        saved.seek(0); checkpoint=torch.load(saved,weights_only=False)
        restored=copy.deepcopy(model); restored.load_state_dict(checkpoint['model'])
        other=torch.optim.Adam(restored.parameters(),lr=.001)
        other.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        loss,_,predictions=learn_chunk(model,opt,tokens[8:15],tokens[9:16],event)
        torch.set_rng_state(checkpoint['rng'])
        recovered_loss,_,recovered_predictions=learn_chunk(restored,other,tokens[8:15],tokens[9:16],checkpoint['state'])
        assert loss==recovered_loss
        torch.testing.assert_close(predictions,recovered_predictions,rtol=0,atol=0)
        for p,q in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        # The current target changes only the subsequent update.
        left,right=copy.deepcopy(model),copy.deepcopy(model)
        rng=torch.get_rng_state()
        torch.set_rng_state(rng)
        _,_,a=learn_chunk(left,torch.optim.Adam(left.parameters()),tokens[:8],tokens[1:9],left.new_state())
        torch.set_rng_state(rng)
        _,_,b=learn_chunk(right,torch.optim.Adam(right.parameters()),tokens[:8],(tokens[1:9]+1)%27,right.new_state())
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        scores=torch.tensor([-1.,0.,1.],dtype=torch.float64)
        torch.manual_seed(19)
        winners=(torch.empty((20000,3),dtype=torch.float64).exponential_()/scores.exp()).argmin(1)
        torch.testing.assert_close(torch.bincount(winners,minlength=3)/20000,
                                   scores.softmax(0).float(),rtol=0,atol=.015)
        return dict(original_single_arrival_nesting=(True if arrivals==1 else 'not applicable'),
            poisson_mark_frequencies=True,poisson_gap_moments=True,
            poisson_lag_joint_frequencies=True,aggregated_teacher_equals_explicit=True,
            conserved_local_content_credit=True,causal=True,chunk_equal=True,teacher_equals_inference=True,
            large_origin=True,separate_head_channels=True,parallel_arrival_bound=True,
            waiting_state_semigroup=True,retained_all_entries=True,
            declared_arrivals_per_head_inference=True,candidate_budget=True,
            temporal_softmax_frequencies=True,full_history_bucket_eligible=True,
            all_head_query_key_value_gate_gradients=True,cross_head_projection_gradients=True,
            waiting_dynamics_gradients=True,exact_next_update_recovery=True,
            independent_head_qkv_projections=True,
            prediction_before_update=True,detached_historical_cache=True,
            cache_storage='lossless packed slabs with current-segment graphs')


def architecture_forward(trace,delta):
    # One rate setting and initial clock per scored key. A winning emitter alone
    # draws/divides a new gap and adds it to its absolute clock (two FLOPs).
    # Every delivered clock has the inherited four-operation bounded encoding.
    removed=delta['scored_clock_rates']+2*delta['kv_emitter_renewals']+4*delta['race_count']
    if trace['arithmetic_flops'] < removed or trace['special_function_evaluations'] < delta['scored_clock_rates']:
        raise ValueError('Clock projection exceeds captured work')
    if not trace['formula_coverage_complete']:
        raise ValueError('Complete floating-operator coverage required')
    return dict(arithmetic_flops=trace['arithmetic_flops']-removed,
        special_function_evaluations=trace['special_function_evaluations']-delta['scored_clock_rates'],
        removed_numeric_clock_flops=removed,physical_rate_settings=delta['scored_clock_rates'],
        physical_arrivals=delta['race_count'],winner_local_renewals=delta['kv_emitter_renewals'],
        emulator_minimum_comparisons=delta['kv_minimum_comparisons'])
