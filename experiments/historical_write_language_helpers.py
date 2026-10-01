"""Guarded parallel-head sparse temporal KV development ladder; no official test.

Architectural floating arithmetic and CPU-emulator work are separate ledgers.
Use unique one-job queues with experiments/queue/run_safe.sh.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_shared_tasks import text_slice
from integrated_language_protocol import learn_chunk
from sparse_language_contracts import hashes
from race_language_screen import capture
from sleeping_machines.sparse_race_language import SparseRaceLanguageModel
from sleeping_machines.historical_write_race_language import HistoricalWriteRaceLanguageModel as EpisodicRaceLanguageModel


def source_hashes():
    extra = ['experiments/historical_write_language_helpers.py',
             'sleeping_machines/historical_write_race_language.py',
             'sleeping_machines/historical_write_credit.py',
             'experiments/parallel_head_race_language_screen.py',
             'sleeping_machines/parallel_head_race_language.py',
             'sleeping_machines/packed_episodic_race_language.py',
             'sleeping_machines/indexed_episodic_race_language.py',
             'experiments/integrated_language_protocol.py']
    return {**hashes(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in extra}}


def activity(state, model):
    kv_scores = getattr(state, 'kv_scores', 0)
    kv_queries = getattr(state, 'kv_queries', 0)
    entries = sum(map(len, getattr(state, 'banks', [])))
    return dict(historical_key_queries=state.historical_key_queries,
        historical_key_candidates=state.historical_key_candidates,
        historical_value_deliveries=state.historical_value_deliveries,
        historical_feature_reads=state.historical_feature_reads,
        historical_feature_writes=state.historical_feature_writes,
        eligibility_raw_bytes=entries*model.payload*model.embedding.weight.element_size(),
        kv_head_groups=getattr(state,'kv_head_groups',0),
        kv_distinct_positions=getattr(state,'kv_distinct_positions',0), index_random_draws=getattr(state, 'index_random_draws', 0), receiver_scores=state.candidate_scores, receiver_updates=state.selected_updates,
        receiver_teacher_values=state.counterfactual_values, kv_queries=kv_queries,
        kv_scores=kv_scores, kv_delivered_values=getattr(state, 'kv_values', 0),
        kv_teacher_values=getattr(state, 'kv_teacher_reads', 0), kv_stored_entries=entries,
        kv_raw_key_value_bytes=entries * 2 * model.payload * 4,
        kv_winner_age_max=getattr(state, 'kv_winner_age_max', 0),
        kv_winner_age_sum=getattr(state, 'kv_winner_age_sum', 0),
        race_count=state.selected_updates + kv_queries,
        scored_clock_rates=state.candidate_scores + kv_scores)


@torch.no_grad()
def evaluate(model, tokens, chunk):
    with torch.random.fork_rng():
        torch.manual_seed(314159); model.eval(); state = model.new_state(); total = 0.
        for start in range(0, len(tokens) - 1, chunk):
            end = min(start + chunk, len(tokens) - 1)
            z, state = model.forward_chunk(tokens[start:end], state)
            total += float(F.cross_entropy(z, tokens[start+1:end+1], reduction='sum'))
        return dict(n=len(tokens) - 1, bpc=total / (len(tokens) - 1) / math.log(2),
                    activity=activity(state, model),
                    packed_storage=state.packed_storage())



def contracts(payload, depth, pool, matching, recent, heads, historical_credit):
    """Guarded timing, separate-channel, teacher and checkpoint contracts."""
    import io
    with torch.random.fork_rng():
        torch.manual_seed(431)
        model = EpisodicRaceLanguageModel(payload, depth, pool, matching=matching,
                                          recent=recent, heads=heads, block_size=7, historical_credit=historical_credit)
        for projections in (model.kv_query,model.kv_key,model.kv_value):
            assert len({p.weight.data_ptr() for p in projections})==heads*depth
        tokens = torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3])
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
        assert state.kv_values==heads*depth*(len(tokens)-1)
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
        return dict(causal=True,chunk_equal=True,teacher_equals_inference=True,
            large_origin=True,separate_head_channels=True,parallel_arrival_bound=True,
            waiting_state_semigroup=True,retained_all_entries=True,
            winner_only_per_head_inference=True,candidate_budget=True,
            temporal_softmax_frequencies=True,full_history_bucket_eligible=True,
            all_head_query_key_value_gate_gradients=True,cross_head_projection_gradients=True,
            waiting_dynamics_gradients=True,exact_next_update_recovery=True,
            independent_head_qkv_projections=True,
            prediction_before_update=True,detached_historical_cache=True,
            cache_storage='lossless packed K/V/eligibility slabs with current-segment graphs',
            old_feature_graph_free=True, historical_write_credit=historical_credit)


def architecture_forward(trace, delta):
    # rates=exp(score), E/rate, and .001+.010*T/(1+T) are explicit
    # numerical clock emulation. Physical competition replaces this path.
    # Score formation, content transport and all credit arithmetic stay charged.
    emulated_clock_flops = delta['scored_clock_rates'] + 4 * delta['race_count']
    return dict(arithmetic_flops=trace['arithmetic_flops'] - emulated_clock_flops,
        special_function_evaluations=trace['special_function_evaluations'] - delta['scored_clock_rates'],
        removed_numeric_clock_flops=emulated_clock_flops,
        physical_rate_settings=delta['scored_clock_rates'], physical_races=delta['race_count'])

