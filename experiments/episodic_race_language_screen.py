"""Small, guarded integrated per-position KV experiment; no official test.

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
from sleeping_machines.episodic_race_language import EpisodicRaceLanguageModel


def source_hashes():
    extra = ['experiments/episodic_race_language_screen.py',
             'sleeping_machines/episodic_race_language.py',
             'experiments/integrated_language_protocol.py']
    return {**hashes(), **{p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in extra}}


def activity(state, model):
    kv_scores = getattr(state, 'kv_scores', 0)
    kv_queries = getattr(state, 'kv_queries', 0)
    entries = sum(map(len, getattr(state, 'banks', [])))
    return dict(receiver_scores=state.candidate_scores, receiver_updates=state.selected_updates,
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
                    activity=activity(state, model))


def contracts(payload, depth, pool, matching, recent):
    """All gradient checks run inside the guarded job, before fitting."""
    with torch.random.fork_rng():
        torch.manual_seed(431)
        model = EpisodicRaceLanguageModel(payload, depth, pool, matching=matching, recent=recent)
        tokens = torch.tensor([1, 2, 1, 3, 1, 2, 4, 1, 2, 1, 5, 1])
        def forward(training, sequence=tokens, origin=0):
            torch.manual_seed(37); model.train(training)
            state = model.new_state(); state.position = origin
            z, state = model.forward_chunk(sequence, state)
            return z, state
        with torch.no_grad():
            z, state = forward(False); taught, teacher = forward(True)
            torch.testing.assert_close(z, taught, rtol=0, atol=0)
            shifted, _ = forward(False, origin=10_000_000)
            torch.testing.assert_close(z, shifted, rtol=5e-5, atol=5e-6)
            altered = tokens.clone(); altered[6:] = (altered[6:] + 7) % 27
            future, _ = forward(False, altered)
            torch.testing.assert_close(z[:6], future[:6], rtol=0, atol=0)
            torch.manual_seed(37); model.eval(); split = model.new_state()
            first, split = model.forward_chunk(tokens[:6], split)
            second, split = model.forward_chunk(tokens[6:], split)
            torch.testing.assert_close(z, torch.cat((first, second)), rtol=0, atol=0)
        assert all(len(bank) == len(tokens) for bank in state.banks)
        assert state.kv_values == depth * (len(tokens) - 1)
        assert teacher.kv_teacher_reads == teacher.kv_scores
        assert state.kv_scores <= state.kv_queries * (matching + recent)
        assert state.selected_updates == depth * len(tokens)
        for bank in state.banks:
            assert [entry[2] for entry in bank] == list(range(len(tokens)))
        # Temporal normalization / choice identity for a small admitted bank.
        scores = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        torch.manual_seed(19)
        winners = (torch.empty((20000, 3), dtype=torch.float64).exponential_() / scores.exp()).argmin(1)
        frequencies = torch.bincount(winners, minlength=3) / 20000
        torch.testing.assert_close(frequencies, scores.softmax(0).float(), rtol=0, atol=.015)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        _, event, _ = learn_chunk(model, opt, tokens[:6], tokens[1:7], model.new_state())
        assert all(not k.requires_grad and not v.requires_grad for bank in event.banks for k,v,_ in bank)
        changed = {name: p.grad is not None and bool(p.grad.abs().sum() > 0)
                   for name, p in model.named_parameters()}
        for prefix in ('kv_query.', 'kv_key.', 'kv_value.', 'kv_gate.'):
            assert any(v for k, v in changed.items() if k.startswith(prefix)), prefix
        other = copy.deepcopy(model); other_opt = torch.optim.Adam(other.parameters(), lr=.001)
        other_opt.load_state_dict(copy.deepcopy(opt.state_dict())); recovered = copy.deepcopy(event)
        rng = torch.get_rng_state()
        loss, _, logits = learn_chunk(model, opt, tokens[6:11], tokens[7:12], event)
        torch.set_rng_state(rng)
        recovered_loss, _, recovered_logits = learn_chunk(other, other_opt, tokens[6:11], tokens[7:12], recovered)
        assert loss == recovered_loss
        torch.testing.assert_close(logits, recovered_logits, rtol=0, atol=0)
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        left, right = copy.deepcopy(model), copy.deepcopy(model)
        torch.set_rng_state(rng)
        _, _, a = learn_chunk(left, torch.optim.Adam(left.parameters()), tokens[:6], tokens[1:7], left.new_state())
        torch.set_rng_state(rng)
        _, _, b = learn_chunk(right, torch.optim.Adam(right.parameters()), tokens[:6], (tokens[1:7]+1)%27, right.new_state())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        return dict(causal=True, chunk_equal=True, teacher_equals_inference=True,
            large_origin=True, retained_all_entries=True, winner_only_inference_values=True,
            candidate_budget=True, temporal_softmax_frequencies=True,
            query_key_value_gate_gradients=True, exact_next_update_recovery=True,
            prediction_before_update=True, detached_historical_cache=True)


def architecture_forward(trace, delta):
    # rates=exp(score), E/rate, and .001+.010*T/(1+T) are explicit
    # numerical clock emulation. Physical competition replaces this path.
    # Score formation, content transport and all credit arithmetic stay charged.
    emulated_clock_flops = delta['scored_clock_rates'] + 4 * delta['race_count']
    return dict(arithmetic_flops=trace['arithmetic_flops'] - emulated_clock_flops,
        special_function_evaluations=trace['special_function_evaluations'] - delta['scored_clock_rates'],
        removed_numeric_clock_flops=emulated_clock_flops,
        physical_rate_settings=delta['scored_clock_rates'], physical_races=delta['race_count'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--memory', choices=('receiver', 'kv'), default='kv')
    p.add_argument('--fit', type=int, default=2048); p.add_argument('--dev', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=4); p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--payload', type=int, default=32); p.add_argument('--depth', type=int, default=6)
    p.add_argument('--pool', type=int, default=2); p.add_argument('--matching', type=int, default=8)
    p.add_argument('--recent', type=int, default=4); p.add_argument('--seed', type=int, default=6)
    p.add_argument('--lr', type=float, default=.001); p.add_argument('--contracts-only', action='store_true')
    a = p.parse_args()
    directory = ROOT / 'experiments/results/episodic_language'; directory.mkdir(exist_ok=True)
    out = directory / f'{a.tag}.json'; running = out.with_suffix('.running.json')
    if Path(a.tag).name != a.tag or out.exists() or running.exists():
        raise ValueError('Unique unused tag required; preserve existing results')
    if min(a.fit, a.dev, a.epochs, a.chunk) < 1 or not 2*a.chunk+1 <= a.fit <= 8192 or not 2 <= a.dev <= 8192:
        raise ValueError('Small disjoint development screen only')
    torch.set_num_threads(1); torch.manual_seed(a.seed); started = time.perf_counter()
    checks = contracts(a.payload, a.depth, a.pool, a.matching, a.recent)
    result = dict(status='running', args=vars(a), source_sha256=source_hashes(), numerical_contracts=checks)
    if a.contracts_only:
        result.update(status='completed', wall_s=time.perf_counter()-started)
        out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True); return
    train = torch.tensor(text_slice(0, a.fit)); dev = torch.tensor(text_slice(90_000_000, a.dev))
    model = (EpisodicRaceLanguageModel(a.payload, a.depth, a.pool, matching=a.matching, recent=a.recent)
             if a.memory == 'kv' else SparseRaceLanguageModel(a.payload, a.depth, a.pool))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); initial = copy.deepcopy(model.state_dict())
    result.update(parameters=sum(p.numel() for p in model.parameters()), initial_dev=evaluate(model, dev, a.chunk),
        curve=[], protocol=dict(fitting=[0,a.fit], development=[90_000_000,90_000_000+a.dev],
            official_test_read=False, test=None, credit_truncation=a.chunk, cold_context=True,
            tokenizer='27-character text8', selection='minimum frozen full-development bpc over fixed passes',
            index='union of last matching-character positions and recent global positions',
            retained_cache='all per-position entries until stream reset; no eviction or receiver compression of KV',
            stale_cache='historical keys/values are cached activations, detached across credit chunks',
            forward='sparse receiver races plus one historical winner per depth when history exists',
            learning='admitted counterfactual values; conserved local surrogate; realized value gradients',
            scope='Exploratory matched small-data intervention; not unrestricted semantic retrieval or physical-energy measurement'),
        fitting_data_sha256=hashlib.sha256(train.numpy().astype('uint8').tobytes()).hexdigest(),
        development_data_sha256=hashlib.sha256(dev.numpy().astype('uint8').tobytes()).hexdigest(),
        hardware=dict(device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__))
    best, best_state = float('inf'), None; traces = {}
    def persist(state=None):
        result.update(wall_s=time.perf_counter()-started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target = out if result['status']=='completed' else running
        temporary = target.with_suffix('.json.tmp'); temporary.write_text(json.dumps(result,indent=2)+'\n'); temporary.replace(target)
        checkpoint = out.with_suffix('.progress.pt'); temporary = checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), result=result,
            best_state=best_state, stream_state=state, torch_rng=torch.get_rng_state()), temporary); temporary.replace(checkpoint)
    persist(); print(json.dumps(dict(started=a.tag, initial_dev=result['initial_dev'], checks=checks)), flush=True)
    full, remainder = divmod(a.fit-1, a.chunk)
    for epoch in range(1,a.epochs+1):
        model.train(); state=model.new_state(); total=0.
        for step,start in enumerate(range(0,a.fit-1,a.chunk),1):
            end=min(start+a.chunk,a.fit-1); size=end-start
            opt.zero_grad(set_to_none=True); before=activity(state,model); box={}
            def forward():
                z, box['state']=model.forward_chunk(train[start:end],state)
                box['loss']=F.cross_entropy(z,train[start+1:end+1])
            name = ('first' if step==1 else 'partial' if size<a.chunk else
                    'warm' if step==max(2,full//2) else None)
            if epoch==1 and name:
                trace={'forward_and_loss':capture(forward)}
                state=box['state']; after=activity(state,model)
                delta={k:after[k]-before[k] for k in after}
                trace['backward']=capture(lambda:box['loss'].backward())
                trace['gradient_clipping']=capture(lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                trace['optimizer']=capture(opt.step)
                trace['architecture_forward_and_loss']=architecture_forward(trace['forward_and_loss'],delta)
                trace['activity']=delta; traces[name]=trace
            else:
                forward(); state=box['state']
                if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite fitting loss')
                box['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step()
            total+=float(box['loss'].detach())*size; state=state.detach()
            if step%64==0:
                result['progress']=dict(epoch=epoch,targets=end,online_bpc=total/end/math.log(2))
                persist(state); print(json.dumps(result['progress']),flush=True)
        score=evaluate(model,dev,a.chunk)
        result['curve'].append(dict(epoch=epoch,dev=score,online_bpc=total/(a.fit-1)/math.log(2),
            activity=activity(state,model),optimizer_steps=step))
        if score['bpc']<best:best=score['bpc'];best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
        persist();print(json.dumps(result['curve'][-1]),flush=True)
    model.load_state_dict(best_state); result['final']=dict(dev=evaluate(model,dev,a.chunk))
    repetitions={'first':a.epochs,'warm':max(0,full-1)*a.epochs}
    if remainder:repetitions['partial']=a.epochs
    stages=('forward_and_loss','backward','gradient_clipping','optimizer')
    cpu={s:sum(repetitions[k]*trace[s]['arithmetic_flops'] for k,trace in traces.items()) for s in stages}
    cpu_special=sum(repetitions[k]*sum(trace[s]['special_function_evaluations'] for s in stages) for k,trace in traces.items())
    arch=dict(cpu);arch['forward_and_loss']=sum(repetitions[k]*trace['architecture_forward_and_loss']['arithmetic_flops'] for k,trace in traces.items())
    arch_special=cpu_special-sum(repetitions[k]*trace['activity']['scored_clock_rates'] for k,trace in traces.items())
    actual={k:sum(row['activity'][k] for row in result['curve']) for k in
            ('receiver_scores','receiver_updates','receiver_teacher_values','kv_queries','kv_scores','kv_delivered_values','kv_teacher_values','race_count','scored_clock_rates')}
    # A mature inference trace, with all historical entries retained but only
    # the declared shortlist scored and winning values read.
    model.eval(); state=model.new_state()
    with torch.no_grad():
        warm=min(256,a.dev-1-a.chunk); _,state=model.forward_chunk(dev[:warm],state)
        before=activity(state,model)
        def inference():
            z,_=model.forward_chunk(dev[warm:warm+a.chunk],state)
            F.cross_entropy(z,dev[warm+1:warm+a.chunk+1],reduction='sum')
        scoring=capture(inference);after=activity(state,model)
    delta={k:after[k]-before[k] for k in after}; projected=architecture_forward(scoring,delta)
    result['work']=dict(fitting_targets=(a.fit-1)*a.epochs,optimizer_steps=math.ceil((a.fit-1)/a.chunk)*a.epochs,
        cpu_emulator=dict(training_stages=cpu,total_training_arithmetic_flops=sum(cpu.values()),
            training_special_function_evaluations=cpu_special,total_training_unit_special_flops=sum(cpu.values())+cpu_special,
            inference_arithmetic_flops_per_character=scoring['arithmetic_flops']/a.chunk,
            inference_special_functions_per_character=scoring['special_function_evaluations']/a.chunk),
        projected_event_architecture=dict(training_stages=arch,total_training_arithmetic_flops=sum(arch.values()),
            training_special_function_evaluations=arch_special,total_training_unit_special_flops=sum(arch.values())+arch_special,
            inference_arithmetic_flops_per_character=projected['arithmetic_flops']/a.chunk,
            inference_special_functions_per_character=projected['special_function_evaluations']/a.chunk,
            physical_clock_rate_settings=actual['scored_clock_rates'],physical_races=actual['race_count'],
            assumptions='Physical competition replaces explicit exp/rate-noise division/bounded clock simulation; content, scores, gradient teacher, clipping and actual Adam remain charged. Clock circuits, RNG, index/address work and traffic are separate costs, not zero energy.'),
        actual_fitting_activity=actual,traces=traces,repetitions=repetitions,inference_trace=scoring,inference_activity=delta,
        scope='Representative first/mature/partial actual training traces, 2 FLOPs/MAC. Specials separate plus unit-weight total. Excludes dev passes and loading; not whole-run instruction or joule measurement.',
        avoided_dense_kv_aggregation=dict(
            admitted_candidates=actual['kv_scores'],queries=actual['kv_queries'],
            explicit_softmax_arithmetic_flops=3*actual['kv_scores']-actual['kv_queries'],
            explicit_softmax_exponentials=actual['kv_scores'],
            value_weighted_sum_arithmetic_flops=(2*actual['kv_scores']-actual['kv_queries'])*a.payload,
            inference_selected_value_elements=actual['kv_delivered_values']*a.payload,
            inference_dense_value_elements=actual['kv_scores']*a.payload,
            training_counterfactual_value_elements=actual['kv_teacher_values']*a.payload,
            scope='Analytical attention-module comparison on identical candidate counts, not a trained dense baseline. Value read savings apply to inference; training reads admitted counterfactual values. Storage and key reads remain.'))
    result['parameter_change_norms_by_category']={}
    for name,param in model.named_parameters():
        category='units' if name.startswith('units.') else name.split('.')[0]
        result['parameter_change_norms_by_category'][category]=result['parameter_change_norms_by_category'].get(category,0.)+float((param.detach()-initial[name]).norm())
    result['status']='completed';persist();running.unlink(missing_ok=True)
    print(json.dumps(dict(completed=a.tag,dev=result['final']['dev'],architecture_fitting_flops=sum(arch.values()),cpu_fitting_flops=sum(cpu.values()))),flush=True)


if __name__=='__main__':main()
