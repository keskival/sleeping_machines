"""Labelled diagnostic: count receivers composed over the input-gated temporal carrier (THEORY §§386-387).

Same protocol, optimizer and work tracing as parallel_event_language.py. Adds context-suffix count receivers of
orders 1..K (leave-one-out on the fitting stream, prequential on development) composed by the escape-race cascade
with learnable discount/concentration; the carrier predictive is the base measure. The carrier is a diagnostic
control, not the integrated native core; this tests the memorization-tax predictions, not the full architecture.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))
from e120_shared_tasks import text_slice
from sleeping_machines.operation_audit import OperationAudit
from sleeping_machines.parallel_stream_language import ParallelEventLanguageModel
from sleeping_machines.language_memory import PROFILES, initialize_language_memory
from sleeping_machines.selective_stream_language import SelectiveEventLanguageModel
from sleeping_machines.count_composed_stream import CountComposedModel
from sleeping_machines.count_carrying_language import fit_stream_counts, eval_stream_counts


@torch.no_grad()
def evaluate(model, tokens, chunk):
    model.eval()
    state, total, block_bpc, block_targets = model.new_state(), 0., [], []
    for start in range(0, len(tokens)-1, chunk):
        end = min(start+chunk, len(tokens)-1)
        logits, state = model.forward_chunk(tokens[start:end], state)
        loss_sum = float(F.cross_entropy(logits, tokens[start+1:end+1], reduction='sum'))
        total += loss_sum
        block_bpc.append(loss_sum/(end-start)/math.log(2)); block_targets.append(end-start)
    return dict(n=len(tokens)-1, bpc=total/(len(tokens)-1)/math.log(2),
                event_deliveries=state.deliveries, block_bpc=block_bpc, block_targets=block_targets)


def capture(action):
    with OperationAudit() as audit:
        action()
    result = audit.result()
    if not result['formula_coverage_complete']:
        raise ValueError(result['unsupported_floating_operators'])
    return result


def fitting_work(model, optimizer, tokens, args):
    """First, persistent full and partial chunks; complete backward/update work."""
    chunk, remainder = args.chunk, (args.fit-1)%args.chunk
    full = (args.fit-1)//chunk
    traces = {}
    def step(size, warm):
        probe = copy.deepcopy(model).train()
        opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        state = probe.new_state()
        if warm:
            with torch.no_grad():
                _, state = probe.forward_chunk(tokens[:chunk], state)
            state = state.detach()
        start = chunk if warm else 0
        opt.zero_grad(set_to_none=True)
        box = {}
        def forward():
            logits, _ = probe.forward_chunk(tokens[start:start+size], state)
            box['loss'] = F.cross_entropy(logits, tokens[start+1:start+size+1])
        return dict(forward_and_loss=capture(forward), backward=capture(lambda:box['loss'].backward()),
                    gradient_clipping=capture(lambda:torch.nn.utils.clip_grad_norm_(
                        probe.parameters(),1.,error_if_nonfinite=True)),optimizer=capture(opt.step))
    traces['first_full_chunk'] = step(chunk, False)
    traces['persistent_full_chunk'] = step(chunk, True)
    if remainder:
        traces['partial_chunk'] = step(remainder, True)
    repetitions = dict(first_full_chunk=args.epochs,
                       persistent_full_chunk=(full-1)*args.epochs)
    if remainder:
        repetitions['partial_chunk'] = args.epochs
    stages = {stage:sum(repetitions[k]*v[stage]['arithmetic_flops'] for k,v in traces.items())
              for stage in ('forward_and_loss','backward','gradient_clipping','optimizer')}
    special = sum(repetitions[k]*sum(row['special_function_evaluations'] for row in v.values())
                  for k,v in traces.items())
    model.eval()
    state = model.new_state()
    with torch.no_grad():
        _, state = model.forward_chunk(tokens[:chunk], state)
        def inference():
            logits, _ = model.forward_chunk(tokens[chunk:2*chunk], state)
            F.cross_entropy(logits, tokens[chunk+1:2*chunk+1], reduction='sum')
        scoring = capture(inference)
    return dict(training_stages=stages,total_training_arithmetic_flops=sum(stages.values()),
                training_special_function_evaluations=special,
                total_training_unit_special_flops=sum(stages.values())+special,
                fitting_targets=(args.fit-1)*args.epochs,
                optimizer_steps=(full+bool(remainder))*args.epochs,
                inference_arithmetic_flops_per_character=scoring['arithmetic_flops']/chunk,
                inference_special_functions_per_character=scoring['special_function_evaluations']/chunk,
                traces=traces,repetitions=repetitions,inference_trace=scoring,
                convention='2 arithmetic FLOPs/MAC; special functions separate, with an additional unit-weight total for comparison to historical neural estimates.',
                scope='Representative saved-parameter complete-step estimates, including the actual first, persistent and partial chunks, backward, clipping and warm Adam. No artificial fitting warmup. Excludes development/test passes, trace setup, loading, index/queue work and physical memory traffic; not measured energy.')


def diagnostics(model, initial, credit_horizon):
    rows = []
    for i, layer in enumerate(model.layers):
        rates = F.softplus(layer.raw_rate.detach())+1e-6
        timescales = 1/rates
        deltas = {name:float((value.detach()-initial[f'layers.{i}.{name}']).norm())
                  for name,value in layer.named_parameters()}
        rows.append(dict(layer=i+1,parameter_change_norms=deltas,
                         memory_time_min=float(timescales.min()),
                         memory_time_median=float(timescales.median()),
                         memory_time_max=float(timescales.max()),
                         fraction_memory_times_above_credit_horizon=float((timescales>credit_horizon).float().mean()),
                         frequency_abs_max=float(layer.frequency.detach().abs().max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--contracts',required=True)
    parser.add_argument('--fit',type=int,default=131072)
    parser.add_argument('--dev',type=int,default=8192)
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--chunk',type=int,default=64)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--modes',type=int,default=32)
    parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=6)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--official-test',action='store_true')
    parser.add_argument('--test',type=int,default=1000000)
    parser.add_argument('--monitor-every',type=int,default=0)
    parser.add_argument('--memory-profile',choices=PROFILES,default='inherited')
    parser.add_argument('--content-memory',action='store_true')
    parser.add_argument('--orders',type=int,default=5)
    args = parser.parse_args()
    out = ROOT/'experiments/results/count_composed_carrier'/f'{args.tag}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    checkpoint, running = out.with_suffix('.progress.pt'), out.with_suffix('.running.json')
    if Path(args.tag).name != args.tag or min(args.fit,args.dev,args.epochs,args.chunk,args.width,args.modes,args.depth) < 1:
        raise ValueError('Positive settings and a unique plain tag required')
    if args.fit < 2*args.chunk+1 or args.fit > 90_000_000 or args.dev < 2 or args.dev > 5_000_000:
        raise ValueError('Invalid fitting/development split')
    if args.official_test:
        raise ValueError('Official test needs test-stream counts; not part of this diagnostic')
    if args.official_test and (args.fit != 10_000_000 or args.dev != 200_000 or args.test != 1_000_000):
        raise ValueError('Official comparison uses the fixed 10M / 200k / 1M protocol')
    contract_path = ROOT/args.contracts
    contract = json.loads(contract_path.read_text())
    if contract['status'] != 'completed':
        raise ValueError('Completed numerical contracts required')
    if contract['args'].get('memory_profile', 'inherited') != args.memory_profile:
        raise ValueError('Numerical contract must cover this memory profile')
    if bool(contract['args'].get('content_memory',False)) != args.content_memory:
        raise ValueError('Numerical contract must cover this content-memory model')
    for name,digest in contract['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Contract source changed: '+name)
    if out.exists() or (running.exists() and not (args.resume and checkpoint.exists())):
        raise ValueError('A prior result must be preserved; unfinished runs require their checkpoint')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    started = time.perf_counter()
    train = torch.tensor(text_slice(0,args.fit)); dev = torch.tensor(text_slice(90_000_000,args.dev))
    model_class = SelectiveEventLanguageModel if args.content_memory else ParallelEventLanguageModel
    model = CountComposedModel(model_class(width=args.width,modes=args.modes,depth=args.depth),args.orders)
    initialize_language_memory(model.base,args.memory_profile)
    count_started = time.perf_counter()
    model.register_stream('fit',fit_stream_counts(train.numpy(),args.orders))
    model.register_stream('dev',eval_stream_counts(train.numpy(),dev.numpy(),args.orders))
    count_seconds = time.perf_counter()-count_started
    model.use_stream('fit')
    def evaluate_dev(tokens):
        model.use_stream('dev'); score = evaluate(model,tokens,args.chunk); model.use_stream('fit')
        D,th = model.escape_parameters(); score['escape'] = dict(discount=D.tolist(),concentration=th.tolist())
        return score
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    initial = copy.deepcopy(model.state_dict())
    sources = [Path(__file__),ROOT/'sleeping_machines/parallel_stream_language.py',
               ROOT/'sleeping_machines/stream_language.py',ROOT/'sleeping_machines/event_state.py',
               ROOT/'sleeping_machines/event_memory.py',ROOT/'sleeping_machines/operation_audit.py',
               ROOT/'experiments/e120_shared_tasks.py',ROOT/'sleeping_machines/language_memory.py',
               ROOT/'sleeping_machines/selective_stream_language.py',ROOT/'sleeping_machines/count_composed_stream.py',
               ROOT/'sleeping_machines/count_carrying_language.py']
    result = dict(status='running',args=vars(args),parameters=sum(p.numel() for p in model.parameters()),
                  curve=[],monitor_curve=[],initial_dev=evaluate_dev(dev),
                  protocol=dict(fitting=[0,args.fit],development=[90_000_000,90_000_000+args.dev],
                      test=([95_000_000,96_000_000] if args.official_test else None),
                      official_test_read=False,tokenizer='fixed 27-character text8 alphabet',
                      statistical_experts=[f'context-suffix count receivers orders 1..{args.orders}, escape-race cascade'],selection='minimum full-development bpc over the fixed epoch budget',
                      cold_context=True,excluded_first_target=True,weights_frozen_on_test=True,
                      credit_truncation=args.chunk,clock_dtype='float64',payload_dtype='float32',
                      memory_initialization=args.memory_profile,
                      content_memory=args.content_memory,
                      execution='causal layer-wise affine scans, persistent state across chunks'),
                  contract_result=args.contracts,contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  fitting_data_sha256=hashlib.sha256(train.numpy().astype('uint8').tobytes()).hexdigest(),
                  hardware=dict(platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),
                  scope='Labelled diagnostic: carrier base plus count receivers (statistical experts present by design). One seed, development only; not the integrated native architecture; no measured energy.')
    epoch,next_start,total,steps,best_loss,best_state = 1,0,0.,0,float('inf'),None
    state,previous_wall = model.new_state(),0.
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        old_args = dict(saved['result']['args']); new_args = dict(vars(args))
        old_args.pop('resume',None); new_args.pop('resume',None)
        if old_args != new_args or saved['result']['source_sha256'] != result['source_sha256']:
            raise ValueError('Resumed settings or sources changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        result = saved['result']; initial = saved['initial']
        epoch,next_start,total,steps,best_loss,best_state,state = (saved[k] for k in
            ('epoch','next_start','total','steps','best_loss','best_state','stream_state'))
        torch.set_rng_state(saved['torch_rng']); previous_wall = result['wall_s']
    def persist(save=True):
        result.update(wall_s=previous_wall+time.perf_counter()-started,
                      max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if save:
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),result=result,
                initial=initial,epoch=epoch,next_start=next_start,total=total,steps=steps,
                best_loss=best_loss,best_state=best_state,stream_state=state.detach(),torch_rng=torch.get_rng_state()),temporary)
            temporary.replace(checkpoint)
        target = out if result['status'] == 'completed' else running
        temporary = target.with_suffix('.json.tmp'); temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(target)
        if result['status'] == 'completed':running.unlink(missing_ok=True)
    persist()
    print(json.dumps(dict(started=args.tag,parameters=result['parameters'],initial_dev=result['initial_dev'])),flush=True)
    while epoch <= args.epochs:
        model.train()
        for start in range(next_start,len(train)-1,args.chunk):
            end = min(start+args.chunk,len(train)-1)
            optimizer.zero_grad(set_to_none=True)
            logits,state = model.forward_chunk(train[start:end],state)
            loss = F.cross_entropy(logits,train[start+1:end+1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fitting loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())*(end-start); steps += 1
            next_start,state = end,state.detach()
            if steps%256 == 0:
                result['progress'] = dict(epoch=epoch,targets=end,steps=steps,
                    fitting_online_bpc=total/end/math.log(2),event_deliveries=state.deliveries)
                persist(); print(json.dumps(result['progress']),flush=True)
            if args.monitor_every and end%args.monitor_every == 0:
                score = evaluate_dev(dev[:min(len(dev),8192)])
                result['monitor_curve'].append(dict(epoch=epoch,targets=end,dev=score))
                print(json.dumps(result['monitor_curve'][-1]),flush=True); model.train()
        score = evaluate_dev(dev)
        result['curve'].append(dict(epoch=epoch,fitting_online_bpc=total/(len(train)-1)/math.log(2),
            optimizer_steps=steps,training_event_deliveries=state.deliveries,dev=score))
        if score['bpc'] < best_loss:
            best_loss,best_state = score['bpc'],copy.deepcopy(model.state_dict()); result['selected_epoch'] = epoch
        print(json.dumps(result['curve'][-1]),flush=True)
        epoch,next_start,total,steps,state = epoch+1,0,0.,0,model.new_state();persist()
    model.load_state_dict(best_state)
    result['final'] = dict(dev=evaluate_dev(dev))
    result['count_receivers'] = dict(orders=args.orders,fit_transitions_counted=(args.fit-1)*args.orders,
        development_transitions_counted=(args.dev-1)*args.orders,lookups_per_target=args.orders,precompute_seconds=count_seconds,
        scope='Integer count increments/lookups reported separately from FLOPs; cascade arithmetic is inside traced forward/backward.')
    if args.official_test:
        result['final']['official_test'] = evaluate(model,torch.tensor(text_slice(95_000_000,args.test)),args.chunk)
        result['protocol']['official_test_read'] = True
    result['diagnostics'] = diagnostics(model.base,{k[5:]:v for k,v in initial.items() if k.startswith('base.')},args.chunk)
    result['work'] = fitting_work(model,optimizer,train,args)
    result['status'] = 'completed';persist()
    print(json.dumps(dict(final=result['final'],selected_epoch=result['selected_epoch'],
        total_training_arithmetic_flops=result['work']['total_training_arithmetic_flops'],wall_s=result['wall_s'])),flush=True)


if __name__ == '__main__':
    main()
