"""Separate predict-before-update adaptation of the integrated neural backbone.

Frozen and adaptive arms start at the same selected checkpoint. Both maintain
event memory; only the adaptive arm changes parameters, after each causal
16-character block. No replay and no official test. Run through run_safe.sh.
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
from integrated_language_protocol import numerical_contracts, sources
from sleeping_machines.sparse_race_language import SparseRaceLanguageModel
from race_language_screen import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True); parser.add_argument('--checkpoint-result', required=True)
    parser.add_argument('--offset', type=int, default=90_065_536)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--race-seed', type=int, default=571)
    a = parser.parse_args()
    if Path(a.tag).name != a.tag or a.n < 2*a.chunk+1 or a.chunk < 1 or not a.lr > 0:
        raise ValueError('Plain unique tag, positive rate and at least two update blocks required')
    if not 90_000_000 <= a.offset < a.offset+a.n <= 95_000_000:
        raise ValueError('Disjoint development stream only; official test reserved')
    out = ROOT / 'experiments/results/online_language' / f'{a.tag}.json'
    running = out.with_suffix('.running.json')
    if out.exists() or running.exists():
        raise ValueError('Use an unused tag; preserve interrupted evidence')
    path = ROOT / a.checkpoint_result; saved = json.loads(path.read_text())
    if saved['status'] != 'completed' or saved['protocol']['official_test_read']:
        raise ValueError('Completed development checkpoint required')
    if a.offset < saved['protocol']['development'][1]:
        raise ValueError('Adaptation stream must follow inherited validation')
    for p, digest in saved['source_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest:
            raise ValueError('Inherited checkpoint source changed: '+p)
    torch.set_num_threads(1); started = time.perf_counter()
    architecture = {key: saved['args'][key] for key in ('payload', 'depth', 'pool')}
    checks = numerical_contracts(**architecture)
    checkpoint = path.with_suffix('.progress.pt')
    inherited = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if inherited['result']['status'] != 'completed' or inherited['result']['source_sha256'] != saved['source_sha256']:
        raise ValueError('Checkpoint and completed result disagree')
    base = SparseRaceLanguageModel(**architecture)
    base.load_state_dict(inherited['best_state'])
    models = {name: copy.deepcopy(base) for name in ('frozen', 'online')}
    states = {name: base.new_state() for name in models}
    optimizer = torch.optim.Adam(models['online'].parameters(), lr=a.lr)
    stream = torch.tensor(text_slice(a.offset, a.n))
    totals = {name: 0. for name in models}; traces = {name: {} for name in models}
    timings = {name: 0. for name in models}; blocks = []
    starts = list(range(0, len(stream)-1, a.chunk))
    trace_indices = {0, len(starts)//2, len(starts)-1}
    result = dict(status='running', args=vars(a), architecture=architecture,
        parameters=saved['parameters'], capacity_units=base.capacity_units,
        source_sha256=sources(__file__), numerical_contracts=checks,
        inherited_result=a.checkpoint_result,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        stream_sha256=hashlib.sha256(stream.numpy().astype('uint8').tobytes()).hexdigest(),
        protocol=dict(stream=[a.offset, a.offset+a.n], scored_targets=a.n-1,
            official_test_read=False, adaptation='all integrated neural parameters',
            predict_before_update=True, feedback_delay_characters=a.chunk,
            persistent_event_memory_in_both_arms=True, cold_context=True,
            replay=False, passes=1, learning_rate='fixed before this stream; no stream-based selection',
            optimizer='fresh Adam, clipping at 1; inherited optimizer moments not reused',
            paired_race_noise='same block seed in both arms; contexts diverge after adaptation',
            inherited_fitting=saved['protocol']['fitting'], inherited_development=saved['protocol']['development']),
        hardware=dict(device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__),
        scope='One checkpoint/window/rate; neural online adaptation with block-delayed feedback. Separate from frozen official benchmarking and statistical mixing adaptation.')
    out.parent.mkdir(exist_ok=True)
    def persist(completed=False):
        result.update(wall_s=time.perf_counter()-started,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            progress=dict(scored_targets=sum(b['n'] for b in blocks),
                bpc={name:totals[name]/max(1,sum(b['n'] for b in blocks))/math.log(2) for name in models}))
        target = out if completed else running
        temp = target.with_suffix('.json.tmp'); temp.write_text(json.dumps(result,indent=2)+'\n'); temp.replace(target)
        temp = out.with_suffix('.progress.pt.tmp')
        torch.save(dict(models={name:m.state_dict() for name,m in models.items()},
            states={name:s.detach() for name,s in states.items()}, optimizer=optimizer.state_dict(),
            result=result, blocks=blocks, next_start=sum(b['n'] for b in blocks)),temp)
        temp.replace(out.with_suffix('.progress.pt'))
    persist(); print(json.dumps(dict(started=a.tag, contracts=checks)),flush=True)
    for i,start in enumerate(starts):
        end = min(start+a.chunk,len(stream)-1); scores = {}
        for name,model in models.items():
            begin=time.perf_counter(); torch.manual_seed(a.race_seed+start)
            model.train(name=='online'); box = {}
            def forward():
                with torch.set_grad_enabled(name=='online'):
                    z,state=model.forward_chunk(stream[start:end],states[name])
                    box.update(state=state,loss=F.cross_entropy(z,stream[start+1:end+1]))
            if i in trace_indices:
                trace={'forward_and_loss':capture(forward)}
            else:
                forward(); trace=None
            if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite online stream loss')
            # Record prediction loss BEFORE backward, clipping or any update.
            loss=float(box['loss'].detach())*(end-start)
            totals[name]+=loss; scores[name]=loss/(end-start)/math.log(2)
            if name=='online':
                optimizer.zero_grad(set_to_none=True)
                clip=lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if trace is not None:
                    trace.update(backward=capture(box['loss'].backward),
                        gradient_clipping=capture(clip),optimizer=capture(optimizer.step))
                else:
                    box['loss'].backward(); clip(); optimizer.step()
            states[name]=box['state'].detach()
            if trace is not None:traces[name][str(i)]=dict(n=end-start,stages=trace)
            timings[name]+=time.perf_counter()-begin
        blocks.append(dict(start=start,end=end,n=end-start,bpc=scores))
        if (i+1)%64==0:
            persist();print(json.dumps(result['progress']),flush=True)
    work={}
    for name in models:
        rows=traces[name]
        # Cold first block, a representative persistent block, and the actual
        # last/partial block. Traces observe real updates, not extra warmup fits.
        counts={str(0):1,str(len(starts)//2):len(starts)-2,str(len(starts)-1):1}
        arithmetic=sum(counts[k]*sum(t['arithmetic_flops'] for t in row['stages'].values()) for k,row in rows.items())
        specials=sum(counts[k]*sum(t['special_function_evaluations'] for t in row['stages'].values()) for k,row in rows.items())
        state=states[name]
        work[name]=dict(arithmetic_flops=arithmetic,special_functions=special,
            unit_special_flops=arithmetic+special,trace_repetitions=counts,traces=rows,
            event_deliveries=state.deliveries,candidate_scores=state.candidate_scores,
            counterfactual_values=state.counterfactual_values,selected_state_updates=state.selected_updates,
            scope='Representative extrapolation of actual first, middle and last-block work; excludes inherited fitting, RNG, index/traffic and contract checks; not measured energy')
    result.update(status='completed',blocks=blocks,work=work,arm_wall_s=timings,
        rows=[dict(arm=name,bpc=totals[name]/(a.n-1)/math.log(2),
            updates=len(starts) if name=='online' else 0,
            parameter_change_l2=math.sqrt(sum(float((p.detach()-base.state_dict()[key]).square().sum())
                for key,p in models[name].named_parameters()))) for name in models])
    assert result['rows'][0]['parameter_change_l2']==0.
    persist(completed=True);running.unlink(missing_ok=True)
    print(json.dumps(dict(completed=a.tag,rows=result['rows'])),flush=True)


if __name__=='__main__':
    main()
