"""Bounded integrated joint-feature learning; no language/supremacy claim."""
import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from balanced_joint_protocol import joint_examples,data_hash,validate_pairs
import count_reference_language as counts
from native_language_helpers import source_hashes as parent_sources
from race_language_screen import RaceAudit
from sleeping_machines.native_stream_language import NativeStreamLanguageModel
from sleeping_machines.dilated_delay_taps import TappedNativeStreamLanguageModel


class JointAudit(RaceAudit):
    def formula(self,func,args,kwargs,out):
        if str(func)=='aten.floor.default':
            return 0,out.numel(),0,'Delay-index floor evaluation (unit special; conversion/control traffic separate)'
        if str(func)=='aten.rsub.Scalar':
            return out.numel(),0,0,'Reverse scalar subtraction'
        return super().formula(func,args,kwargs,out)


def capture(action):
    with JointAudit() as audit:action()
    result=audit.result()
    if not result['formula_coverage_complete']:raise ValueError(result['unsupported_floating_operators'])
    result['exponential_random_draws']=sum(r['floating_output_elements'] for r in result['operators'].values()
        if r['classification'].startswith('RNG generation'))
    return result


def sources():
    names=['experiments/balanced_joint_benchmark.py','experiments/balanced_joint_protocol.py',
        'experiments/paired_parity_protocol.py','experiments/count_reference_language.py',
        'experiments/race_language_screen.py','sleeping_machines/operation_audit.py',
        'sleeping_machines/dilated_delay_taps.py']
    return {**parent_sources(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True);p.add_argument('--model',choices=('native','tapped'),default='native')
    p.add_argument('--payload',type=int,default=8);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--pool',type=int,default=2)
    p.add_argument('--gap',type=int,default=8);p.add_argument('--fit-groups',type=int,default=8)
    p.add_argument('--dev-groups',type=int,default=16);p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--update-targets',type=int,default=8);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=6);p.add_argument('--fit-data-seed',type=int,default=71001)
    p.add_argument('--dev-data-seed',type=int,default=72001);p.add_argument('--contracts')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-updates',type=int)
    return p


def scientific_args(a):
    return {k:v for k,v in vars(a).items() if k not in ('tag','resume','stop_after_updates')}


def make_model(a):
    torch.manual_seed(a.seed)
    cls=NativeStreamLanguageModel if a.model=='native' else TappedNativeStreamLanguageModel
    return cls(a.payload,a.depth,a.pool,a.heads)


def episode(model,row,seed):
    # Shared noise for all four bit pairs of a suffix. No target enters input,
    # address, clock, seed or hidden state. Both fitting and evaluation reset.
    with torch.random.fork_rng():
        torch.manual_seed(seed+row['group'])
        logits,state=model.forward_chunk(torch.tensor(row['inputs']))
    return logits[-1,:2],state


def train_window(model,opt,rows,a,epoch):
    model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;events=keys=updates=teachers=0;max_bytes=0
    for row in rows:
        logits,state=episode(model,row,100000+a.seed+10000*epoch)
        loss=F.cross_entropy(logits.unsqueeze(0),torch.tensor([row['target']]),reduction='sum')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite query loss')
        loss.backward();loss_sum+=float(loss.detach())
        events+=state.events;keys+=state.candidate_scores;updates+=state.selected_updates
        teachers+=state.counterfactual_values;max_bytes=max(max_bytes,state.storage()['persistent_tensor_bytes'])
    # Each backward contributes a sum over one target; normalize exactly once.
    for p in model.parameters():
        if p.grad is not None:p.grad.div_(len(rows))
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    opt.step()
    return dict(targets=len(rows),loss_sum=loss_sum,input_events=events,key_scores=keys,
        selected_updates=updates,counterfactual_values=teachers,max_state_tensor_bytes=max_bytes,
        gradient_norm_before_clip=norm)


@torch.no_grad()
def evaluate(model,rows):
    model.eval();losses=[];probabilities=[];hits=[];events=keys=updates=0;max_bytes=0
    for row in rows:
        z,state=episode(model,row,314159)
        losses.append(float(F.cross_entropy(z.unsqueeze(0),torch.tensor([row['target']])))/math.log(2))
        probabilities.append(float(z.softmax(-1)[1]));hits.append(int(z.argmax())==row['target'])
        events+=state.events;keys+=state.candidate_scores;updates+=state.selected_updates
        max_bytes=max(max_bytes,state.storage()['persistent_tensor_bytes'])
    return dict(targets=len(rows),query_bits=float(np.mean(losses)),accuracy=float(np.mean(hits)),
        per_target_bits=losses,class1_probabilities=probabilities,input_events=events,
        key_scores=keys,selected_updates=updates,max_state_tensor_bytes=max_bytes)


def query_counts(tables,row,prefix_updates):
    hist=row['inputs'];query=counts.Counts(tables.K)
    for k in range(tables.K+1):
        code=counts.ctx_code(hist,k)
        c=tables.t[k].get(code,np.zeros(27)).copy()
        if prefix_updates:
            for position,symbol in enumerate(hist):
                if position>=k and counts.ctx_code(hist[:position],k)==code:c[symbol]+=1
        query.t[k][code]=c
    return query


def count_controls(fit,dev):
    started=time.perf_counter();tables=counts.Counts(8);increments=0
    for row in fit:
        sequence=row['inputs']+[row['target']];tables.add_sequence(sequence)
        increments+=sum(max(0,len(sequence)-k) for k in range(9))
    vectors={};scores={(k,m,adaptive):[] for k in range(9) for m in ('wb','ad') for adaptive in (False,True)}
    for row in dev:
        for adaptive in (False,True):
            query=query_counts(tables,row,adaptive)
            v=np.stack([query.t[k][counts.ctx_code(row['inputs'],k)] for k in range(9)])
            key=(row['group'],adaptive)
            if key in vectors:np.testing.assert_array_equal(v,vectors[key])
            else:vectors[key]=v
            for k in range(9):
                for method in ('wb','ad'):
                    p=counts.predict(query,row['inputs'],k,method)[:2];p=p/p.sum()
                    scores[k,method,adaptive].append(-math.log2(p[row['target']]))
    return dict(rows=[dict(order=k,method=m,prefix_updates=adaptive,query_bits=float(np.mean(v)))
        for (k,m,adaptive),v in scores.items()],all_query_count_vectors_including_root_equal=True,
        represented_fitting_count_increments=increments,fitting_passes=1,
        wall_s=time.perf_counter()-started,
        scope='All 36 controls separately reported, binary answer alphabet shared with neural arms. Fitting counts reset at episode boundaries; optional observed-prefix updates reset per development episode, no previous development labels. Integer lookups/copies are separate from neural FLOPs. Bound does not cover arbitrary reads at other count addresses.')


def _atomic_json(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def run(a,output_parent=None):
    if Path(a.tag).name!=a.tag:raise ValueError('Plain unique tag required')
    if not(8<=a.gap<=64 and 1<=a.fit_groups<=32 and 1<=a.dev_groups<=64 and 1<=a.epochs<=32):
        raise ValueError('Bounded episode protocol required')
    if min(a.payload,a.depth,a.heads,a.pool,a.update_targets)<1 or a.payload%2 or a.update_targets%4:
        raise ValueError('Even payload and complete quartets per optimizer window required')
    if a.lr<=0 or (a.stop_after_updates is not None and a.stop_after_updates<1):raise ValueError('Positive learning/run budget')
    torch.set_num_threads(1);started=time.perf_counter();sha=sources()
    if a.contracts:
        contract=json.loads((ROOT/a.contracts).read_text())
        if contract['status']!='completed' or contract['source_sha256']!=sha:
            raise ValueError('Completed current-source contracts required')
    elif a.fit_groups>2 or a.dev_groups>2 or a.epochs>2:
        raise ValueError('Only bounded contracts/smokes may omit completed prerequisites')
    parent=ROOT/'experiments/results/balanced_joint' if output_parent is None else Path(output_parent)
    parent.mkdir(parents=True,exist_ok=True);out=parent/(a.tag+'.json');running=out.with_suffix('.running.json')
    ckpath=out.with_suffix('.progress.pt')
    if out.exists() or (not a.resume and (running.exists() or ckpath.exists())):raise ValueError('Preserve previous result')
    fit=joint_examples(a.gap,a.fit_groups,a.fit_data_seed);dev=joint_examples(a.gap,a.dev_groups,a.dev_data_seed)
    proof=validate_pairs(fit);validate_pairs(dev)
    if {tuple(r['inputs'][6:-1]) for r in fit}&{tuple(r['inputs'][6:-1]) for r in dev}:
        raise ValueError('Held-out noise suffixes must be disjoint')
    model=make_model(a);opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    epoch=1;cursor=0;updates=0;targets=0;pass_loss=0.;best=float('inf');best_state=None
    initial=copy.deepcopy(model.state_dict());rng=torch.get_rng_state()
    if a.resume:
        ck=torch.load(ckpath,weights_only=False)
        if ck['scientific_args']!=scientific_args(a) or ck['source_sha256']!=sha:raise ValueError('Changed recovery configuration/source')
        model.load_state_dict(ck['online_model']);opt.load_state_dict(ck['optimizer'])
        epoch,cursor,updates,targets,pass_loss=ck['cursor'];best,best_state=ck['best'],ck['best_state']
        result=ck['result'];initial=ck['initial'];torch.set_rng_state(ck['torch_rng']);rng=ck['torch_rng']
    else:
        result=dict(status='running',args=vars(a),source_sha256=sha,parameters=sum(p.numel() for p in model.parameters()),
            fitting_data_sha256=data_hash(fit),development_data_sha256=data_hash(dev),
            numerical_protocol=proof,initial_dev=evaluate(model,dev),curve=[],work_samples=[],
            counts=count_controls(fit,dev),activity=dict(input_events=0,key_scores=0,selected_updates=0,counterfactual_values=0),
            protocol=dict(objective='Query-only binary cross entropy for every arm; no bit/XOR feature supplied',
                state='New empty state per complete episode; full-episode credit',
                fit_targets_per_pass=len(fit),input_events_per_episode=len(fit[0]['inputs']),
                development='Held-out paired suffixes; min query loss over fixed passes; exploratory',
                race_noise='Coupled by group/pass, independent of target; evaluation cannot change fitting RNG',
                scope='Integrated controlled dependency diagnostic, not semantic language or general resource supremacy'),
            hardware=dict(device='cpu',threads=1,platform=platform.platform(),torch=torch.__version__))
    previous_wall=result.get('wall_s',0.)
    def persist(completed=False):
        result.update(wall_s=previous_wall+time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_json(out if completed else running,result)
        ck=dict(scientific_args=scientific_args(a),source_sha256=sha,online_model=model.state_dict(),optimizer=opt.state_dict(),
            initial=initial,best=best,best_state=best_state,cursor=(epoch,cursor,updates,targets,pass_loss),
            torch_rng=torch.get_rng_state(),result=result)
        tmp=ckpath.with_suffix('.pt.tmp');torch.save(ck,tmp);tmp.replace(ckpath)
    persist();print(json.dumps(dict(started=a.tag,initial=result['initial_dev']['query_bits'])),flush=True)
    while epoch<=a.epochs:
        while cursor<len(fit):
            end=min(cursor+a.update_targets,len(fit));box={}
            def action():box['value']=train_window(model,opt,fit[cursor:end],a,epoch)
            audited=epoch==1 and cursor in (0,((len(fit)-1)//a.update_targets)*a.update_targets)
            if audited:
                trace=capture(action);result['work_samples'].append(dict(epoch=epoch,cursor=cursor,targets=end-cursor,trace=trace))
            else:action()
            row=box['value'];pass_loss+=row['loss_sum'];updates+=1;targets+=row['targets'];cursor=end
            for k in result['activity']:result['activity'][k]+=row[k]
            result['max_training_state_tensor_bytes']=max(result.get('max_training_state_tensor_bytes',0),row['max_state_tensor_bytes'])
            persist()
            if a.stop_after_updates is not None and updates>=a.stop_after_updates:
                result['status']='interrupted_for_contract';persist();return result
        score=evaluate(model,dev)
        result['curve'].append(dict(epoch=epoch,fitting_query_bits=pass_loss/len(fit)/math.log(2),dev=score,
            optimizer_updates=updates,fitted_targets=targets,wall_s=previous_wall+time.perf_counter()-started))
        if score['query_bits']<best:
            best=score['query_bits'];best_state=copy.deepcopy(model.state_dict());result['selected_epoch']=epoch
        print(json.dumps(dict(epoch=epoch,fitting=result['curve'][-1]['fitting_query_bits'],
            dev_bits=score['query_bits'],dev_accuracy=score['accuracy'],updates=updates)),flush=True)
        epoch+=1;cursor=0;pass_loss=0.;persist()
    # Preserve online optimizer/model consistency; use a separate selected copy.
    selected=copy.deepcopy(model);selected.load_state_dict(best_state);selected.eval()
    result['final']=evaluate(selected,dev)
    inference_box={}
    def infer():inference_box['score']=evaluate(selected,dev[:4])
    inference_trace=capture(infer)
    samples=result['work_samples'];sample_targets=sum(s['targets'] for s in samples)
    arithmetic=sum(s['trace']['arithmetic_flops'] for s in samples)/sample_targets
    specials=sum(s['trace']['special_function_evaluations'] for s in samples)/sample_targets
    result['work']=dict(fitting_targets=targets,optimizer_updates=updates,
        whole_fit_arithmetic_flops_estimate=arithmetic*targets,whole_fit_special_evaluations_estimate=specials*targets,
        whole_fit_unit_special_flops_estimate=(arithmetic+specials)*targets,
        fit_unit_special_flops_per_target_estimate=arithmetic+specials,
        inference_query_targets=4,inference_input_events=inference_box['score']['input_events'],
        inference_unit_special_flops_per_target=(inference_trace['arithmetic_flops']+inference_trace['special_function_evaluations'])/4,
        inference_trace=inference_trace,formula_coverage_complete=True,
        scope='Whole-fit estimate extrapolates recorded complete first/last optimizer windows of pass1, not exhaustive routing-occupancy accounting. Prefix forward/loss/backward/normalization/clipping/Adam and losing proposals charged; 2 FLOPs/MAC plus unit specials. Development passes, RNG, integer lookups, checkpoint I/O, traffic and energy separate. Inference includes all episode inputs and query loss.')
    result['parameter_delta']={n:float((p.detach()-initial[n]).norm()) for n,p in model.named_parameters()}
    result['evaluation_query_targets']=(a.epochs+2)*len(dev)+4
    result['evaluation_input_events']=result['evaluation_query_targets']*len(dev[0]['inputs'])
    result['exploratory_dependency_gate_passed']=result['final']['accuracy']>=.75 and result['final']['query_bits']<=.8
    result['status']='completed';persist(completed=True);running.unlink(missing_ok=True)
    print(json.dumps(dict(completed=a.tag,final_bits=result['final']['query_bits'],gate=result['exploratory_dependency_gate_passed'])),flush=True)
    return result


if __name__=='__main__':run(parser().parse_args())
