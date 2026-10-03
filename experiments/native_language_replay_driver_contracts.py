"""End-to-end replay fitting, interrupted recovery and accounting contracts."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import native_language_replay_benchmark as D
import dvs_native_benchmark as N


def exact(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:exact(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for k,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'.'+str(k))
    elif hasattr(a,'__dict__'):
        assert type(a)==type(b),path
        exact(vars(a),vars(b),path)
    else:assert type(a)==type(b) and a==b,(path,a,b)


def checkpoint(folder,tag):return torch.load(folder/(tag+'.progress.pt'),weights_only=False)


def compare(reference,recovered,work=True):
    for key in ('model','optimizer','gradients','stream_state','cursor','current_window','counters','best_state','best','torch_rng'):
        exact(reference[key],recovered[key],key)
    for key in ('parameters','initial_dev','curve','selected_epoch','final','activity','learner_counters','window_size_counts'):
        exact(reference['result'][key],recovered['result'][key],key)
    if work:
        for key in ('work','work_samples'):exact(reference['result'][key],recovered['result'][key],key)


def rejects(call):
    try:call()
    except ValueError as e:return str(e)
    raise AssertionError('Invalid recovery/overwrite was accepted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json');assert not out.exists()
    started=time.perf_counter();checks=[];rows=[];train=torch.tensor([1,2,1,3,1,2,4,1,2]);dev=torch.tensor([2,3,4,1,2])
    contract='experiments/results/diagnostics/local_causal_language_replay_accumulator_20261003T012100Z.json'
    with tempfile.TemporaryDirectory(prefix='language-replay-driver-') as tmp:
        folder=Path(tmp)
        for family in ('private','depth'):
            args=D.parser().parse_args(['--tag',family+'_full','--receiver-sharing',family,'--contracts',contract,
                '--fit','9','--dev','5','--dev-start','1000','--payload','4','--chunk','2',
                '--update-targets','4','--warmup-targets','8'])
            r=D.run(args,folder,train,dev);base=checkpoint(folder,args.tag)
            assert r['status']=='completed' and r['work']['optimizer_updates']==2
            assert r['activity']==dict(targets=8,key_scores=256,selected_updates=128,shadow_lanes=256,shadow_events=512)
            assert all(stage['formula_coverage_complete'] for row in r['work_samples'] for stage in row['stages'].values())
            checks.append(f'{family}: two actual target-weighted replay updates complete with all shadow/optimizer work covered')
            for boundary in ('pending','updated'):
                arm=copy.deepcopy(args);arm.tag=family+'_'+boundary
                if boundary=='pending':arm.stop_after_microchunks=1
                else:arm.stop_after_updates=1
                stopped=D.run(arm,folder,train,dev);saved=checkpoint(folder,arm.tag)
                assert stopped['status']=='running' and saved['stream_state'] is not None
                assert saved['counters']['pending_targets']==(2 if boundary=='pending' else 0)
                assert saved['counters']['updates']==(0 if boundary=='pending' else 1)
                assert saved['cursor']['next_target']==(2 if boundary=='pending' else 4)
                if boundary=='pending':
                    assert saved['gradients'] and saved['current_window']['accumulate']
                    bad=copy.deepcopy(arm);bad.resume=True;bad.lr*=2
                    rejects(lambda:D.run(bad,folder,train,dev))
                    bad=copy.deepcopy(arm);bad.resume=True;changed=train.clone();changed[0]=5
                    rejects(lambda:D.run(bad,folder,changed,dev))
                    cp=folder/(arm.tag+'.progress.pt');broken=copy.deepcopy(saved)
                    broken['source_sha256']['experiments/native_language_replay_benchmark.py']='0'*64
                    torch.save(broken,cp);rejects(lambda:D.run(bad,folder,train,dev));torch.save(saved,cp)
                    checks.append(f'{family}: source/settings/data mismatch refuses saved pending-gradient recovery')
                arm.stop_after_microchunks=None;arm.stop_after_updates=None;arm.resume=True
                D.run(arm,folder,train,dev);compare(base,checkpoint(folder,arm.tag))
                checks.append(f'{family}: {boundary} interruption recovers every weight/Adam/gradient/state/RNG/counter/cursor and curve/work bitwise')
            arm=copy.deepcopy(args);arm.tag=family+'_untraced';arm.no_trace=True
            plain=D.run(arm,folder,train,dev);compare(base,checkpoint(folder,arm.tag),False)
            assert plain['work']['whole_fit_unit_special_flops_estimate'] is None
            checks.append(f'{family}: traced and untraced fitting have identical actual learning, state, RNG and DEV selection')
            rejects(lambda:D.run(args,folder,train,dev));checks.append(f'{family}: completed result cannot be overwritten')
            rows.append(dict(family=family,parameters=r['parameters'],targets=8,optimizer_updates=2,
                shadow_lanes=r['activity']['shadow_lanes'],shadow_events=r['activity']['shadow_events'],
                fitting_work=r['work'],work_samples=r['work_samples'],
                scope='Synthetic eight-target correctness fit, not text8 quality evidence'))
    names=['experiments/native_language_replay_driver_contracts.py','experiments/theory/114_language_replay_benchmark_driver_contract.md']
    result=dict(status='completed',contracts_passed=len(checks),contracts=checks,families=rows,
        args=vars(a),source_sha256={**D.sources(),**{n:N.sha(ROOT/n) for n in names}},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Restartable sibling full-write causal native language driver numerical admission only; '
            'all fitting serial in one guarded job; active AWS original-teacher controls unchanged; no benchmark gain.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','contracts_passed','wall_s','max_rss_kb')}))


if __name__=='__main__':main()
