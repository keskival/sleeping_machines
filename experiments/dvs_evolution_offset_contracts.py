"""Physical-time coupling, exact nesting, phase ownership and recovery contracts."""
import argparse
import copy
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_evolution_offset_benchmark as E
import dvs_native_contracts as C
import dvs_batched_benchmark as B
from sleeping_machines import batched_addressed_fit as K
from sleeping_machines.evolution_offset_heads import parameter_block


def gradient_map(model):
    return {n:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}


def serial(model,rows,seed):
    logits=[];states=[]
    for row in rows:
        z,state=E.N.predict(model,row,seed,True);logits.append(z);states.append(state)
    return torch.stack(logits),states


def integrated(config,rows):
    original=E.N.make_model(config).double();offset=E.make_model(config).double()
    old_state=original.state_dict();new_state=offset.state_dict()
    for name in old_state:torch.testing.assert_close(old_state[name],new_state[name],rtol=0,atol=0)
    target=torch.tensor([r['target'] for r in rows]);seed=1207
    old,os,_=K.forward(original,rows,seed);new,ns,_=E.forward(offset,rows,seed)
    torch.testing.assert_close(old,new,rtol=0,atol=0);C.equal(os,ns)
    F.cross_entropy(old,target,reduction='sum').backward();F.cross_entropy(new,target,reduction='sum').backward()
    og=gradient_map(original);ng=gradient_map(offset)
    for name in og:torch.testing.assert_close(og[name],ng[name],rtol=1e-11,atol=1e-12)
    assert ng['raw_evolution_offset'].norm()>1e-8
    # Nonzero offset: same serialized parameters across independent serial/batched code.
    with torch.no_grad():offset.raw_evolution_offset.fill_(.13)
    reference=copy.deepcopy(offset);offset.zero_grad();reference.zero_grad()
    bz,bs,_=E.forward(offset,rows,seed);sz,ss=serial(reference,rows,seed)
    torch.testing.assert_close(bz,sz,rtol=1e-10,atol=1e-11)
    for clip,state in enumerate(ss):
        for depth in range(offset.depth):
            for head in range(offset.heads):
                for unit in range(offset.pool):
                    key=(depth,head,0,unit);index=head*offset.pool+unit
                    if key in state.memories:
                        torch.testing.assert_close(state.memories[key],bs['memories'][depth][clip,index],rtol=1e-10,atol=1e-11)
                        torch.testing.assert_close(state.arrivals[key],bs['arrivals'][depth][clip,index],rtol=1e-10,atol=1e-12)
    F.cross_entropy(bz,target,reduction='sum').backward();F.cross_entropy(sz,target,reduction='sum').backward()
    bg=gradient_map(offset);sg=gradient_map(reference)
    for name in bg:torch.testing.assert_close(bg[name],sg[name],rtol=1e-8,atol=1e-9)
    return dict(zero_offset_original_state_logits_routes_and_gradients_match=True,
        nonzero_serial_batched_state_and_every_parameter_gradient_match=True,offset_gradient_nonzero=True)


def physical_time():
    config=E.parser().parse_args(['--tag','physical','--data','unused','--controls','unused','--payload','2','--heads','1','--depth','1'])
    model=E.make_model(config).double();rho=.7;omega=1.3
    with torch.no_grad():
        model.transport_rate.fill_(math.log(math.expm1(rho-1e-6)))
        model.transport_frequency.fill_(omega);model.raw_evolution_offset.fill_(.2)
    message=torch.tensor([.8,-.4],dtype=torch.float64);age=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    value=model.transport(message,age,0,0)
    age_derivative=[];offset_derivative=[]
    for component in range(2):
        da,db=torch.autograd.grad(value[component],(age,model.raw_evolution_offset),retain_graph=True)
        age_derivative.append(da);offset_derivative.append(db[0,0,0])
    da=torch.stack(age_derivative);db=torch.stack(offset_derivative);eps=1e-6
    numerical=(model.transport(message,age.detach()+eps,0,0)-model.transport(message,age.detach()-eps,0,0))/(2*eps)
    torch.testing.assert_close(da,numerical,rtol=1e-8,atol=1e-9)
    with torch.no_grad():
        model.raw_evolution_offset.add_(eps);positive=model.transport(message,age.detach(),0,0)
        model.raw_evolution_offset.sub_(2*eps);negative=model.transport(message,age.detach(),0,0)
        model.raw_evolution_offset.add_(eps)
    torch.testing.assert_close(db,(positive-negative)/(2*eps),rtol=1e-8,atol=1e-9)
    determinant=da[0]*db[1]-da[1]*db[0];assert abs(float(determinant))>1e-3
    # Rotation-only case has collinear age/phase directions; damping makes them independent.
    from sleeping_machines.parallel_head_race_language import precise_rotate
    phase=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    pure=precise_rotate(message,(omega*age+phase).reshape(1));cols=[]
    for component in range(2):cols.append(torch.autograd.grad(pure[component],(age,phase),retain_graph=True))
    assert abs(float(cols[0][0]*cols[1][1]-cols[1][0]*cols[0][1]))<1e-12
    return dict(physical_age_gradient_retained_and_matches_finite_difference=True,
        phase_offset_gradient_matches_finite_difference=True,damped_time_offset_jacobian_determinant=float(determinant),
        rotation_only_time_offset_rank_one=True)


def ownership(config,rows):
    config.update_schedule='alternating';model=E.make_model(config);optimizer=torch.optim.Adam(model.parameters(),lr=config.lr);trials=[]
    for expected in ('message','route','message'):
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        states={n:copy.deepcopy(optimizer.state.get(p,{})) for n,p in model.named_parameters()}
        result=E.train_window(model,optimizer,rows,config,1,True);assert result['phase']==expected
        for stage in result['stages'].values():assert stage['formula_coverage_complete'],stage['unsupported_floating_operators']
        for n,p in model.named_parameters():
            if parameter_block(n) not in (expected,'shared'):
                torch.testing.assert_close(p,before[n],rtol=0,atol=0);C.equal(states[n],optimizer.state.get(p,{}));assert p.grad is None
        changed=not torch.equal(before['raw_evolution_offset'],model.raw_evolution_offset)
        assert changed==(expected=='message')
        trials.append(dict(phase=expected,offset_updated=changed,inactive_parameters_and_adam_state_preserved=True))
    return trials


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--data',required=True);p.add_argument('--controls',required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1);out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique tag required')
    common=['--data',a.data,'--controls',a.controls,'--fit','4','--dev','2','--epochs','2','--update-targets','3']
    config=E.parser().parse_args(['--tag','contract']+common);fitting,_,metadata=E.N.load(config)
    nesting=integrated(config,fitting[:2]);coupling=physical_time();phases=ownership(config,fitting[:2]);recoveries=[]
    with tempfile.TemporaryDirectory(prefix='dvs-evolution-offset-') as tmp:
        for schedule in ('joint','alternating'):
            continuous=E.parser().parse_args(['--tag',schedule+'_continuous','--update-schedule',schedule]+common)
            first=E.run(continuous,tmp)
            for sample in first['work_samples']:
                for stage in sample['stages'].values():assert stage['formula_coverage_complete'],stage['unsupported_floating_operators']
            resumed=E.parser().parse_args(['--tag',schedule+'_resumed','--update-schedule',schedule]+common)
            resumed.stop_after_updates=1;E.run(resumed,tmp);resumed.stop_after_updates=None;resumed.resume=True;second=E.run(resumed,tmp)
            x=torch.load(Path(tmp)/(continuous.tag+'.progress.pt'),weights_only=False);y=torch.load(Path(tmp)/(resumed.tag+'.progress.pt'),weights_only=False)
            for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):C.equal(x[key],y[key])
            for key in ('final','activity','work','work_samples','selected_epoch','evolution_offset_protocol'):C.equal(first[key],second[key])
            recoveries.append(dict(schedule=schedule,full_partial_window_counts=first['window_size_counts'],
                actual_model_adam_phase_cursor_rng_recovery=True,formula_coverage_complete=True))
    result=dict(status='completed',args=vars(a),contracts_passed=4,data=metadata,nesting=nesting,physical_time=coupling,
        phase_ownership=phases,recovery=recoveries,source_sha256={**E.sources(),'experiments/dvs_native_contracts.py':E.N.sha(ROOT/'experiments/dvs_native_contracts.py')},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Zero-offset native and nonzero serial/batched gradients/state; physical-time and independent offset finite differences/rank cases; inactive parameters/Adam state and offset ownership; actual joint/alternating full/partial interrupted recovery/accounting. Numerical readiness, not held-out benefit or independent signal/evolution clocks.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
