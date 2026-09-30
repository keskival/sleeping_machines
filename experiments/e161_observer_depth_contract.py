"""Numerical research contract for directional identity-depth output units."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.observer_conditioned_depth import grow_observer_conditioned,materialize_observer_maps


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=Path('experiments/results/e161')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(161);started=time.perf_counter()
    base=CoalescedEventStateEncoder(sources=16,width=8,modes=4,depth=6,classes=3).double().train()
    grown,geometry=grow_observer_conditioned(base,12)
    source=torch.arange(24)%16
    raw_t=torch.tensor([.002,.008,.013,.019,.023,.029,.032,.038,.043,.049,.052,.058]*2,dtype=torch.double)
    assignment=torch.arange(12).repeat_interleave(2)
    times=torch.tensor([.01,.02,.03,.04,.05,.06]*2,dtype=torch.double)
    receivers=torch.arange(2).repeat_interleave(6);counts=torch.ones(12,dtype=torch.double)*2
    data=(source,raw_t,assignment,times,receivers,counts,2)
    old,old_payload,old_times,_=base(*data);new,new_payload,new_times,_=grown(*data)
    value_error=float((old-new).abs().max());payload_error=float((old_payload-new_payload).abs().max())
    time_error=float((new_times-old_times-.036).abs().max())
    if max(value_error,payload_error,time_error)>1e-13:raise ValueError('Identity not preserved')
    labels=torch.tensor([0,2]);F.cross_entropy(old,labels).backward();F.cross_entropy(new,labels).backward()
    named=dict(grown.named_parameters());teacher_error=0.
    for name,param in base.named_parameters():
        other=named[name]
        if param.grad is None:
            if other.grad is not None and float(other.grad.abs().max())>1e-13:raise ValueError('New old teacher')
        else:teacher_error=max(teacher_error,float((param.grad-other.grad).abs().max()))
    live=[float(layer.output.parametrizations.weight.original.grad.norm()) for layer in grown.layers[6:]]
    if teacher_error>1e-12 or min(live)<=0:raise ValueError('Old/live teacher failure')
    layer=grown.layers[6];x=torch.randn(5,8,dtype=torch.double);state=torch.randn(5,8,dtype=torch.double)
    t=torch.linspace(0,.4,5,dtype=torch.double);g=torch.randn_like(x)
    units=layer.output.parametrizations.weight[0]
    R=units(torch.eye(8,dtype=torch.double))
    features=layer.norm(state)*torch.sigmoid(layer.gate(F.gelu(x)))
    parameter=layer.output.parametrizations.weight.original
    value,_,_=layer.emit(x,state,t,credit=False)
    actual=torch.autograd.grad((g*value).sum(),parameter)[0]
    explicit=layer.gain*R.T@g.T@features
    local_error=float((actual-explicit).abs().max())
    head=base.head.weight.detach();H=head-head.mean(0,keepdim=True)
    observed=float(torch.linalg.svdvals(H@R)[0])
    target=torch.randn_like(parameter);coordinate_error=float((R@torch.linalg.solve(R,target)-target).abs().max())
    delta=torch.randn_like(parameter)*.03
    with torch.no_grad():parameter.copy_(delta)
    changed,_,_=layer.emit(x,state,t,credit=False)
    before_logits=F.linear(x,head,base.head.bias);after_logits=F.linear(changed,head,base.head.bias)
    p=before_logits.softmax(-1)
    kl=(p*(before_logits.log_softmax(-1)-after_logits.log_softmax(-1))).sum(-1)
    bound=layer.gain**2*8*float(torch.linalg.matrix_norm(delta,ord=2))**2/4
    if float(kl.max())>bound+1e-12 or observed>1+1e-12:raise ValueError('Finite observer bound failure')
    grown.eval();deployment=materialize_observer_maps(grown)
    fold_error=float((grown(*data)[0]-deployment(*data)[0]).abs().max())
    if max(local_error,coordinate_error,fold_error)>1e-12:raise ValueError('Coordinate/fold failure')
    # Check the normalization radial derivative separately from model accuracy.
    z=torch.randn(8,dtype=torch.double);z-=z.mean();z.requires_grad_(True)
    gamma=.003;epsilon=1e-5
    def norm(q):
        centered=q-q.mean();return gamma*centered/torch.sqrt(centered.square().mean()+epsilon)
    radial=torch.autograd.functional.jvp(norm,z,z.detach())[1]
    coefficient=gamma*epsilon/float(z.square().mean()+epsilon)**1.5
    radial_error=float((radial-coefficient*z.detach()).abs().max())
    if radial_error>1e-12:raise ValueError('Normalization radial teacher mismatch')
    result=dict(status='completed',function_max_error=value_error,payload_max_error=payload_error,
        added_delay_max_error=time_error,old_teacher_max_error=teacher_error,
        new_output_teacher_norms=live,explicit_output_teacher_max_error=local_error,
        observed_output_operator_norm=observed,output_coordinate_inverse_error=coordinate_error,
        maximum_finite_kl=float(kl.max()),finite_kl_bound=bound,deployment_fold_max_error=fold_error,
        normalization_radial_teacher_max_error=radial_error,growth_geometry=geometry,
        scope='Numerical local/identity/full-rank/fixed-observer contracts; no speech-quality or global optimization claim',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            (Path(__file__),Path('sleeping_machines/observer_conditioned_depth.py'),Path('sleeping_machines/event_state.py'))},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
