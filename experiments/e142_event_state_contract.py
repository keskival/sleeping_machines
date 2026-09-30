"""Bounded numerical experiment for exact coalescing and deep event teachers."""
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
from sleeping_machines.event_state import (signed_state_scan, EventStateBlock,
    CoalescedEventStateEncoder)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tag",required=True)
    a=p.parse_args()
    out=Path("experiments/results/e142")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError("Unique tag required")
    torch.set_num_threads(1)
    torch.manual_seed(142)
    started=time.perf_counter()
    n,m=31,7
    t=torch.rand(n,dtype=torch.float64)
    k=torch.arange(n)%3
    drive=torch.randn(n,2*m,dtype=torch.float64,requires_grad=True)
    rates=torch.logspace(-1,1,m,dtype=torch.float64).requires_grad_()
    omega=torch.linspace(-13.,11.,m,dtype=torch.float64).requires_grad_()
    parallel,work=signed_state_scan(drive,t,k,rates,omega)
    serial,_=signed_state_scan(drive,t,k,rates,omega,sequential=True)
    reference=torch.zeros_like(parallel)
    for receiver in k.unique():
        ix=torch.nonzero(k==receiver).flatten()
        ix=ix[torch.argsort(t[ix],stable=True)]
        state=torch.zeros(m,dtype=torch.complex128)
        last=t[ix[0]]
        for i in ix:
            state=state*torch.exp(torch.complex(-rates,omega)*(t[i]-last))+torch.view_as_complex(drive[i].reshape(m,2))
            reference[i]=torch.view_as_real(state).flatten()
            last=t[i]
    primitive_error=float((parallel-reference).detach().abs().max())
    serial_error=float((parallel-serial).detach().abs().max())
    teacher=torch.randn_like(parallel)
    g1=torch.autograd.grad((parallel*teacher).sum(),(drive,rates,omega),retain_graph=True)
    g2=torch.autograd.grad((reference*teacher).sum(),(drive,rates,omega))
    primitive_teacher_error=max(float((u-v).abs().max()) for u,v in zip(g1,g2))
    assert max(primitive_error,serial_error,primitive_teacher_error)<1e-10
    # Packet closures coincide with the last raw impulse, making the reference
    # endpoints directly observable. Each receiver has its own ordered bursts.
    model=CoalescedEventStateEncoder(sources=11,width=16,modes=5,depth=6,classes=4).double()
    raw_t=torch.tensor([.003,.008,.010,.013,.017,.020,.031,.037,.040]*2,dtype=torch.float64)
    receiver=torch.tensor([0,0,0,1,1,1])
    closure=torch.tensor([.010,.020,.040]*2,dtype=torch.float64)
    assignment=torch.arange(6).repeat_interleave(3)
    source=torch.randint(0,11,(len(raw_t),))
    counts=torch.full((6,),3.,dtype=torch.float64)
    first=model.layers[0]
    captured=[]
    hook=first.output.register_forward_pre_hook(lambda mod,args:captured.append(args[0]))
    scores,values,arrivals,stats=model(source,raw_t,assignment,closure,receiver,counts,2)
    hook.remove()
    raw_drive=first.input(model.embedding(source))
    raw_receiver=receiver[assignment]
    full,_=signed_state_scan(raw_drive,raw_t,raw_receiver,F.softplus(first.raw_rate)+1e-6,first.frequency)
    endpoints=full[torch.arange(2,len(raw_t),3)]
    coalescing_error=float((captured[0]-endpoints).detach().abs().max())
    endpoint_teacher=torch.randn_like(endpoints)
    params=(model.embedding.weight,first.input.weight,first.raw_rate,first.frequency)
    ga=torch.autograd.grad((captured[0]*endpoint_teacher).sum(),params,retain_graph=True)
    gb=torch.autograd.grad((endpoints*endpoint_teacher).sum(),params)
    coalescing_teacher_error=max(float((u-v).abs().max()) for u,v in zip(ga,gb))
    assert max(coalescing_error,coalescing_teacher_error)<1e-10
    loss=F.cross_entropy(scores,torch.tensor([1,3]))
    loss.backward()
    credit=[]
    for layer in model.layers:
        credit.append({name:float(param.grad.norm()) for name,param in layer.named_parameters()
            if param.grad is not None})
    required=("input.weight","output.weight","raw_rate","frequency","gate.weight")
    for row in credit:
        assert all(row[name]>0 and torch.isfinite(torch.tensor(row[name])) for name in required),row
    assert all(row["clock.weight"]>0 for row in credit[:-1])
    # The completed-query readout ignores final emission time. Its last clock
    # is unobservable to this loss; expecting a teacher would be incorrect.
    assert "clock.weight" not in credit[-1]
    assert all(row["emitted_vectors"]==len(closure) for row in stats)
    assert torch.all(arrivals>=closure)
    # Future inputs cannot affect an earlier affine endpoint.
    extra=torch.randn(1,2*m,dtype=torch.float64)
    later,_=signed_state_scan(torch.cat((drive.detach(),extra)),torch.cat((t,t.new_tensor([2.]))),
        torch.cat((k,k.new_tensor([0]))),rates.detach(),omega.detach())
    prefix_error=float((later[:n]-parallel.detach()).abs().max())
    assert prefix_error<1e-10
    block=EventStateBlock(8,4,options=1,nonlinear=False,residual=False).double()
    x=torch.randn(12,8,dtype=torch.float64)
    times=torch.sort(torch.rand(12,dtype=torch.float64))[0]
    keys=torch.zeros(12,dtype=torch.long)
    state,_=signed_state_scan(block.input(x),times,keys,F.softplus(block.raw_rate)+1e-6,block.frequency)
    v,arrival,_=block(x,times,keys)
    inclusion_error=float((v-block.output(state)-block.direct*x).detach().abs().max())
    assert inclusion_error<1e-10
    result={"status":"completed","primitive_state_error":primitive_error,
        "primitive_serial_error":serial_error,"primitive_teacher_error":primitive_teacher_error,
        "coalesced_endpoint_error":coalescing_error,"coalesced_teacher_error":coalescing_teacher_error,
        "causal_prefix_error":prefix_error,"linear_operator_inclusion_error":inclusion_error,
        "six_layer_gradient_norms":credit,"events":len(raw_t),"packets":len(closure),
        "final_clock_teacher":"None: completed-query loss has no emission-time dependence",
        "emitted_vectors_per_layer":[r["emitted_vectors"] for r in stats],
        "primitive_scan_compositions":work,"fitting_loss":float(loss.detach()),
        "scope":"Numerical identities and nonzero credit, not SHD accuracy or global optimization proof. First affine endpoints/teachers exact; nonlinear output only at packet closures; clock credit is a local surrogate.",
        "wall_s":time.perf_counter()-started,"max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_sha256":{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('sleeping_machines/event_state.py'),Path('sleeping_machines/event_memory.py'),Path('sleeping_machines/rotating_memory.py')]}}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ('six_layer_gradient_norms','source_sha256')}),flush=True)


if __name__=="__main__":main()
