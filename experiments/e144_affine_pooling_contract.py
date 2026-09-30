"""Numerical experiment for exact modal endpoint AND pooled-state teachers."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sleeping_machines.affine_packets import compose,pooled_state_scan


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=Path("experiments/results/e144")/(a.tag+".json");out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError("Unique output required")
    torch.set_num_threads(1);torch.manual_seed(144);started=time.perf_counter()
    n,m=27,9
    rates=torch.rand(n,m,dtype=torch.float64,requires_grad=True)
    frequencies=torch.randn(n,m,dtype=torch.float64,requires_grad=True)
    dt=torch.rand(n,dtype=torch.float64)*.02
    transitions=torch.exp(torch.complex(-rates,frequencies)*dt[:,None])
    drive=torch.randn(n,m,dtype=torch.complex128,requires_grad=True)
    weights=(torch.rand(n,dtype=torch.float64)+.1).requires_grad_()
    packet_first=torch.arange(n)%3==0
    receiver_first=torch.arange(n)%9==0
    state,sums,mass,work=pooled_state_scan(transitions,drive,weights,packet_first,receiver_first)
    states,queries,masses=[],[],[]
    s=torch.zeros(m,dtype=torch.complex128);q=s.clone();total=weights.new_tensor(0.)
    for i in range(n):
        if receiver_first[i]:s=torch.zeros_like(s)
        if packet_first[i]:q=torch.zeros_like(q);total=weights.new_tensor(0.)
        s=transitions[i]*s+drive[i];q=q+weights[i]*s;total=total+weights[i]
        states.append(s);queries.append(q);masses.append(total)
    expected_state=torch.stack(states);expected_sum=torch.stack(queries);expected_mass=torch.stack(masses)
    errors={"state":float((state-expected_state).detach().abs().max()),
        "state_sum":float((sums-expected_sum).detach().abs().max()),
        "mass":float((mass-expected_mass).detach().abs().max())}
    target=torch.randn_like(sums)
    loss=torch.real((sums/mass[:,None])*target.conj()).sum()
    reference=torch.real((expected_sum/expected_mass[:,None])*target.conj()).sum()
    params=(rates,frequencies,drive,weights)
    g=torch.autograd.grad(loss,params,retain_graph=True)
    h=torch.autograd.grad(reference,params)
    teacher_errors={name:float((x-y).abs().max()) for name,x,y in
        zip(("decay","frequency","drive","pool_weight"),g,h)}
    assert max(list(errors.values())+list(teacher_errors.values()))<1e-11
    # Associativity includes weighted-query resets, not just state composition.
    def descriptor():
        aa=torch.randn(1,m,dtype=torch.complex128)
        bb=torch.randn_like(aa);rr=torch.randn_like(aa);dd=torch.randn_like(aa)
        return aa,bb,rr,dd,torch.rand(1,dtype=torch.float64),torch.rand(1,dtype=torch.float64)
    d1,d2,d3=descriptor(),descriptor(),descriptor()
    left=compose(compose(d1,d2),d3);right=compose(d1,compose(d2,d3))
    associative_error=max(float((x-y).abs().max()) for x,y in zip(left,right))
    assert associative_error<1e-11
    endpoints=torch.arange(2,n,3)
    endpoint_mean=sums[endpoints]/mass[endpoints,None]
    assert torch.allclose(endpoint_mean,expected_sum[endpoints]/expected_mass[endpoints,None])
    result=dict(status="completed",events=n,modes=m,packets=len(endpoints),errors=errors,
        teacher_errors=teacher_errors,associative_error=associative_error,
        descriptor_compositions=work,dense_output_map_evaluations=len(endpoints),
        scope="Exact weighted linear-state pooling, including input-dependent precomputable modal transitions. No trained classifier or nonlinear raw-event equivalence",
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('sleeping_machines/affine_packets.py')]})
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
