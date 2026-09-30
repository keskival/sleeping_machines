"""Numerical temporal counterfactual identity and shrinking orbit geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e148')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(148);started=time.perf_counter()
    A=torch.zeros(4,4,dtype=torch.float64)
    for j,(rate,omega) in enumerate(((.2,3.),(1.2,7.))):
        A[2*j:2*j+2,2*j:2*j+2]=torch.tensor([[-rate,-omega],[omega,-rate]],dtype=torch.float64)
    times=torch.tensor([.1,.3,.55,.9],dtype=torch.float64)
    drive=torch.randn(4,4,dtype=torch.float64)
    def states(t):
        s=torch.zeros(4,dtype=torch.float64);last=t[0];rows=[]
        for ti,bi in zip(t,drive):
            s=torch.linalg.matrix_exp(A*(ti-last))@s+bi;rows.append(s);last=ti
        return torch.stack(rows)
    base=states(times);delta=.04;changed=times.clone();changed[1]+=delta;shifted=states(changed)
    contribution=torch.linalg.matrix_exp(A*(times[-1]-times[1]))@drive[1]
    expected=(torch.linalg.matrix_exp(-A*delta)-torch.eye(4,dtype=A.dtype))@contribution
    future_error=float((shifted[-1]-base[-1]-expected).abs().max())
    own_expected=(torch.linalg.matrix_exp(A*delta)-torch.eye(4,dtype=A.dtype))@(base[1]-drive[1])
    own_error=float((shifted[1]-base[1]-own_expected).abs().max())
    differentiable=times.clone().requires_grad_();q=torch.randn(4,4,dtype=torch.float64)
    loss=(states(differentiable)*q).sum();teacher=torch.autograd.grad(loss,differentiable)[0]
    adjoints=[None]*4;adjoints[-1]=q[-1]
    for j in range(2,-1,-1):
        F=torch.linalg.matrix_exp(A*(times[j+1]-times[j]))
        adjoints[j]=q[j]+F.T@adjoints[j+1]
    local=q[1]@(A@(base[1]-drive[1]))-adjoints[2]@(A@torch.linalg.matrix_exp(A*(times[2]-times[1]))@drive[1])
    teacher_error=float((teacher[1]-local).abs())
    assert max(future_error,own_error,teacher_error)<1e-11
    h=torch.tensor([1.,.3,-.2,.8],dtype=torch.float64)
    krylov=torch.stack([torch.linalg.matrix_power(A,j)@h for j in range(1,5)],1)
    assert torch.linalg.matrix_rank(krylov)==4
    nodes=torch.tensor([-1.5,-.5,.5,1.5],dtype=torch.float64)
    rows=[]
    for epsilon in (.03,.01,.003,.001):
        D=torch.stack([(torch.linalg.matrix_exp(-A*(epsilon*c))-torch.eye(4,dtype=A.dtype))@h for c in nodes],1)
        singular=torch.linalg.svdvals(D)
        rows.append(dict(epsilon=epsilon,singular_values=singular.tolist(),
            condition=float(singular[0]/singular[-1]),absolute_determinant=float(torch.linalg.det(D).abs())))
    exponents=[]
    for i in range(4):
        ratio=rows[-1]['singular_values'][i]/rows[-2]['singular_values'][i]
        exponents.append(float(torch.log(torch.tensor(ratio,dtype=A.dtype))/torch.log(torch.tensor(rows[-1]['epsilon']/rows[-2]['epsilon'],dtype=A.dtype))))
    assert max(abs(x-y) for x,y in zip(exponents,(1,2,3,4)))<.02,exponents
    result=dict(status='completed',future_counterfactual_error=future_error,
        own_event_counterfactual_error=own_error,adjacent_interval_teacher_error=teacher_error,
        krylov_rank=4,orbit_rows=rows,small_delay_singular_exponents=exponents,
        scope='Fixed drives and unchanged event order in a linear state. Not an exact nonlinear suffix replay, trained optionality score or SHD accuracy result',
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
