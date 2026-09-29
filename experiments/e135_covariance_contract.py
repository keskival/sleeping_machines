"""Audit the covariance query teacher and partitioned depth certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e135_content_memory import ContentMemory
from sleeping_machines.shared_event import RaceLayer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e135')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    torch.set_num_threads(1);torch.manual_seed(205)
    x=torch.randn(7,4,dtype=torch.float64)
    t=torch.arange(7,dtype=torch.float64)*.02
    c=torch.linspace(.5,2.,7,dtype=torch.float64);keys=torch.arange(7)%2
    tau=torch.tensor([.03,.2,.8],dtype=torch.float64)
    mem=ContentMemory(4).double()
    y,_,_=mem(x,t,c,keys,tau);teacher=torch.randn_like(y)
    actual=torch.autograd.grad((y*teacher).sum(),mem.query)[0]
    k=torch.tanh(x@mem.bounded(mem.key).T)
    predicted=torch.zeros_like(mem.query)
    for i in range(len(x)):
        eligible=(torch.arange(len(x))<=i)&(keys==keys[i])
        weight=c[eligible,None]*torch.exp(-(t[i]-t[eligible,None])/tau[None,:])
        p=weight/(weight.sum(0,keepdim=True)+1e-4)
        mean=torch.einsum('ek,ed->kd',p,x[eligible])
        kv=torch.einsum('ek,ej,ed->kjd',p,k[eligible],x[eligible])
        km=torch.einsum('ek,ej->kj',p,k[eligible])
        covariance=kv-km[:,:,None]*mean[:,None,:]
        coefficient=torch.einsum('kd,kjd->j',teacher[i],covariance)
        predicted+=mem.strength**2/mem.pairs*coefficient[:,None]*x[i]
    covariance_error=float((actual-predicted).detach().abs().max());assert covariance_error<2e-14
    with torch.no_grad():mem.query.copy_(torch.randn_like(mem.query)*.2)
    layer=RaceLayer(4,8,1.,False,memory_backend='linear').double();layer.memory=mem
    schedule={'winner':torch.arange(len(x))%3,'delays':x.new_full((len(x),3),.005)}
    jac=torch.autograd.functional.jacobian(lambda z:layer(z,t,c,keys,schedule=schedule)[0],x)
    jac=jac.reshape(x.numel(),x.numel());identity=torch.eye(x.numel(),dtype=x.dtype)
    radius=float(x.abs().max());ly=mem.input_lipschitz_bound(radius)
    w=layer.value.detach()/layer.value.detach().abs().sum(-1,keepdim=True).clamp_min(1)
    gain=float((w[:,:,:4].abs().sum(-1)+ly*w[:,:,4:8].abs().sum(-1)).max())
    measured=float((jac-identity).abs().sum(-1).max());bound=layer.alpha*gain
    assert measured<=bound+1e-12 and bound<1
    inverse_gain=float(torch.linalg.inv(jac).abs().sum(-1).max())
    assert inverse_gain<=1/(1-bound)+1e-12
    result={'status':'completed','covariance_teacher_max_error':covariance_error,
      'covariance_teacher_norm':float(actual.norm()),'fixed_schedule_residual_jacobian_max_row_sum':measured,
      'partitioned_residual_bound':bound,'actual_inverse_max_row_sum':inverse_gain,
      'certified_inverse_max_row_sum_bound':1/(1-bound),
      'first_order_content_weight_ratio_bound':(1+mem.strength**2)/(1-mem.strength**2),
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/e135_content_memory.py'),Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py')]},
      'scope':'Small deterministic numerical checks of equations 205.5/205.6, not a global proof or SHD optimization/quality result.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
