"""Norm, adjoint, depth, angle and hidden-state boundaries of event scattering."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e136_event_scattering import scatter


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e136')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    torch.set_num_threads(1);torch.manual_seed(206)
    e,d,banks,depth=9,4,3,12
    packet=torch.randn(e,d,dtype=torch.float64,requires_grad=True)
    initial=torch.randn(banks,d,dtype=torch.float64,requires_grad=True)
    angle=torch.linspace(-2.,2.,e,dtype=torch.float64,requires_grad=True)
    keys=torch.arange(e)%banks
    y,s,work=scatter(packet,angle,keys,initial)
    y0,s0,_=scatter(packet,angle,keys,initial,True)
    scan_error=float(torch.cat(((y-y0).flatten(),(s-s0).flatten())).detach().abs().max())
    assert scan_error<2e-14 and work<2*e
    energy_in=(packet.square().sum()+initial.square().sum()).detach()
    energy_out=(y.square().sum()+s.square().sum()).detach()
    assert abs(float(energy_out-energy_in))<1e-12
    gy,gs=torch.randn_like(y),torch.randn_like(s)
    gp,gi,ga=torch.autograd.grad((gy*y).sum()+(gs*s).sum(),(packet,initial,angle))
    adjoint_error=float((gp.square().sum()+gi.square().sum()-gy.square().sum()-gs.square().sum()).abs())
    assert adjoint_error<1e-12
    fd=[];eps=1e-6
    for j in range(e):
        plus=angle.detach().clone();minus=plus.clone();plus[j]+=eps;minus[j]-=eps
        yp,sp,_=scatter(packet.detach(),plus,keys,initial.detach())
        ym,sm,_=scatter(packet.detach(),minus,keys,initial.detach())
        fd.append(((gy*(yp-ym)).sum()+(gs*(sp-sm)).sum())/(2*eps))
    angle_error=float((ga-torch.stack(fd)).abs().max());assert angle_error<5e-9
    angles=[angle.detach()+.07*j for j in range(depth)]
    addresses=[(keys+j)%banks for j in range(depth)]
    def stack(values):
        x=values[:e*d].reshape(e,d);states=values[e*d:].reshape(depth,banks,d)
        finals=[]
        for j in range(depth):
            x,st,_=scatter(x,angles[j],addresses[j],states[j]);finals.append(st)
        return torch.cat((x.flatten(),torch.stack(finals).flatten()))
    joined=torch.randn(e*d+depth*banks*d,dtype=torch.float64)
    jac=torch.autograd.functional.jacobian(stack,joined)
    singular=torch.linalg.svdvals(jac)
    orthogonality_error=float((jac.T@jac-torch.eye(len(joined),dtype=joined.dtype)).abs().max())
    assert orthogonality_error<2e-14 and float((singular-1).abs().max())<2e-14
    # Local angle teacher uses the post-exchange state and packet.
    one_x=packet.detach()[:1];one_s=initial.detach()[:1];one_a=angle.detach()[:1].requires_grad_()
    oy,os,_=scatter(one_x,one_a,torch.zeros(1,dtype=torch.long),one_s)
    gy1,gs1=gy[:1],gs[:1]
    ag=torch.autograd.grad((gy1*oy).sum()+(gs1*os).sum(),one_a)[0]
    local=(gs1*oy-gy1*os).sum();assert abs(float(ag-local))<1e-14
    hidden_y,hidden_s,_=scatter(one_x,one_a.new_full((1,),torch.pi/2),torch.zeros(1,dtype=torch.long),torch.zeros_like(one_s))
    assert hidden_y.abs().max()<1e-14 and (hidden_s-one_x).abs().max()<1e-14
    result={'status':'completed','depth':depth,'augmented_jacobian_dimension':len(joined),
      'scan_vs_sequential_max_error':scan_error,'scan_compositions':work,
      'payload_state_energy_error':float((energy_out-energy_in).abs()),'adjoint_norm_squared_error':adjoint_error,
      'angle_gradient_finite_difference_max_error':angle_error,'local_angle_teacher_error':abs(float(ag-local)),
      'deep_augmented_orthogonality_max_error':orthogonality_error,'deep_singular_value_range':[float(singular.min()),float(singular.max())],
      'hidden_state_boundary_counterexample_verified':True,
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/e136_event_scattering.py'),Path('sleeping_machines/event_memory.py')]},
      'scope':'Conditional reversible value transport prototype with all local initial/final states included. No accuracy, arbitrary nonlinear-policy gradient, optimizer convergence or complete shared-model claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
