"""Conditional angular/Lie support and scalar-reserve limits."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def generator(n,a,b):
    m=torch.zeros(n,n,dtype=torch.float64);m[a,b]=1;m[b,a]=-1
    return m


def closure(generators):
    basis=[]
    def add(m):
        v=m.clone()
        for q in basis:v-=torch.sum(v*q)*q
        norm=v.norm()
        if norm>1e-10:basis.append(v/norm);return True
        return False
    for m in generators:add(m)
    changed=True
    while changed:
        changed=False;previous=list(basis)
        for a in previous:
            for b in previous:changed=add(a@b-b@a) or changed
    return len(basis)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e136')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    torch.set_num_threads(1)
    left=generator(3,0,1);right=generator(3,1,2)
    commutator_error=float((left@right-right@left-generator(3,0,2)).abs().max());assert commutator_error==0
    connected=closure([generator(5,j,j+1) for j in range(4)])
    disconnected=closure([generator(5,0,1),generator(5,1,2),generator(5,3,4)])
    assert connected==10 and disconnected==4
    g=torch.diag(torch.tensor([1.,0.],dtype=torch.float64));c=g.clone()
    u=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64);rotated=u@g@u.T
    assert g.trace()==rotated.trace() and torch.trace(g@c)==1 and torch.trace(rotated@c)==0
    assert torch.trace(g@torch.eye(2))==torch.trace(rotated@torch.eye(2))
    result={'status':'completed','commutator_identity_max_error':commutator_error,
      'connected_five_coordinate_lie_dimension':connected,'disconnected_three_plus_two_dimension':disconnected,
      'same_scalar_trace_different_anisotropic_capacity':[float(torch.trace(g@c)),float(torch.trace(rotated@c))],
      'isotropic_scalar_reserve_preserved':True,
      'source_sha256':{str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
      'scope':'Static Lie closure with freely revisitable edges, and a scalar-trace counterexample. Does not prove causal event-schedule controllability or trained SHD performance. A state-conditioned scalar critic can encode alignment beyond trace.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
