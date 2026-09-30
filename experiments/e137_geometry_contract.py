"""Finite distribution checks of observed control and covariance identities."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def covariance(points, probabilities):
    mean = probabilities @ points
    centered = points - mean
    return centered.T @ (probabilities[:, None] * centered)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); a=ap.parse_args()
    out=Path('experiments/results/e137')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists(): raise ValueError('Invalid output')
    torch.set_num_threads(1); torch.manual_seed(212); dtype=torch.float64
    p=torch.tensor([.1,.2,.3,.4],dtype=dtype)
    q=torch.tensor([.4,.1,.2,.3],dtype=dtype)
    teachers=p[None,:]-torch.eye(4,dtype=dtype)
    j=torch.randn(7,5,dtype=dtype); projection=torch.randn(4,7,dtype=dtype)
    gramian=j@j.T; kernel=projection@gramian@projection.T
    mu=p-q; cov=torch.diag(q)-q[:,None]*q[None,:]
    exact=torch.sum(q*torch.einsum('ki,ij,kj->k',teachers,kernel,teachers))
    predicted=mu@kernel@mu+torch.trace(kernel@cov)
    expectation_error=float((exact-predicted).abs())
    measured_cov=covariance(teachers,q)
    covariance_error=float((measured_cov-cov).abs().max())
    deterministic=torch.tensor([1.,0.,0.,0.],dtype=dtype)
    deterministic_cov=covariance(teachers,deterministic)
    deterministic_capacity=float(teachers[0]@kernel@teachers[0])
    assert float(deterministic_cov.abs().max())==0 and deterministic_capacity>0
    route_points=torch.randn(3,7,dtype=dtype)
    probs=torch.tensor([.2,.3,.5],dtype=dtype)
    duplicated=torch.cat((route_points[:1],route_points[:1],route_points[1:]),0)
    duplication_error=float((covariance(route_points,probs)-covariance(duplicated,torch.tensor([.1,.1,.3,.5],dtype=dtype))).abs().max())
    regularized=gramian+torch.eye(7,dtype=dtype)*.3
    direction=torch.randn(7,dtype=dtype); weight=.7
    delta=torch.linalg.slogdet(regularized+weight*direction[:,None]*direction[None,:])[1]-torch.linalg.slogdet(regularized)[1]
    identity=torch.log1p(weight*direction@torch.linalg.solve(regularized,direction))
    determinant_error=float((delta-identity).abs())
    rot=torch.tensor([[1.,1.],[-1.,1.]],dtype=dtype)/2**.5
    correlated=torch.ones(2,2,dtype=dtype); anticorrelated=torch.tensor([[1.,-1.],[-1.,1.]],dtype=dtype)
    first=(rot@correlated@rot.T).diag(); second=(rot@anticorrelated@rot.T).diag()
    assert torch.allclose(first,torch.tensor([2.,0.],dtype=dtype),atol=1e-14)
    assert torch.allclose(second,torch.tensor([0.,2.],dtype=dtype),atol=1e-14)
    read=torch.tensor([[1.,0.]],dtype=dtype); tangent=torch.tensor([[1.],[0.]],dtype=dtype)
    suffix=torch.tensor([[0.,-1.],[1.,0.]],dtype=dtype)
    assert tangent.norm()==(suffix@tangent).norm()
    observed=[float((read@tangent).square().sum()),float((read@suffix@tangent).square().sum())]
    assert observed==[1.,0.]
    assert max(expectation_error,covariance_error,duplication_error,determinant_error)<1e-12
    result={'status':'completed','mean_plus_covariance_expectation_error':expectation_error,
       'categorical_teacher_covariance_error':covariance_error,
       'deterministic_label_covariance_zero':True,'deterministic_label_mean_correction_capacity':deterministic_capacity,
       'duplicate_route_covariance_error':duplication_error,'logdet_rank_one_identity_error':determinant_error,
       'same_input_scalar_variances_opposite_output_variances':[first.tolist(),second.tolist()],
       'norm_preserved_but_readout_visible_capacity':observed,
       'source_sha256':{str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
       'scope':'Deterministic algebraic checks, not an optionality learning experiment or SHD accuracy claim.'}
    out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
