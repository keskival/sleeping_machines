"""Research contract for exact paired-view categorical risk and head geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sleeping_machines.nuisance_readout import paired_view_metric, paired_logit_risk


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e153')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(153);started=time.perf_counter()
    clean=torch.randn(100,8,dtype=torch.double)
    changed=clean+torch.randn_like(clean)*torch.linspace(.01,2.,8,dtype=torch.double)
    mean,metric,geometry=paired_view_metric(clean,changed)
    w=torch.randn(8,5,dtype=torch.double);b=torch.randn(5,dtype=torch.double)
    labels=torch.randint(5,(100,))
    risk=paired_logit_risk(clean@w+b,changed@w+b,labels)
    assert risk['identity_max_error']<1e-13 and risk['maximum_bound_violation']<1e-13
    assert geometry['covariance_decomposition_max_error']<1e-13
    both=torch.cat((clean,changed));x=both-mean
    delta=changed-clean;within=delta.T@delta/(4*len(clean))
    covariance=x.T@x/len(x)
    response=metric@w
    folded=both@response+b-mean@response
    white=(both-mean)@metric@w+b
    fold_error=float((folded-white).abs().max());assert fold_error<1e-13
    inverse=torch.linalg.solve(metric,response)
    penalty=float(torch.trace(response.T@(covariance+within+
        geometry['absolute_ridge']*torch.eye(8,dtype=torch.double))@response))
    penalty_error=abs(penalty-float(inverse.square().sum()));assert penalty_error<1e-12
    result=dict(status='completed',risk=risk,geometry=geometry,
        coordinate_fold_max_error=fold_error,regularizer_identity_error=penalty_error,
        scope='Exact paired-view CE/KL, covariance decomposition and folded-head regularizer; no SHD score implied',
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('sleeping_machines/nuisance_readout.py')]})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
