"""Numerical research contract for fitting-only affine branch absorption."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sleeping_machines.readout_absorption import project_component, optimize_head, absorption_certificate


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e149')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique output required')
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(149)
    h=torch.randn(240,12,dtype=torch.double)*torch.logspace(-2,1,12,dtype=torch.double)
    true_weight=torch.randn(12,5,dtype=torch.double);true_bias=torch.randn(5,dtype=torch.double)
    parent=h@true_weight+true_bias
    w,b,mean,metric,stats=project_component(h,parent,1e-10)
    folded=h@w+b
    error=float((folded-(parent-parent.mean(-1,keepdim=True))).abs().max())
    assert error<1e-5
    certificate=absorption_certificate(parent,folded)
    assert certificate['certified_decision_failures']==0 and certificate['changed_decisions']==0
    labels=parent.argmax(-1)
    weight,bias,fit=optimize_head(h,parent,labels,w.T,b,mean,metric,iterations=30)
    z=(h-mean)@metric
    coef=torch.linalg.solve(metric,weight.T)
    white_score=z@coef+bias+mean@weight.T
    fold_error=float((white_score-(h@weight.T+bias)).abs().max())
    assert fold_error<1e-12 and fit['objective_final']<fit['objective_initial']
    torch.manual_seed(150)
    random_teacher=torch.randn(400,9,dtype=torch.double)
    random_student=random_teacher+10*torch.randn_like(random_teacher)
    global_bound=absorption_certificate(random_teacher,random_student)
    assert global_bound['maximum_bound_violation']<1e-12
    result=dict(status='completed',affine_component_max_error=error,
        optimizer_coordinate_fold_max_error=fold_error,projection=stats,
        certificate=certificate,head_fit=fit,large_perturbation_bound=global_bound,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Synthetic affine-head inclusion and categorical bounds; not deep optimization or held-speaker accuracy',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('sleeping_machines/readout_absorption.py')]})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
