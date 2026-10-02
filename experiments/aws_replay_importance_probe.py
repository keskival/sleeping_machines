"""Training-only race priorities, positive-probability importance correction."""
import argparse
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N


def credit(pi,u): return pi*(u-(pi*u).sum(-1,keepdim=True))


def contract():
    vectors=torch.tensor([[1.,2.],[-3.,4.],[5.,-6.]],dtype=torch.float64)
    p=torch.tensor([.1,.3,.6],dtype=torch.float64);target=vectors.sum(0)
    for k in (1,2):
        mean=torch.zeros_like(target);mse=0.
        for draws in itertools.product(range(3),repeat=k):
            probability=float(p[list(draws)].prod())
            estimate=(vectors[list(draws)]/p[list(draws),None]).mean(0)
            mean+=probability*estimate;mse+=probability*float((estimate-target).square().sum())
        torch.testing.assert_close(mean,target,rtol=0,atol=1e-14)
        expected=float(((vectors.square().sum(1)/p).sum()-target.square().sum())/k)
        assert abs(mse-expected)<1e-12
    return 'all draws enumerate unbiased importance mean and shared-vector variance'


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    parent=ROOT/'experiments/results/diagnostics/aws_signed_replay_variance_20261002T230400Z.json';j=json.loads(parent.read_text())
    for n,h in j['source_sha256'].items():assert N.sha(ROOT/n)==h,n
    artifact=parent.with_suffix('.pt');assert N.sha(artifact)==j['artifact_sha256'];banks=torch.load(artifact,weights_only=False,map_location='cpu')
    rows=[]
    for seed in (7,8):
        examples=banks[str(seed)]['examples'];g=[credit(pi,u) for _,pi,u in examples]
        train_squared=torch.stack([x.square().sum(-1) for x in g[:32]]).mean(0)
        importance=train_squared.sqrt();importance/=importance.sum();importance=.9*importance+.1/len(importance)
        assert float(importance.min())>0 and abs(float(importance.sum())-1)<1e-12
        baseline=sum((len(x)/4-1)*float(x.square().sum()) for x in g[32:])
        def mse(x,proposal,k):
            # Score blocks are disjoint; the summed-gradient norm equals total block energy.
            v=x.square().sum(-1);return float(((v/proposal).sum()-v.sum())/k)
        ratios={str(k):sum(mse(x,importance,k) for x in g[32:])/baseline for k in (1,2)}
        uniform=torch.full_like(importance,1/len(importance))
        rows.append(dict(seed=seed,race_probabilities=importance.tolist(),training_site_energy=train_squared.tolist(),
            heldout_importance_k_vs_uniform_without_replacement_k4=ratios,
            heldout_uniform_with_replacement_ratios={str(k):sum(mse(x,uniform,k) for x in g[32:])/baseline for k in (1,2)},
            nomination_passed={k:v<=1 for k,v in ratios.items()},
            expected_unique_races_k2=float((1-(1-importance).square()).sum()),
            heldout_prefixes=[dict(index=32+i,energy=float(x.square().sum()),importance_k2_mse=mse(x,importance,2)) for i,x in enumerate(g[32:])]))
    out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists()
    result=dict(status='completed',tag=a.tag,contract=contract(),rows=rows,parent_sha256=N.sha(parent),artifact_sha256=N.sha(artifact),
        source_sha256={**j['source_sha256'],'experiments/aws_replay_importance_probe.py':N.sha(Path(__file__))},
        scope='Fixed ordinal race proposal fitted on32 FIT prefixes, next32 heldout; corrected importance law. Conditional score-space variance, no model update, parameter covariance or quality claim.',
        parent_costs='Original producer, full cached replay and critic work retained; no new replay or optimizer; arithmetic diagnostic FLOPs unmeasured',
        wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
