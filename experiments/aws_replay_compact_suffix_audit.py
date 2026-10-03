"""First guarded compact-suffix admission; not permission for a quality run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import torch
import causal_language_precision_audit as A
import aws_replay_compact_suffix as Compact


def evaluate(helper, model, tokens, targets, state, seed):
    model.zero_grad(set_to_none=True)
    objective, logits, following, activity=helper.batched_objective(model,tokens,targets,state,seed)
    objective.backward()
    return dict(gradients=A.C.grads(model), logits=logits.detach(), state=following.detach(), activity=activity)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--sources',required=True);args=p.parse_args()
    hashes=json.loads((ROOT/args.sources).read_text())
    for n,h in hashes.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    out=ROOT/'experiments/results/diagnostics'/(args.tag+'.json');assert not out.exists()
    torch.set_num_threads(1);start=time.perf_counter();rows=[]
    observed=torch.tensor([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3,2])
    for family in ('private','depth'):
        model=A.P.make(family);state=A.P.initial(model)
        for count in (1,3,16):
            for precision in ('float32','float64'):
                base=copy.deepcopy(model);entering=copy.deepcopy(state)
                if precision=='float64':base=base.double();entering=A.promote(entering)
                torch.manual_seed(116329);seed=torch.get_rng_state().clone()
                old=evaluate(A.New,copy.deepcopy(base),observed[:count],observed[1:count+1],copy.deepcopy(entering),seed)
                new=evaluate(Compact,copy.deepcopy(base),observed[:count],observed[1:count+1],copy.deepcopy(entering),seed)
                A.C.close(torch.get_rng_state(),seed,True)
                A.C.close(old['logits'],new['logits'],True);A.C.state_close(old['state'],new['state'],True)
                A.C.close(old['activity']['factual_end_rng'],new['activity']['factual_end_rng'],True)
                if precision=='float64':A.C.grads_close(old['gradients'],new['gradients'])
                assert old['activity']['shadow_lanes']==new['activity']['shadow_lanes']
                expected=model.depth*model.heads*(model.pool-1)*count*(count+1)//2
                assert new['activity']['shadow_events']==expected
                rows.append(dict(family=family,targets=count,precision=precision,
                    gradient_errors=A.errors(new['gradients'],old['gradients']),
                    old_shadow_events=old['activity']['shadow_events'],compact_shadow_events=expected))
    result=dict(status='completed',rows=rows,source_sha256=hashes,wall_s=time.perf_counter()-start,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='First gradient/state/RNG/event-count audit only. All shadow route histories, optimizer recovery, traced accounting, wall/RSS and learning admission still required. No quality run authorized by this result. Diagnostic work unknown, not zero.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',cases=len(rows))))


if __name__=='__main__':main()
