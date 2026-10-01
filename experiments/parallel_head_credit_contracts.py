"""Guarded longer-credit contracts; optimizer steps require the host lock."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from parallel_head_race_language_screen import contracts,source_hashes
from parallel_head_gradient_accumulation import GradientAccumulator
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel


def credit_contracts(payload,depth,pool,heads,credit):
    with torch.random.fork_rng():
        torch.manual_seed(117)
        model=ParallelHeadRaceLanguageModel(payload,depth,pool,heads=heads)
        reference=copy.deepcopy(model)
        tokens=torch.tensor(([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3]*5)[:credit+1])
        opt=torch.optim.Adam(model.parameters(),lr=.002)
        learner=GradientAccumulator(model,opt,.002)
        rng=torch.get_rng_state()
        _,state,z=learner.accumulate(tokens[:-1],tokens[1:],model.new_state())
        assert learner.pending_targets==credit
        assert state.packed_storage()['differentiable_entries']==0
        torch.set_rng_state(rng)
        logits,other=reference.forward_chunk(tokens[:-1],reference.new_state())
        F.cross_entropy(logits,tokens[1:],reduction='sum').backward()
        torch.testing.assert_close(z,logits.detach(),rtol=0,atol=0)
        learner.normalize()
        for p,q in zip(model.parameters(),reference.parameters()):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad/credit,rtol=2e-6,atol=2e-7)
        # Gradient normalization above is verified once, then apply the same
        # clipping/Adam step manually to the explicit whole-credit reference.
        for q in reference.parameters():
            if q.grad is not None:q.grad.div_(credit)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);learner.step()
        ropt=torch.optim.Adam(reference.parameters(),lr=.002)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True);ropt.step()
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        # Lossless detach affects gradients, never forward content/races/RNG.
        model.eval();torch.manual_seed(37)
        with torch.no_grad():whole,_=model.forward_chunk(tokens[:-1])
        torch.manual_seed(37);parts=[];state=model.new_state()
        with torch.no_grad():
            for start in range(0,credit,16):
                part,state=model.forward_chunk(tokens[start:min(start+16,credit)],state)
                parts.append(part);state=state.detach()
        torch.testing.assert_close(whole,torch.cat(parts),rtol=0,atol=0)
        return dict(declared_credit_targets=credit,whole_credit_gradient_and_update_equal=True,
            forward_independent_of_detach_partition=True,longer_credit_detached_before_optimizer=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--credit',type=int,choices=(32,64),default=64);p.add_argument('--payload',type=int,default=32)
    p.add_argument('--depth',type=int,default=8);p.add_argument('--pool',type=int,default=2);p.add_argument('--heads',type=int,default=2)
    a=p.parse_args();out=ROOT/'experiments/results/episodic_language'/f'{a.tag}.json'
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Use unused tag')
    started=time.perf_counter();torch.set_num_threads(1)
    checks={**contracts(a.payload,a.depth,a.pool,8,4,a.heads),**credit_contracts(a.payload,a.depth,a.pool,a.heads,a.credit)}
    sources=source_hashes()
    for f in ('experiments/parallel_head_credit_contracts.py','experiments/parallel_head_gradient_accumulation.py'):
        sources[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    result=dict(status='completed',args=vars(a),numerical_contracts=checks,source_sha256=sources,
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
