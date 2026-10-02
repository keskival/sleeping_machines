"""Actual gated native 64-credit gradient normalization and Adam recovery prerequisite."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from count_carrying_language_benchmark import sources
from parallel_head_gradient_accumulation import GradientAccumulator
from sleeping_machines.count_carrying_language import fit_stream_counts
from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel


def check(payload,depth):
    with torch.random.fork_rng():
        torch.manual_seed(917)
        tokens=torch.tensor(([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3]*17)[:257])
        model=GatedCountCarryingNativeModel(payload=payload,depth=depth,heads=2,pool=2,
            orders=4,count_message=False,escape_gate=True)
        model.register_stream('fit',fit_stream_counts(tokens.numpy(),4));model.use_stream('fit');model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        learner=GradientAccumulator(model,optimizer,.002,512)
        reference=copy.deepcopy(model)
        state=model.new_state();other=reference.new_state();rng=torch.get_rng_state()
        with torch.no_grad():
            torch.set_rng_state(rng);_,state=model.forward_chunk(tokens[:64],state)
            torch.set_rng_state(rng);_,other=reference.forward_chunk(tokens[:64],other)
        state.detach();other.detach();rng=torch.get_rng_state()
        torch.set_rng_state(rng)
        _,state,p=learner.accumulate(tokens[64:128],tokens[65:129],state)
        torch.set_rng_state(rng);z,other=reference.forward_chunk(tokens[64:128],other)
        F.cross_entropy(z,tokens[65:129],reduction='sum').backward();other.detach()
        torch.testing.assert_close(p,z.detach(),rtol=0,atol=0);learner.normalize()
        for a,b in zip(model.parameters(),reference.parameters()):
            assert (a.grad is None)==(b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad/64,rtol=2e-6,atol=2e-7)
                assert bool(torch.isfinite(a.grad).all())
        assert float(model.escape_gate.weight.grad.norm())>0
        for layer in model.queries:
            for query in layer: assert query.weight.grad is not None and float(query.weight.grad.norm())>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);learner.step()
        # Real serialized optimizer and persistent event/count cursor, after a complete window.
        memory=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            state=state,rng=torch.get_rng_state(),total=learner.total_targets,updates=learner.updates),memory)
        memory.seek(0);saved=torch.load(memory,weights_only=False)
        recovered=copy.deepcopy(model);recovered.load_state_dict(saved['model'])
        recovered_opt=torch.optim.Adam(recovered.parameters(),lr=.002);recovered_opt.load_state_dict(saved['optimizer'])
        recovered_learner=GradientAccumulator(recovered,recovered_opt,.002,512)
        recovered_learner.total_targets=saved['total'];recovered_learner.updates=saved['updates']
        replay=copy.deepcopy(saved['state']);assert replay.events==state.events==128
        outputs=[]
        for net,acc,live in ((model,learner,state),(recovered,recovered_learner,replay)):
            torch.set_rng_state(saved['rng']);_,live,q=acc.accumulate(tokens[128:192],tokens[129:193],live)
            outputs.append(q);acc.update();assert live.events==192 and acc.updates==2 and acc.total_targets==128
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
        for a,b in zip(model.parameters(),recovered.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Detached chunk partitioning must preserve the forward state, independently of credit.
        left,right=copy.deepcopy(model),copy.deepcopy(model);rng=torch.get_rng_state()
        with torch.no_grad():
            torch.set_rng_state(rng);full,_=left.forward_chunk(tokens[:64])
            torch.set_rng_state(rng);live=right.new_state();parts=[]
            for start in range(0,64,16):
                q,live=right.forward_chunk(tokens[start:start+16],live);live.detach();parts.append(q)
        torch.testing.assert_close(full,torch.cat(parts),rtol=0,atol=0)
        return dict(payload=payload,depth=depth,heads=2,pool=2,orders=4,escape_gate=True,count_message=False,
            chunk=64,update_targets=64,warmup_targets=512,normalized_gradient_matches_actual_sum=True,
            finite_gradients=True,all_head_query_and_escape_gradients=True,exact_serialized_next_update=True,
            exact_count_cursor_recovery=True,exact_forward_64_vs_four16=True,
            optimizer_updates_per_training_copy=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);started=time.perf_counter();variants=[check(16,8),check(2,1)]
    provenance=sources();name=str(Path(__file__).relative_to(ROOT));provenance[name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r=dict(status='completed',args=vars(a),variants=variants,source_sha256=provenance,
        hardware=dict(host=__import__('os').uname().nodename,device='cpu',threads=1,torch=torch.__version__),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Numerical training prerequisite for both actual gated native variants; not language quality or unbiased route-credit evidence.')
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(completed=a.tag,wall_s=r['wall_s'])),flush=True)


if __name__=='__main__':main()
