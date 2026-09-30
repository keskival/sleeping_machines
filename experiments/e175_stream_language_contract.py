"""Persistent-state/chunk/cutoff and credit contract; no benchmark claim."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.stream_language import StreamingEventLanguageModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e175')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(175);started=time.perf_counter()
    model=StreamingEventLanguageModel(vocabulary=7,width=8,modes=4,depth=8).double()
    sequence=torch.tensor([0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1])
    model.eval()
    full,state=model.forward_chunk(sequence)
    split,s=model.forward_chunk(sequence[:5]);rest,s=model.forward_chunk(sequence[5:],s)
    chunk_error=float((full-torch.cat((split,rest))).abs().max().detach())
    altered=sequence.clone();altered[7:]=(altered[7:]+2)%7
    changed,_=model.forward_chunk(altered)
    prefix_error=float((full[:7]-changed[:7]).abs().max().detach())
    detached=s.detach()
    assert not any(x.requires_grad for x in detached.modes)
    model.train();model.zero_grad(set_to_none=True)
    logits,state=model.forward_chunk(sequence[:-1])
    loss=F.cross_entropy(logits,sequence[1:]);loss.backward()
    gradient=[sum(float(q.grad.square().sum()) for q in layer.parameters() if q.grad is not None)**.5
              for layer in model.layers]
    clock_gradient=[0. if layer.clock.weight.grad is None else float(layer.clock.weight.grad.norm())
                    for layer in model.layers]
    assert chunk_error<1e-12 and prefix_error<1e-12 and all(g>0 for g in gradient)
    assert state.deliveries==len(sequence[:-1])*model.depth
    result=dict(status='completed',chunk_split_max_logit_error=chunk_error,
        future_mutation_prefix_error=prefix_error,depth=model.depth,layer_gradient_norm=gradient,
        clock_gradient_norm=clock_gradient,loss=float(loss.detach()),source_tokens=len(sequence)-1,
        event_deliveries=state.deliveries,expected_deliveries=(len(sequence)-1)*model.depth,
        pending_messages=len(state.pending),detach_preserves_position=detached.position==s.position,
        scope='Synthetic local contract only. Persistent signed states and event queue; no prefix replay. Future token mutations do not change earlier predictions. Fixed token-interval serialization, dense local maps, no trained accuracy/scaling claim. Final clock is unobserved when it arrives before the query deadline.',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path('sleeping_machines/stream_language.py'),Path('sleeping_machines/event_state.py'))},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
