"""Research contract for staged head/hidden credit in a serial event suffix."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.serial_residual import SerialResidualEncoder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=Path('experiments/results/e168')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(168);started=time.perf_counter()
    prefix=CoalescedEventStateEncoder(sources=16,width=8,modes=4,depth=6,classes=3).double().eval()
    model=SerialResidualEncoder(prefix,suffix_depth=6).train()
    state={name:q.detach().clone() for name,q in model.prefix.state_dict().items()}
    source=torch.arange(24)%16
    raw_t=torch.tensor([.002,.008,.013,.019,.023,.029,.032,.038,.043,.049,.052,.058]*2,dtype=torch.double)
    assignment=torch.arange(12).repeat_interleave(2)
    times=torch.tensor([.01,.02,.03,.04,.05,.06]*2,dtype=torch.double)
    receivers=torch.arange(2).repeat_interleave(6);counts=torch.ones(12,dtype=torch.double)*2
    data=(source,raw_t,assignment,times,receivers,counts,2)
    old=prefix(*data)[0];score,_,_,stats=model(*data)
    error=float((old-score).detach().abs().max())
    labels=torch.tensor([0,2]);loss=F.cross_entropy(score,labels);loss.backward()
    head_gradient=float(model.suffix.head.weight.grad.norm())
    initial_hidden=[float(torch.sqrt(sum((q.grad.square().sum() for q in layer.parameters() if q.grad is not None),score.new_tensor(0.))))
        for layer in model.suffix.layers]
    if error!=0 or head_gradient<=0 or max(initial_hidden)!=0:raise ValueError('Initial ownership failure')
    with torch.no_grad():
        model.suffix.head.weight.add_(model.suffix.head.weight.grad,alpha=-.001)
        model.suffix.head.bias.add_(model.suffix.head.bias.grad,alpha=-.001)
    model.zero_grad(set_to_none=True);after,_,_,_=model(*data);F.cross_entropy(after,labels).backward()
    live=[float(torch.sqrt(sum((q.grad.square().sum() for q in layer.parameters() if q.grad is not None),score.new_tensor(0.))))
        for layer in model.suffix.layers]
    if min(live)<=0:raise ValueError('Suffix credit did not become observable')
    if any(q.grad is not None for q in model.prefix.parameters()):raise ValueError('Prefix received forbidden update credit')
    if any(not torch.equal(q,state[name]) for name,q in model.prefix.state_dict().items()):raise ValueError('Changed prefix')
    result=dict(status='completed',initial_logit_error=error,total_depth=stats['total_depth'],
        initial_head_gradient_norm=head_gradient,initial_hidden_gradient_norms=initial_hidden,
        post_head_update_hidden_gradient_norms=live,prefix_state_unchanged=True,
        initial_loss=float(loss.detach()),post_head_update_loss=float(F.cross_entropy(after,labels).detach()),
        scope='Exact baseline and staged head/six-block hidden teacher support with immutable prefix; not an accuracy/global convergence result',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            (Path(__file__),Path('sleeping_machines/serial_residual.py'),Path('sleeping_machines/event_state.py'))},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
