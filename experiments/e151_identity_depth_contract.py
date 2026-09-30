"""Research contract: D6 -> D12 retains function and old teachers at growth."""
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
from sleeping_machines.depth_growth import grow_event_encoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e151')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(151);started=time.perf_counter()
    base=CoalescedEventStateEncoder(sources=16,width=8,modes=4,depth=6,classes=3).double().train()
    grown=grow_event_encoder(base,12)
    source=torch.arange(24)%16
    raw_t=torch.tensor([.002,.008,.013,.019,.023,.029,.032,.038,.043,.049,.052,.058]*2,dtype=torch.double)
    assignment=torch.arange(12).repeat_interleave(2)
    times=torch.tensor([.01,.02,.03,.04,.05,.06]*2,dtype=torch.double)
    receivers=torch.arange(2).repeat_interleave(6);counts=torch.ones(12,dtype=torch.double)*2
    data=(source,raw_t,assignment,times,receivers,counts,2)
    old,old_payload,old_times,_=base(*data)
    new,new_payload,new_times,_=grown(*data)
    function_error=float((old-new).abs().max());payload_error=float((old_payload-new_payload).abs().max())
    time_error=float((new_times-old_times-.036).abs().max())
    assert function_error<1e-14 and payload_error<1e-14 and time_error<1e-14
    labels=torch.tensor([0,2]);F.cross_entropy(old,labels).backward();F.cross_entropy(new,labels).backward()
    new_named=dict(grown.named_parameters());teacher_error=0.
    for name,param in base.named_parameters():
        other=new_named[name]
        if param.grad is None:
            assert other.grad is None or float(other.grad.abs().max())<1e-14
        else:teacher_error=max(teacher_error,float((param.grad-other.grad).abs().max()))
    assert teacher_error<1e-13
    new_teachers=[float(layer.output.weight.grad.norm()) for layer in grown.layers[6:]]
    assert min(new_teachers)>0
    # The local centered LayerNorm/gate derivative is 1/2 at this initialization.
    probe=grown.layers[6];x=torch.randn(5,8,dtype=torch.double);s=torch.randn(5,8,dtype=torch.double)
    t=torch.linspace(0,.4,5,dtype=torch.double);g=torch.randn_like(x)
    value,_,_=probe.emit(x,s,t,credit=False)
    actual=torch.autograd.grad((g*value).sum(),probe.output.weight)[0]
    explicit=probe.gain/2*(g-g.mean(-1,keepdim=True)).T@s
    local_teacher_error=float((actual-explicit).abs().max());assert local_teacher_error<1e-13
    result=dict(status='completed',function_max_error=function_error,payload_max_error=payload_error,
        extra_constant_delay_max_error=time_error,old_teacher_max_error=teacher_error,
        appended_output_gradient_norms=new_teachers,explicit_output_teacher_max_error=local_teacher_error,
        depth=12,initial_constant_delay_increment=.036,
        scope='Exact initialization and old/local teachers for a completed untimed query. Not a trained D12 SHD result or a global optimization theorem',
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path('sleeping_machines/depth_growth.py'),Path('sleeping_machines/event_state.py')]})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
