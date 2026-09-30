"""Small evidence-free persistent language run; trained tokens consumed once.

Targets match E133 byte offsets; this model keeps all causal state and truncates
credit every chunk. Different model and update count: not a matched ablation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e120_shared_tasks import text_slice
from sleeping_machines.stream_language import StreamingEventLanguageModel


@torch.no_grad()
def evaluate(model,tokens,warm=31):
    model.eval();state=model.new_state()
    for token in tokens[:warm]:model.consume(token,state)
    total=0.;count=0
    for i in range(warm,len(tokens)-1):
        z=model.consume(tokens[i],state)
        total+=float(F.cross_entropy(z[None],tokens[i+1:i+2]));count+=1
    return dict(n=count,nll=total/count,bpc=total/count/math.log(2),
                consumed_tokens=state.position,event_deliveries=state.deliveries)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--depth',type=int,default=8);p.add_argument('--fit',type=int,default=8192)
    p.add_argument('--dev',type=int,default=1024);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--chunk',type=int,default=64);a=p.parse_args()
    out=Path('experiments/results/e176')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    train=torch.tensor(text_slice(0,a.fit+32));dev=torch.tensor(text_slice(90000000,a.dev+32))
    model=StreamingEventLanguageModel(depth=a.depth);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    result=dict(status='running',args=vars(a),curve=[],parameters=sum(p.numel() for p in model.parameters()),
        initial=evaluate(model,dev),protocol=dict(training_targets=[32,a.fit+32],
        development_targets=[90000032,90000000+a.dev+32],official_test_read=False,
        tokenizer='fixed 27-character text8 alphabet',experts=[],warm_tokens=31,
        state_reset='once at each fitting pass and each evaluation stream',
        credit_truncation=a.chunk,stream_time_unit='one token interval',query_budget=.5),
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
            Path(__file__),Path('sleeping_machines/stream_language.py'),Path('sleeping_machines/event_state.py'))},
        scope='Fresh generic signed event-state model, no count/copy/word evidence. Persistent stream, no prefix replay. Same target byte ranges as E133, different topology/parameters/update budget, broader causal history. Small one-seed development study.',
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1,device='cpu'),energy_joules=None)
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary=out.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out)
    persist();print(json.dumps({'initial':result['initial']}),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();state=model.new_state()
        with torch.no_grad():
            for token in train[:31]:model.consume(token,state)
        total=0.;steps=0;teacher=[0.]*a.depth;events=0
        for start in range(31,len(train)-1,a.chunk):
            end=min(start+a.chunk,len(train)-1)
            optimizer.zero_grad(set_to_none=True)
            logits,state=model.forward_chunk(train[start:end],state)
            loss=F.cross_entropy(logits,train[start+1:end+1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite stream loss')
            loss.backward()
            for j,layer in enumerate(model.layers):
                teacher[j]+=sum(float(q.grad.square().sum()) for q in layer.parameters() if q.grad is not None)**.5
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*(end-start);steps+=1
            state=state.detach()
            if steps%32==0:
                result['progress']=dict(epoch=epoch,targets=end-31,online_bpc=total/(end-31)/math.log(2))
                persist();print(json.dumps(result['progress']),flush=True)
        events=state.deliveries
        row=dict(epoch=epoch,online_bpc=total/a.fit/math.log(2),optimizer_steps=steps,
            consumed_tokens=state.position,training_event_deliveries=events,
            layer_teacher_norm_sum=teacher,dev=evaluate(model,dev),fit=evaluate(model,train))
        result['curve'].append(row);result['final']=row;persist()
        torch.save(dict(state_dict=model.state_dict(),args=vars(a),optimizer=optimizer.state_dict()),
                   out.with_suffix(f'.epoch{epoch}.pt'))
        print(json.dumps({'epoch':epoch,'fit_bpc':row['fit']['bpc'],'dev_bpc':row['dev']['bpc']}),flush=True)
    result.update(status='completed');persist()


if __name__=='__main__':main()
