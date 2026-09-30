"""Rescore saved E64 controls on exactly E173's target positions, including tail.

No training, checkpoint selection or parameter changes. Cold stream, position
1 through n-1 inclusive. LSTM retains causal state; Transformer uses its
original half-context protocol and explicitly scores the omitted final tail.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e64_lm_baselines import LSTMLM, TfLM
from e120_shared_tasks import text_slice


@torch.no_grad()
def score(model,tokens,kind,context,batch_size):
    model.eval();total=0.;count=0
    if kind=='lstm':
        state=None
        for start in range(0,len(tokens)-1,4096):
            end=min(start+4096,len(tokens)-1)
            logits,state=model(tokens[start:end][None],state)
            loss=F.cross_entropy(logits.flatten(0,1),tokens[start+1:end+1],reduction='sum')
            total+=float(loss);count+=end-start
    else:
        if len(tokens)<=context:starts=[0]
        else:
            starts=list(range(0,len(tokens)-context,context//2))
            final=len(tokens)-context-1
            if starts[-1]!=final:starts.append(final)
        done=1
        for offset in range(0,len(starts),batch_size):
            selected=starts[offset:offset+batch_size]
            length=min(context,len(tokens)-1)
            x=torch.stack([tokens[s:s+length] for s in selected])
            y=torch.stack([tokens[s+1:s+length+1] for s in selected])
            logits,_=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape(len(selected),length)
            for i,s in enumerate(selected):
                lower=max(done,s+1)-(s+1);upper=min(s+length+1,len(tokens))-(s+1)
                if upper>lower:
                    total+=float(loss[i,lower:upper].sum());count+=upper-lower;done=s+upper+1
    if count!=len(tokens)-1:raise AssertionError((count,len(tokens)-1))
    return dict(test_bpc=total/count/math.log(2),test_n=count)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--model',choices=('lstm','tf'),required=True)
    p.add_argument('--n',type=int,default=1000000);p.add_argument('--batch',type=int,default=4)
    a=p.parse_args();out=Path('experiments/results/e174')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);start=time.perf_counter()
    name='lstm_D10000000_s512_p6_dr0.1_v' if a.model=='lstm' else 'tf_D10000000_s256_L4_p4_dr0.1_v'
    checkpoint=Path('experiments/results/e64')/(name+'.pt')
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu');cfg=saved['args']
    net=LSTMLM(cfg['size'],cfg['dropout']) if a.model=='lstm' else TfLM(cfg['size'],cfg['layers'],cfg['ctx'],cfg['dropout'])
    net.load_state_dict(saved['state']);tokens=torch.tensor(text_slice(95000000,a.n))
    result=score(net,tokens,a.model,cfg['ctx'],a.batch)
    if any(not torch.equal(net.state_dict()[k],v) for k,v in saved['state'].items()):
        raise AssertionError('Evaluation changed checkpoint')
    result.update(status='completed',args=vars(a),inherited_args=cfg,checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path('experiments/e64_lm_baselines.py'))},
        data_sha256=hashlib.sha256(tokens.numpy().astype('uint8').tobytes()).hexdigest(),
        protocol=dict(test=[95000000,95000000+a.n],scored_positions=[1,a.n],cold_scored_stream_context=True),
        scope='Saved validation-selected E64 model, no retraining or test-driven selection. Exact E173 target population; Transformer final tail included; training capacities/schedules remain unmatched.',
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
