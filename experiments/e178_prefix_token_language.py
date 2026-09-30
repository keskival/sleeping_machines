"""Causal token compression versus E176, on identical raw-character targets.

This vocabulary control adds no prediction expert. A complete prefix tree gives
unique decoding and exact next-character marginal probabilities, even at the
end of an unfinished phrase. Both arms use raw-character arrival timestamps.
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
from sleeping_machines.prefix_tokenizer import (CompletePrefixTokenizer,
    ByteTimedTokenModel,new_prefix_state,observe_character)


def next_character_loss(codec,state,target):
    return state.log_mass-codec.log_mass(state.log_probability,state.prefix+(int(target),))


def refresh_head(model,codec,state):
    # A truncated boundary retains history; it starts a fresh head graph under
    # the current parameters, including when a phrase is unfinished.
    state.log_probability=F.log_softmax(model.head(state.event.output),0)
    state.log_mass=codec.log_mass(state.log_probability,state.prefix)


@torch.no_grad()
def evaluate(model,codec,tokens):
    model.eval();state=new_prefix_state(model,codec)
    for c in tokens[:31]:observe_character(model,codec,state,c)
    total=0.;count=0
    for i in range(31,len(tokens)-1):
        observe_character(model,codec,state,tokens[i])
        total+=float(next_character_loss(codec,state,tokens[i+1]));count+=1
    return dict(n=count,nll=total/count,bpc=total/count/math.log(2),
        consumed_characters=state.raw_position,released_tokens=state.event.position,
        layer_deliveries=state.event.deliveries,unfinished_prefix=list(state.prefix))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--fit',type=int,default=8192);p.add_argument('--dev',type=int,default=1024)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--expansions',type=int,default=4);a=p.parse_args()
    out=Path('experiments/results/e178')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(6);start=time.perf_counter()
    train=torch.tensor(text_slice(0,a.fit+32));dev=torch.tensor(text_slice(90000000,a.dev+32))
    codec=CompletePrefixTokenizer.fit(train.tolist(),expansions=a.expansions)
    character=ByteTimedTokenModel()
    if len(codec.phrases)==27:model=character
    else:
        model=ByteTimedTokenModel(vocabulary=len(codec.phrases))
        model.layers.load_state_dict(character.layers.state_dict())
    assert all(torch.equal(x,y) for x,y in zip(model.layers.parameters(),character.layers.parameters()))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    result=dict(status='running',args=vars(a),curve=[],initial=evaluate(model,codec,dev),
        vocabulary=[list(p) for p in codec.phrases],parameters=sum(p.numel() for p in model.parameters()),
        protocol=dict(fitting_targets=[32,a.fit+32],development_targets=[90000032,90000000+a.dev+32],
            tokenizer_fitting=[0,a.fit+32],tokenizer='frequency-prioritized complete 27-ary prefix dictionary',
            fitted_on_held_data=False,official_test_read=False,prediction_experts=[],
            credit_truncation_raw_characters=a.chunk,warm_characters=31,
            time_unit='raw-character arrival index; token released at final character',
            query_budget=.5,body_initialization_matches_e176=True),
        scope='One small generic tokenization control. Same raw targets/body initialization/depth/passes/credit window as E176; vocabulary/head capacities differ. No count/copy/word prediction expert. Exact character marginal likelihood including unfinished phrases; no energy or frontier claim.',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
            Path(__file__),Path('sleeping_machines/prefix_tokenizer.py'),Path('sleeping_machines/stream_language.py'))},
        data_sha256={name:hashlib.sha256(bytes(tokens.tolist())).hexdigest() for name,tokens in [('fit',train),('dev',dev)]},
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1,device='cpu'),energy_joules=None)
    def persist():
        result.update(wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary=out.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out)
    persist();print(json.dumps({'initial':result['initial'],'vocabulary':len(codec.phrases)}),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();state=new_prefix_state(model,codec)
        with torch.no_grad():
            for c in train[:31]:observe_character(model,codec,state,c)
        state=state.detach();total=0.;steps=0;teachers=[0.]*model.depth
        for begin in range(31,len(train)-1,a.chunk):
            end=min(begin+a.chunk,len(train)-1)
            optimizer.zero_grad(set_to_none=True);refresh_head(model,codec,state)
            losses=[]
            for i in range(begin,end):
                observe_character(model,codec,state,train[i])
                losses.append(next_character_loss(codec,state,train[i+1]))
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite tokenized loss')
            loss.backward()
            for j,layer in enumerate(model.layers):
                teachers[j]+=sum(float(q.grad.square().sum()) for q in layer.parameters() if q.grad is not None)**.5
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*(end-begin);steps+=1;state=state.detach()
            if steps%32==0:
                result['progress']=dict(epoch=epoch,targets=end-31,online_bpc=total/(end-31)/math.log(2))
                persist();print(json.dumps(result['progress']),flush=True)
        row=dict(epoch=epoch,online_bpc=total/a.fit/math.log(2),optimizer_steps=steps,
            consumed_characters=state.raw_position,released_tokens=state.event.position,
            training_layer_deliveries=state.event.deliveries,layer_teacher_norm_sum=teachers,
            fit=evaluate(model,codec,train),dev=evaluate(model,codec,dev))
        result['curve'].append(row);result['final']=row;persist()
        torch.save(dict(state_dict=model.state_dict(),args=vars(a),vocabulary=result['vocabulary'],
                        optimizer=optimizer.state_dict()),out.with_suffix(f'.epoch{epoch}.pt'))
        print(json.dumps({'epoch':epoch,'fit_bpc':row['fit']['bpc'],'dev_bpc':row['dev']['bpc']}),flush=True)
    result['status']='completed';persist()


if __name__=='__main__':main()
