"""Corrected, bounded-memory native language rerun with explicit expert arms.

Counts are rebuilt from training-only characters. Word keys depend strictly
on characters before the target. Copy is a bounded causal cache. Both arms
select mixing rate on validation and freeze it before test. This is a native
count/copy mixture, not a generic neural language model or matched-capacity
scaling experiment. Never overwrite E79's historical evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import e62_charlm as S1
import e79_race_mixer as E79
from e63_mixlm import COPY_L
from e120_shared_tasks import text_slice
from sleeping_machines.causal_contexts import word_codes_before


class CausalWordOrder:
    def __init__(self,train):
        code=word_codes_before(train)
        pair=code*27+train
        self.pairs,self.cnt=np.unique(pair,return_counts=True)
        self.ctx,first=np.unique(self.pairs//27,return_index=True)
        self.n=np.add.reduceat(self.cnt,first)


def bounded_copy(stream,window=256):
    raw=np.asarray(stream,dtype=np.uint8).tobytes()
    last={length:{} for length in COPY_L}
    predicted=np.full(len(raw),-1,np.int64);match=np.zeros(len(raw),np.int64)
    for i in range(len(raw)):
        for li in range(len(COPY_L)-1,-1,-1):
            length=COPY_L[li]
            if i>=length:
                previous=last[length].get(raw[i-length:i])
                if previous is not None and i-previous<=window:
                    predicted[i]=raw[previous];match[i]=li+1;break
        for length in COPY_L:
            if i>=length:last[length][raw[i-length:i]]=i
            expired=i-window
            if expired>=length:
                key=raw[expired-length:expired]
                if last[length].get(key)==expired:del last[length][key]
    return predicted,match


def batches(train,stream,orders,word,K,chunk):
    # Cold test/validation context, matching baseline observation at the first token.
    codes=[S1.ctx_codes(stream,k) for k in range(K+1)]
    word_codes=word_codes_before(stream)
    pred,match=bounded_copy(stream)
    selector=match*27+np.r_[0,stream[:-1]]
    for start in range(0,len(stream),chunk):
        stop=min(start+chunk,len(stream));p=np.full((stop-start,27),1/27)
        columns=[]
        for k,order in enumerate(orders):
            key=codes[k][start:stop];counts,total=E79.lookup(order,key)
            columns.append(np.log((counts+.5)/(total[:,None]+13.5)))
            index=np.searchsorted(order.ctx,key);clipped=np.clip(index,0,len(order.ctx)-1)
            u=np.where((key>=0)&(order.ctx[clipped]==key),order.u[clipped],0)
            p=np.where((total>0)[:,None],(counts+u[:,None]*p)/np.maximum(total+u,1e-9)[:,None],p)
        columns.append(np.log(np.maximum(p,1e-9)))
        counts,total=E79.lookup(word,word_codes[start:stop])
        columns.append(np.log((counts+.5)/(total[:,None]+13.5)))
        copy=np.full((stop-start,27),np.log(1/27),np.float32)
        actual=pred[start:stop];has=actual>=0
        copy[has]=np.log(.05/26);copy[np.flatnonzero(has),actual[has]]=np.log(.95)
        columns.append(copy)
        yield start, np.stack(columns,1).astype(np.float32),selector[start:stop]


def probe(orders,word,train,K):
    stream=np.tile(np.array([1,2,3,0,4,5,6,0]),8)
    baseline=next(batches(train,stream,orders,word,K,len(stream)))[1]
    error=0.
    for t in range(len(stream)):
        changed=stream.copy();changed[t:]=(changed[t:]+1)%27
        alternative=next(batches(train,changed,orders,word,K,len(stream)))[1]
        error=max(error,float(np.max(np.abs(baseline[t]-alternative[t]))))
    if error!=0:raise AssertionError('Corrected expert still sees current/future target')
    return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--D',type=int,default=10000000);p.add_argument('--K',type=int,default=6)
    p.add_argument('--validation',type=int,default=1000000);p.add_argument('--test',type=int,default=1000000)
    p.add_argument('--chunk',type=int,default=20000);a=p.parse_args()
    out=Path('experiments/results/e173')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag or not 0<a.D<=90000000 or a.K>10:
        raise ValueError('Unique output, reserved training split and safe context required')
    start=time.perf_counter();train=text_slice(0,a.D)
    valid=text_slice(90000000,a.validation);test=text_slice(95000000,a.test)
    result=dict(status='running',args=vars(a),protocol=dict(training=[0,a.D],
        validation=[90000000,90000000+a.validation],test=[95000000,95000000+a.test],
        scored_test_positions=[1,a.test],copy_window=256,cold_scored_stream_context=True,
        validation_fits_mixing_weights=True,weight_updates_on_test=False,
        label='Native context-count/copy mixture; optional causal partial-word expert; no generic learned backbone'),
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
            Path(__file__),Path('experiments/e62_charlm.py'),Path('experiments/e79_race_mixer.py'),
            Path('experiments/e120_shared_tasks.py'),Path('sleeping_machines/causal_contexts.py'))},
        data_sha256={name:hashlib.sha256(value.astype(np.uint8).tobytes()).hexdigest()
                     for name,value in [('train',train),('validation',valid),('test',test)]},
        hardware=dict(platform=platform.platform(),device='cpu',threads=1),energy_joules=None)
    def persist():
        result.update(wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary=out.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out)
    persist();orders=[]
    for k in range(a.K+1):
        orders.append(S1.Order(train,k));print(json.dumps({'built_order':k,'contexts':len(orders[-1].ctx),'rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
    word=CausalWordOrder(train)
    result['causality_max_logp_change']=probe(orders,word,train,a.K)
    result['stored_count_contexts']=sum(len(o.ctx) for o in orders)
    result['stored_word_contexts']=len(word.ctx)
    result['count_numeric_bytes']=sum(v.nbytes for o in orders for v in (o.pairs,o.cnt,o.ctx,o.n,o.u))
    result['word_numeric_bytes']=sum(v.nbytes for v in (word.pairs,word.cnt,word.ctx,word.n))
    rates=(.002,.01,.05);arms=('without_word','with_causal_word')
    weights={arm:{rate:np.full(((len(COPY_L)+1)*27,a.K+3+(arm=='with_causal_word')),1/(a.K+3+(arm=='with_causal_word')),np.float32)
                  for rate in rates} for arm in arms}
    losses={arm:{rate:0. for rate in rates} for arm in arms}
    for offset,lp,selectors in batches(train,valid,orders,word,a.K,a.chunk):
        y=valid[offset:offset+len(lp)]
        for arm in arms:
            experts=lp if arm=='with_causal_word' else lp[:,[j for j in range(lp.shape[1]) if j!=a.K+2]]
            for rate in rates:
                loss,weights[arm][rate]=E79.race_mix(experts,y,selectors,rate,weights[arm][rate],True)
                losses[arm][rate]+=float(loss.sum())
        result['validation_progress']=offset+len(lp);persist()
        if (offset+len(lp))%100000==0:print(json.dumps({'validation_chars':offset+len(lp)}),flush=True)
    chosen={arm:min(rates,key=lambda r:losses[arm][r]) for arm in arms}
    result['arms']={arm:dict(selected_learning_rate=chosen[arm],validation_bpc=losses[arm][chosen[arm]]/len(valid),
        validation_grid={str(rate):losses[arm][rate]/len(valid) for rate in rates},
        experts=[*[f'KT{k}' for k in range(a.K+1)],'Witten-Bell',*(['causal partial-word'] if arm=='with_causal_word' else []),'copy256']) for arm in arms}
    totals={arm:0. for arm in arms};counts={arm:0 for arm in arms}
    for offset,lp,selectors in batches(train,test,orders,word,a.K,a.chunk):
        y=test[offset:offset+len(lp)]
        for arm in arms:
            experts=lp if arm=='with_causal_word' else lp[:,[j for j in range(lp.shape[1]) if j!=a.K+2]]
            loss,_=E79.race_mix(experts,y,selectors,chosen[arm],weights[arm][chosen[arm]],False)
            keep=np.arange(offset,offset+len(lp))>=1
            totals[arm]+=float(loss[keep].sum());counts[arm]+=int(keep.sum())
        result['test_progress']=offset+len(lp);persist()
    for arm in arms:result['arms'][arm].update(test_bpc=totals[arm]/counts[arm],test_n=counts[arm])
    result.update(status='completed',scope='Corrected one-run native mixture; explicit two expert arms, selected independently on validation. Additional 1M validation labels fit mixture weights. Not matched capacity/optimization or a generic architecture superiority claim.')
    np.savez(out.with_suffix('.weights.npz'),**{arm:weights[arm][chosen[arm]] for arm in arms})
    persist();print(json.dumps(result['arms']),flush=True)


if __name__=='__main__':main()
