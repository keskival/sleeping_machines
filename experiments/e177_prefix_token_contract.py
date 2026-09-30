"""Numerical causality/likelihood contracts for complete prefix tokenization."""
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
from sleeping_machines.prefix_tokenizer import (CompletePrefixTokenizer,
    ByteTimedTokenModel,new_prefix_state,observe_character)
from sleeping_machines.stream_language import StreamingEventLanguageModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e177')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(6);start=time.perf_counter()
    raw=[0,0,1,0,2,1,0,1,0,0,2,0,1,0,2,1]*3
    codec=CompletePrefixTokenizer.fit(raw[:24],alphabet=3,expansions=3)
    ids,tail=codec.encode(raw);assert codec.decode(ids)+tail==tuple(raw)
    lp=F.log_softmax(torch.randn(len(codec.phrases),dtype=torch.float64),0)
    normalization=0.;telescoping=0.
    for prefix in codec.descendants:
        if prefix in codec.leaves:continue
        children=torch.stack([codec.log_mass(lp,prefix+(c,))-codec.log_mass(lp,prefix)
                              for c in range(codec.alphabet)])
        normalization=max(normalization,abs(float(children.exp().sum())-1))
    for phrase,index in codec.leaves.items():
        logp=lp.new_zeros(())
        for k in range(len(phrase)):
            logp+=codec.log_mass(lp,phrase[:k+1])-codec.log_mass(lp,phrase[:k])
        telescoping=max(telescoping,abs(float(logp-lp[index])))
    model=ByteTimedTokenModel(vocabulary=len(codec.phrases),width=8,modes=4,depth=8).double()
    def scores(chars,chunks=None):
        state=new_prefix_state(model,codec);rows=[]
        for i,c in enumerate(chars):
            rows.append(observe_character(model,codec,state,c))
            if chunks and i in chunks:state=state.detach()
        return torch.stack(rows),state
    with torch.no_grad():
        base,state=scores(raw)
        split,splitstate=scores(raw,{4,17,31})
        changed,_=scores(raw[:21]+[(c+1)%3 for c in raw[21:]])
    chunk_error=float((base-split).abs().max());causal_error=float((base[:21]-changed[:21]).abs().max())
    value,_=scores(raw);value[8:].mean().backward()
    teachers=[sum(float(q.grad.square().sum()) for q in layer.parameters() if q.grad is not None)**.5
              for layer in model.layers]
    # A one-character leaf dictionary is exactly the original character stream.
    chars=CompletePrefixTokenizer.fit(raw[:24],alphabet=3,expansions=0)
    original=StreamingEventLanguageModel(vocabulary=3,width=8,modes=4,depth=8).double()
    timed=ByteTimedTokenModel(vocabulary=3,width=8,modes=4,depth=8).double()
    timed.load_state_dict(original.state_dict())
    with torch.no_grad():
        old=original.new_state();new=new_prefix_state(timed,chars);equivalence=0.
        previous=None
        for c in raw:
            score=observe_character(timed,chars,new,c)
            if previous is not None:
                target=torch.tensor([c]);reference=F.cross_entropy(previous[None],target)
                equivalence=max(equivalence,abs(float(score-reference)))
            previous=original.consume(c,old)
    assert normalization<1e-12 and telescoping<1e-12
    assert chunk_error==0 and causal_error==0 and equivalence<1e-12
    assert min(teachers)>0 and state.event.position==len(ids)
    result=dict(status='completed',normalization_error=normalization,
        token_score_telescoping_error=telescoping,chunk_error=chunk_error,
        future_suffix_error=causal_error,character_stream_equivalence_error=equivalence,
        layer_teacher_norms=teachers,raw_characters=len(raw),released_tokens=len(ids),
        unfinished_prefix=list(tail),layer_deliveries=state.event.deliveries,
        phrases=[list(p) for p in codec.phrases],alphabet=3,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
            Path(__file__),Path('sleeping_machines/prefix_tokenizer.py'),Path('sleeping_machines/stream_language.py'))},
        scope='Synthetic float64 contract, not language accuracy or optimal tokenization.',
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
