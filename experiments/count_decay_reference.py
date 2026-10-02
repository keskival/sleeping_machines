"""Elapsed-time-decayed in-stream counts (THEORY Corollary 377.3, §387): exploratory grid on the shared dev window.

Static Witten-Bell counts of text8[0:N] plus prequential development-stream counts weighted by beta and decayed by
exp(-elapsed/tau) characters (tau=0 means no decay). Grid values are dev-selected: exploratory, not a benchmark.
"""
import json, platform, time
from pathlib import Path
import sys, math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import count_reference_language as C
dev=C.text_slice(C.DEV_START,C.DEV_LEN).tolist()
def run(N,K,beta,tau):
    fit=C.text_slice(0,N).tolist(); T=C.Counts(K); T.add_sequence(fit)
    stat=T.t; dyn=[{} for _ in range(K+1)]  # ctx -> (vector, last_time)
    tot=0.
    for i in range(1,len(dev)):
        hist=dev[max(0,i-K):i]; p=np.full(27,1/27)
        for k in range(min(K,len(hist))+1):
            h=C.ctx_code(hist,k); c=stat[k].get(h); c=np.zeros(27) if c is None else c.copy()
            d=dyn[k].get(h)
            if d is not None:
                v,t0=d; f=1.0 if tau==0 else math.exp(-(i-t0)/tau); c=c+beta*v*f
            n=c.sum()
            if n<=0: continue
            Tn=(c>0).sum(); p=(c+Tn*p)/(n+Tn)
        tot-=math.log2(p[dev[i]])
        for k in range(min(K,len(hist))+1):  # observe after scoring
            h=C.ctx_code(hist,k); d=dyn[k].get(h)
            if d is None: v=np.zeros(27)
            else:
                v,t0=d; v=v*(1.0 if tau==0 else math.exp(-(i-t0)/tau))
            v=v.copy(); v[dev[i]]+=1; dyn[k][h]=(v,i)
    return tot/(len(dev)-1)
def main():
    out=Path(sys.argv[1])
    if out.exists(): raise SystemExit('refusing to overwrite')
    rows=[];t0=time.time()
    for N,K in ((8192,3),(1048576,5)):
        for beta in (1,4,16):
            for tau in (0,4096,1024,256):
                rows.append(dict(fit=N,order=K,beta=beta,tau=tau,bpc=run(N,K,beta,tau)));print(json.dumps(rows[-1]),flush=True)
    out.write_text(json.dumps(dict(status='completed',kind='count_decay_reference',rows=rows,wall_seconds=time.time()-t0,
        scope='Exploratory dev-selected grid; 8,191 shared development targets; Witten-Bell interpolation.',
        hardware=dict(device='cpu',threads=1,platform=platform.platform())),indent=1))


if __name__=='__main__':
    main()
