"""Native causal quarter-second count packets; unchanged addressed race core."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import dvs_batched_benchmark as B
import dvs_clock_calibrated_benchmark as C
BASE_LOAD=N.load

def parser():
    p=B.parser();p.add_argument('--bins',type=int,choices=(4,20),default=4)
    p.add_argument('--clock-step',type=float,default=.25);return p

def sources():
    names=['experiments/aws_coarse_native.py','experiments/aws_coarse_native_contracts.py','experiments/theory/aws_20261002_coarse_native_admission.md']
    return {**B.sources(),**C.sources(),**{n:N.sha(ROOT/n) for n in names}}

def coalesce(counts,bins):
    return counts.reshape(len(counts),bins,20//bins,32).sum(2)

def encode(counts,center,scale,bins):
    normalized=((np.log1p(counts.reshape(-1))-center)/scale).reshape(bins,32)
    events=[((i+1)/bins,np.r_[normalized[i],0.].astype(np.float32)) for i in range(bins)]
    events.append((1.,np.r_[np.zeros(32),1.].astype(np.float32)))
    return events

def load(a):
    started=time.perf_counter();fit,dev,info=BASE_LOAD(a)
    if a.bins==20:return fit,dev,info
    parent=json.loads((ROOT/a.data).read_text());z=np.load(ROOT/parent['data_artifact'],allow_pickle=False)
    counts=coalesce(z['fit_counts'],a.bins);logged=np.log1p(counts.reshape(len(counts),-1))
    center=logged.mean(0);scale=np.maximum(logged.std(0),.5)
    for split,rows in [('fit',fit),('dev',dev)]:
        compact=coalesce(z[split+'_counts'],a.bins)
        for i,row in enumerate(rows):row['events']=encode(compact[i],center,scale,a.bins)
    info=dict(info,temporal_bins=a.bins,packet_step=1/a.bins,
        feature_transform='fit-only log-count mean/std-clamp.5 after causal250ms coalescing',
        transform_sha256=hashlib.sha256(center.tobytes()+scale.tobytes()).hexdigest(),
        temporal_scope='Same raw counts and1s query; within250ms timing discarded')
    return fit,dev,info

@contextmanager
def activate():
    old=N.load,N.make_model,N.train_window,N.sources
    N.load,N.make_model,N.train_window,N.sources=load,C.make_model,B.train_window,sources
    try:yield
    finally:N.load,N.make_model,N.train_window,N.sources=old

def run(a,directory=None):
    with activate():return N.run(a,directory)
if __name__=='__main__':run(parser().parse_args())
