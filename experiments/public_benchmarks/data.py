"""Strict archive loader and TRAIN-only deterministic development splits.

No runtime or ML dependencies; fetching is separate from benchmark execution.
"""
import hashlib
import math
import random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXPECTED={'JapaneseVowels':(270,370,12,9),'ECG200':(100,100,1,2),'PenDigits':(7494,3498,2,10)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ts(path):
    metadata={};rows=[];started=False
    for raw in Path(path).read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#'):continue
        if not started:
            name,_,value=line.partition(' ');metadata[name.lower()]=value
            if name.lower()=='@data':started=True
            continue
        pieces=line.split(':');label=pieces.pop()
        channels=[[float(x) for x in piece.split(',')] for piece in pieces]
        if len({len(c) for c in channels})!=1:raise ValueError('Misaligned channels')
        if not all(math.isfinite(x) for c in channels for x in c):raise ValueError('Missing/nonfinite data requires a separate protocol')
        rows.append(dict(label=label,values=list(map(list,zip(*channels)))))
    if metadata.get('@timestamps','false').lower()!='false':raise ValueError('Timestamped TS requires a separate parser')
    if not rows:raise ValueError('Empty data')
    return rows,metadata


def load_train(name):
    rows,meta=read_ts(ROOT/'data/public_benchmarks/raw'/f'{name}_TRAIN.ts')
    n,_,dim,classes=EXPECTED[name]
    assert len(rows)==n and all(len(r['values'][0])==dim for r in rows)
    labels=sorted({r['label'] for r in rows});assert len(labels)==classes
    mapping={c:i for i,c in enumerate(labels)}
    for i,r in enumerate(rows):r.update(target=mapping[r['label']],identity=f'TRAIN:{i}')
    return rows,mapping,meta


def split(rows,seed=20261004):
    rng=random.Random(seed);fit=[];dev=[]
    for label in sorted({r['target'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['target']==label];rng.shuffle(indices)
        count=max(1,round(.2*len(indices)))
        dev.extend(indices[:count]);fit.extend(indices[count:])
    fit.sort();dev.sort()
    assert not set(fit)&set(dev) and len(fit)+len(dev)==len(rows)
    return fit,dev


def contracts():
    a=[dict(target=i%3) for i in range(30)];f,d=split(a)
    assert len(d)==6 and split(a)==(f,d)
    assert all(sum(a[i]['target']==c for i in d)==2 for c in range(3))
    return 'TRAIN-only split deterministic, disjoint and stratified'
