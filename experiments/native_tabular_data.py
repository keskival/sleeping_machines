"""Hash-verified numeric tabular rows with feature-duplicate group isolation."""
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import io

import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def load(dataset,fit=128,dev=128):
    entry=next(r for r in json.loads((ROOT/'experiments/tabular_data_manifest.json').read_text()) if r['dataset']==dataset)
    path=ROOT/entry['file']
    if not path.exists():
        archive=urllib.request.urlopen(entry['url'],timeout=60).read()
        if hashlib.sha256(archive).hexdigest()!=entry['archive_sha256']:raise ValueError('Dataset archive changed')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            raw=z.read(path.name)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Dataset bytes changed')
    if dataset=='banknote':
        rows=np.loadtxt(io.StringIO(raw.decode()),delimiter=','); names=['variance','skewness','curtosis','entropy']
    else:
        records=list(csv.reader(io.StringIO(raw.decode()),delimiter=';'));names=records[0][:-1];rows=np.asarray(records[1:],dtype=float)
    x,y=rows[:,:-1],rows[:,-1]
    # Group by features, never use labels to make duplicate groups or split.
    _,groups=np.unique(x,axis=0,return_inverse=True)
    rng=np.random.default_rng(1201);order=rng.permutation(groups.max()+1)
    n=len(order);blocks=(order[:int(.6*n)],order[int(.6*n):int(.8*n)],order[int(.8*n):])
    indices=[rng.permutation(np.flatnonzero(np.isin(groups,b))) for b in blocks]
    if fit>len(indices[0]) or dev>len(indices[1]):raise ValueError('Requested rows exceed isolated split')
    a,b=indices[0][:fit],indices[1][:dev]
    # Only these actual fitting rows train preprocessing.
    mean=x[a].mean(0);std=x[a].std(0);std=np.maximum(std,1e-6)
    scaled_fit=(x[a]-mean)/std;scaled_dev=(x[b]-mean)/std
    digest=lambda ids:hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()+np.asarray(rows[ids],dtype='<f8').tobytes()).hexdigest()
    return dict(x_fit=scaled_fit,y_fit=y[a],x_dev=scaled_dev,y_dev=y[b],feature_names=names,
        protocol=dict(dataset=dataset,raw_sha256=entry['sha256'],fit_sha256=digest(a),dev_sha256=digest(b),
            fit_indices=a.tolist(),dev_indices=b.tolist(),reserved_test_indices=indices[2].tolist(),
            group_split_seed=1201,duplicate_features_grouped=True,preprocessing_mean=mean.tolist(),
            preprocessing_std=std.tolist(),scaler_fit_rows=fit,test_labels_scored=False,
            test_scope='Reserved raw rows parsed during verification, never evaluated or selected on',
            classes_present_fit=sorted(set(y[a].tolist())) if dataset=='banknote' else None,
            source=entry['source'],license=entry['license']))
