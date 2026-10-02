"""Portable dense 33-anchor readout of an unchanged native query context."""
import json
from pathlib import Path

import numpy as np
from .query_feature_inference import QueryFeatureInference


def basis(x, anchors, gamma):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-gamma*((x[..., None, :]-anchors)**2).sum(-1))


def probabilities(x, weights, bias):
    z = x@weights.T+bias; z -= z.max(-1, keepdims=True)
    p = np.exp(z); return p/p.sum(-1, keepdims=True)


def save(path, model):
    arrays = {}
    def encode(v):
        if isinstance(v, np.ndarray):
            key = f'a{len(arrays)}'; arrays[key] = v
            return {'array':key}
        if isinstance(v, tuple): return {'tuple':[encode(x) for x in v]}
        if isinstance(v, list): return [encode(x) for x in v]
        if isinstance(v, dict): return {k:encode(x) for k,x in v.items()}
        if isinstance(v, np.generic): return v.item()
        return v
    tree = json.dumps(encode(model), separators=(',',':'))
    np.savez(path, metadata=np.array(tree), **arrays)
    return dict(array_bytes=sum(a.nbytes for a in arrays.values()), metadata_utf8_bytes=len(tree.encode()), serialized_bytes=Path(path).stat().st_size)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        def decode(v):
            if isinstance(v, dict):
                if set(v)=={'array'}: return z[v['array']].copy()
                if set(v)=={'tuple'}: return tuple(decode(x) for x in v['tuple'])
                return {k:decode(x) for k,x in v.items()}
            if isinstance(v, list): return [decode(x) for x in v]
            return v
        return decode(json.loads(str(z['metadata'])))


class Predictor:
    def __init__(self, model):
        self.m = model
        self.port = QueryFeatureInference(model['packed']) if model['kind']=='native' else None

    def predict(self, counts):
        m = self.m; bins = m['bins']
        coarse = np.asarray(counts).reshape(bins, 20//bins, 32).sum(1)
        raw = np.log1p(coarse.reshape(-1))
        if self.port is None:
            features = raw
        else:
            values = ((raw-m['packet_center'])/m['packet_scale']).reshape(bins,32)
            events = [((i+1)/bins, np.r_[v,0.].astype(np.float32)) for i,v in enumerate(values)]
            events.append((1.,np.r_[np.zeros(32),1.].astype(np.float32)))
            features,_ = self.port.predict_query(events)
        normalized = (features-m['center'])/m['scale']
        return probabilities(basis(normalized,m['anchors'],m['gamma']),m['weights'],m['bias'])
