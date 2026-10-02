"""Bounded exact inclusion probabilities for sequential weighted replay sampling."""
from functools import lru_cache
import itertools

import numpy as np


@lru_cache(maxsize=8)
def orders(races, k):
    if not 1 <= k <= 3 or not k <= races <= 32:
        raise ValueError('Exact small-pool protocol requires k<=3 and R<=32')
    return np.array(list(itertools.permutations(range(races),k)),dtype=np.int64)


def inclusion(proposal,k):
    p=np.asarray(proposal,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<=0) or abs(p.sum()-1)>1e-12:
        raise ValueError('Finite positive normalized proposal required')
    draws=orders(len(p),k);values=p[draws]
    used=np.cumsum(values,axis=1)-values
    weights=np.prod(values/(1-used),axis=1)
    marginal=np.bincount(draws.ravel(),weights=np.repeat(weights,k),minlength=len(p))
    pair=np.diag(marginal)
    for i in range(k):
        for j in range(i+1,k):
            np.add.at(pair,(draws[:,i],draws[:,j]),weights)
            np.add.at(pair,(draws[:,j],draws[:,i]),weights)
    if abs(weights.sum()-1)>1e-12 or abs(marginal.sum()-k)>1e-11:
        raise ArithmeticError('Inclusion normalization failed')
    return marginal,pair


def parameter_variance(vectors,marginal,pair):
    v=np.asarray(vectors,dtype=np.float64);gram=v@v.T
    return float(((pair/np.outer(marginal,marginal)-1)*gram).sum())


def sample(rng,proposal,k):
    marginal,_=inclusion(proposal,k)
    chosen=rng.choice(len(proposal),size=k,replace=False,p=proposal)
    return chosen,1/marginal[chosen]
