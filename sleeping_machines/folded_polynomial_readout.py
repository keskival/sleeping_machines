"""Frozen polynomial decoder with fit-only normalization folded into weights."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def polynomial_features(value, degree):
    value=np.asarray(value,dtype=np.float64)
    if degree==1:return value
    if degree!=2:raise ValueError('Only affine/degree2 local readout is covered')
    i,j=np.triu_indices(value.shape[-1]);products=value[...,i]*value[...,j]/math.sqrt(value.shape[-1])
    return np.concatenate([value,products],-1)


class FoldedPolynomialReadout(nn.Module):
    def __init__(self, coefficients, intercept, center, scale, dimension, degree):
        super().__init__()
        effective=np.asarray(coefficients,dtype=np.float64)/np.asarray(scale,dtype=np.float64)
        bias=np.asarray(intercept,dtype=np.float64)-effective@np.asarray(center,dtype=np.float64)
        self.dimension,self.degree=dimension,degree
        expected=dimension+(dimension*(dimension+1)//2 if degree==2 else 0)
        if effective.shape[1]!=expected:raise ValueError('Polynomial dimension mismatch')
        self.register_buffer('linear_weight',torch.from_numpy(effective[:,:dimension].copy()))
        self.register_buffer('bias',torch.from_numpy(bias.copy()))
        self.register_buffer('quadratic_weight',torch.from_numpy(effective[:,dimension:].copy()))
        i,j=np.triu_indices(dimension)
        self.register_buffer('first',torch.from_numpy(i));self.register_buffer('second',torch.from_numpy(j))

    def forward(self,value):
        value=value.to(self.linear_weight.dtype)
        result=F.linear(value,self.linear_weight,self.bias)
        if self.degree==2:
            products=value[...,self.first]*value[...,self.second]/math.sqrt(self.dimension)
            result=result+F.linear(products,self.quadratic_weight)
        return result
