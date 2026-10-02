"""Demand-only final-query features from the frozen native numeric port."""
import numpy as np
from .numpy_addressed_inference import NumpyAddressedInference


class QueryFeatureInference(NumpyAddressedInference):
    def __init__(self,packed):
        packed=dict(packed)
        packed['head']=(np.empty((0,packed['heads']*packed['payload']),dtype=packed['embedding'].dtype),None)
        super().__init__(packed)
        self.query_vector=None

    def align(self,values,times,arrival,depth):
        result=super().align(values,times,arrival,depth)
        self.query_vector=result
        return result

    def predict_query(self,events):
        if not events or events[-1][1][-1]!=1 or any(c[-1]!=0 for _,c in events[:-1]):
            raise ValueError('Exactly one explicit observed terminal query required')
        self.query_vector=None
        _,state=super().predict(events)
        if self.query_vector is None:raise ValueError('Query computation required')
        return self.query_vector.copy(),state
