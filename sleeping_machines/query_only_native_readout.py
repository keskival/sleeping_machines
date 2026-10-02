"""Inference-only query admission for the unchanged native affine classifier."""
import copy

import torch
from torch import nn


class QueryOnlyLinear(nn.Linear):
    query_active = False

    def forward(self, x):
        if self.training:
            raise RuntimeError('Query-only wrapper is inference-only; retain original training driver')
        if self.query_active:
            return super().forward(x)
        return x.new_empty((0,))


def install(model):
    if type(model.head) is not nn.Linear:
        raise ValueError('Exact optimization covers the original affine classifier only')
    result = copy.deepcopy(model)
    result.head.__class__ = QueryOnlyLinear
    result.eval()
    return result


@torch.no_grad()
def predict(model, row, seed=314159):
    events = row['events']
    if not events or events[-1][1][-1] != 1 or any(c[-1] != 0 for _,c in events[:-1]):
        raise ValueError('One explicit observed terminal query required')
    if not isinstance(model.head, QueryOnlyLinear):
        raise ValueError('Install inference-only affine wrapper first')
    model.eval(); state = model.new_state()
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for timestamp, content in events:
                model.head.query_active = bool(content[-1] == 1)
                logits,_ = model.consume_event(0,timestamp,content,state)
    finally:
        model.head.query_active = False
    return logits,state
