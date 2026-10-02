"""Inference-only numeric port of the one-source addressed native temporal core.

Training and counterfactual credit remain in PyTorch. This path uses the same
winner-only proposals, clock races, local admission and persistent updates.
Static decay-rate transforms and target-independent race draws can be cached.
"""
import math
import numpy as np
from scipy.special import erf


def sigmoid(x):
    small = np.exp(-np.abs(x))
    return np.where(x >= 0, 1. / (1. + small), small / (1. + small))


def softplus(x):
    exact = np.maximum(x, 0.) + np.log1p(np.exp(-np.abs(x)))
    return np.where(x > 20., x, exact)


def norm(x):
    centered = x - x.mean()
    return centered / np.sqrt((centered * centered).mean() + x.dtype.type(1e-5))


def linear(x, values):
    weight, bias = values
    result = weight @ x
    return result if bias is None else result + bias


def rotate(value, angles):
    pairs = value.reshape(-1, 2); cosine = np.cos(angles).astype(value.dtype)
    sine = np.sin(angles).astype(value.dtype)
    return np.stack((cosine*pairs[:, 0]-sine*pairs[:, 1],
        sine*pairs[:, 0]+cosine*pairs[:, 1]), -1).reshape(-1)


def pack(model, noise_seed=314159, maximum_events=21):
    """One-time frozen packing; includes cached exact-reference random draws."""
    import torch
    from torch.nn import functional as F
    if model.sources != 1 or model.credit != 'counterfactual':
        raise ValueError('This inference port covers one-source counterfactual native models')
    if not 1 <= maximum_events <= 1000:
        raise ValueError('Bounded draw cache required')
    def array(x): return x.detach().cpu().numpy().copy()
    def affine(layer): return array(layer.weight), None if layer.bias is None else array(layer.bias)
    layers = []
    for depth in range(model.depth):
        heads = []
        for head in range(model.heads):
            units = []
            for unit in model.units[depth][head][0]:
                units.append(dict(key=array(unit.key), key_read=affine(unit.key_read),
                    clock_bias=float(unit.clock_bias.detach()), input=affine(unit.input), output=affine(unit.output),
                    gate=affine(unit.gate), control=affine(unit.control), rate=array(F.softplus(unit.raw_rate)+1e-6),
                    frequency=array(unit.frequency).astype(np.float64), gain=unit.gain))
            heads.append(dict(query=affine(model.queries[depth][head]), units=units))
        layers.append(dict(mix=affine(model.channel_mix[depth]), heads=heads))
    with torch.random.fork_rng():
        torch.manual_seed(noise_seed)
        noise = np.array([array(torch.empty(model.pool, dtype=torch.float64).exponential_())
            for _ in range(maximum_events*model.depth*model.heads)])
    return dict(embedding=array(model.embedding.weight[0]), content=affine(model.content),
        source_gate=affine(model.source_gate), head=affine(model.head), layers=layers,
        transport_rate=array(F.softplus(model.transport_rate)+1e-6),
        transport_frequency=array(model.transport_frequency).astype(np.float64),
        payload=model.payload, depth=model.depth, heads=model.heads, pool=model.pool,
        content_dim=model.content_dim, noise=noise, noise_seed=noise_seed, maximum_events=maximum_events)


class NumpyAddressedInference:
    def __init__(self, packed):
        self.p = packed; self.dtype = packed['embedding'].dtype
        self.L, self.H, self.U, self.P = (packed[k] for k in ('depth', 'heads', 'pool', 'payload'))

    def new_state(self):
        return dict(memories=np.zeros((self.L, self.H, self.U, self.P), dtype=self.dtype),
            arrivals=np.zeros((self.L, self.H, self.U), dtype=np.float64),
            seen=np.zeros((self.L, self.H, self.U), dtype=bool), context=None, context_times=None,
            events=0, queue_wait_sum=0., last_input_time=-math.inf, noise_cursor=0, winners=[])

    def transport(self, value, age, depth, head):
        age = max(0., float(age)); rate = self.p['transport_rate'][depth, head]
        decayed = value * np.repeat(np.exp(-self.dtype.type(age)*rate), 2)
        return rotate(decayed, age*self.p['transport_frequency'][depth, head])

    def align(self, values, times, arrival, depth):
        return np.concatenate([self.transport(value, arrival-times[head], depth, head)
            for head, value in enumerate(values)])

    def consume(self, timestamp, content, state):
        timestamp = float(timestamp); content = np.asarray(content, dtype=self.dtype)
        if not math.isfinite(timestamp) or timestamp < state['last_input_time']:
            raise ValueError('Finite nondecreasing physical timestamps required')
        if content.shape != (self.p['content_dim'],) or not np.isfinite(content).all():
            raise ValueError('Finite observed fixed-width content required')
        if state['events'] >= self.p['maximum_events']:
            raise ValueError('Explicit bounded cache exhausted; never wrap the random stream')
        x = self.p['embedding'] + linear(content, self.p['content']); arrival = timestamp
        if state['context'] is not None:
            ready = float(state['context_times'].max()); arrival = max(arrival, ready)
            state['queue_wait_sum'] += arrival-timestamp
            context = self.align(state['context'], state['context_times'], arrival, self.L-1)
            x = norm(x + sigmoid(linear(x, self.p['source_gate'])) * context)
        for depth, layer in enumerate(self.p['layers']):
            mixed = linear(x, layer['mix']); features = norm(mixed)
            values = []; times = []
            for head, definition in enumerate(layer['heads']):
                incoming = mixed[head*self.P:(head+1)*self.P]
                query = linear(features, definition['query']); scores = []
                for index, unit in enumerate(definition['units']):
                    read = unit['key'] + linear(state['memories'][depth, head, index], unit['key_read'])
                    scores.append((query @ read) / math.sqrt(self.P) + self.dtype.type(unit['clock_bias']))
                scores = np.clip(np.asarray(scores, dtype=self.dtype), -12., 12.).astype(np.float64)
                noise = self.p['noise'][state['noise_cursor']]; state['noise_cursor'] += 1
                races = noise / np.exp(scores); index = int(races.argmin()); race_time = races[index]
                delay = .001 + .010*race_time/(1.+race_time)
                unit = definition['units'][index]; memory = state['memories'][depth, head, index]
                controls = linear(norm(incoming), unit['control'])
                forget = softplus(controls[0]) / math.log(2.); write = 2.*sigmoid(controls[1])
                if state['seen'][depth, head, index]:
                    age = max(0., arrival-state['arrivals'][depth, head, index])
                    decay = np.exp(-self.dtype.type(age)*unit['rate']*forget)
                    memory = rotate(memory*np.repeat(decay, 2), age*unit['frequency'])
                memory = memory + write*linear(incoming, unit['input'])
                y = norm(linear(memory, unit['output']) + incoming)
                gelu = .5*y*(1.+erf(y*math.sqrt(.5)))
                value = incoming + unit['gain']*y*sigmoid(linear(gelu, unit['gate']))
                state['memories'][depth, head, index] = memory
                state['arrivals'][depth, head, index] = arrival; state['seen'][depth, head, index] = True
                state['winners'].append(index); values.append(value); times.append(arrival+delay)
            times = np.array(times, dtype=np.float64); arrival = float(times.max())
            x = self.align(values, times, arrival, depth)
        state['context'] = np.array(values); state['context_times'] = times
        state['events'] += 1; state['last_input_time'] = timestamp
        return linear(x, self.p['head']), arrival

    def predict(self, events):
        state = self.new_state()
        for timestamp, content in events:
            logits, arrival = self.consume(timestamp, content, state)
        return logits, state
