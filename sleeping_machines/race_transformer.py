"""Run a trained causal Transformer as an asynchronous race-attention event stream (THEORY §397).

The E64 character Transformer (pre-norm nn.TransformerEncoder, ReLU feed-forward, learned positions, no final
norm) is executed one token event at a time with persistent per-layer key/value state.  Each attention head is a
race: clock rates exp(q.k/sqrt(d_h) - lambda_h * (t_now - t_i)) over the stored keys, whose first arrival is key
i with probability softmax_i (Theory §96).  Delivery modes per head:

  expected   the race's expected value sum_i pi_i v_i: exactly the Transformer's attention (contract-tested);
  sampled    the mean value of S independent races (exponential clocks): unbiased, S value reads instead of n;
  shortlist  the race restricted to the m highest-scoring stored keys (expected delivery within the shortlist).

lambda_h (per layer, head) adds elapsed-physical-time decay to the clocks.  It is zero at conversion, so the
converted model equals the Transformer, and it can be fitted to asynchronous streams while the imported
language weights stay fixed.  Key scoring still reads every stored key in sampled mode; only shortlist with a
candidate index would reduce key reads, which is reported separately (value reads, key scores).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RaceTransformerState:
    def __init__(self, layers):
        self.keys = [[] for _ in range(layers)]
        self.values = [[] for _ in range(layers)]
        self.times = []
        self.position = 0
        self.key_scores = self.value_reads = 0


class RaceTransformer(nn.Module):
    def __init__(self, net, mode='expected', samples=1, shortlist=None, generator=None):
        """net: a trained e64_lm_baselines.TfLM (or any pre-norm batch_first TransformerEncoder LM with emb/pos/out)."""
        super().__init__()
        if mode not in ('expected', 'sampled', 'shortlist'):
            raise ValueError('expected, sampled or shortlist delivery')
        self.net, self.mode, self.samples, self.shortlist = net, mode, samples, shortlist
        self.generator = generator
        layers = net.enc.layers
        self.heads = layers[0].self_attn.num_heads
        self.temporal_decay = nn.Parameter(torch.zeros(len(layers), self.heads))
        for layer in layers:
            if not layer.norm_first or layer.self_attn.batch_first is not True:
                raise ValueError('pre-norm batch-first encoder layers required')

    def new_state(self):
        return RaceTransformerState(len(self.net.enc.layers))

    def deliver(self, scores, values):
        """scores (H, n), values (H, n, dh) -> (H, dh) by the configured race delivery."""
        if self.mode == 'shortlist' and self.shortlist is not None and scores.shape[1] > self.shortlist:
            top = scores.topk(self.shortlist, dim=1).indices
            scores = scores.gather(1, top)
            values = values.gather(1, top[..., None].expand(-1, -1, values.shape[2]))
        pi = torch.softmax(scores, -1)
        if self.mode != 'sampled':
            return (pi[..., None] * values).sum(1), values.shape[1]
        clocks = torch.empty((self.samples,) + tuple(scores.shape), dtype=scores.dtype).exponential_(generator=self.generator)
        winners = (clocks / pi.clamp_min(1e-300)).argmin(-1)                      # (S, H): first arrival per race
        picked = values[torch.arange(values.shape[0])[None], winners]            # (S, H, dh)
        distinct = sum(int(winners[:, h].unique().numel()) for h in range(winners.shape[1]))
        return picked.mean(0), distinct / winners.shape[1]                       # distinct values read per head

    def consume(self, token, time, state):
        net, d = self.net, self.net.emb.weight.shape[1]
        H = self.heads; dh = d // H
        x = net.emb.weight[token] + net.pos.weight[state.position]
        state.times.append(float(time))
        ages = time - torch.tensor(state.times, dtype=x.dtype)
        for li, layer in enumerate(net.enc.layers):
            attn = layer.self_attn
            h = layer.norm1(x)
            q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).split(d)
            state.keys[li].append(k.view(H, dh)); state.values[li].append(v.view(H, dh))
            K = torch.stack(state.keys[li], 1); V = torch.stack(state.values[li], 1)    # (H, n, dh)
            scores = (q.view(H, 1, dh) * K).sum(-1) / math.sqrt(dh) - self.temporal_decay[li][:, None] * ages[None]
            out, reads = self.deliver(scores, V)
            state.key_scores += H * K.shape[1]; state.value_reads += H * reads
            x = x + F.linear(out.reshape(d), attn.out_proj.weight, attn.out_proj.bias)
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        state.position += 1
        return net.out(x)

    def forward_stream(self, tokens, times=None, state=None):
        state = self.new_state() if state is None else state
        times = range(len(tokens)) if times is None else times
        return torch.stack([self.consume(int(t), float(s), state) for t, s in zip(tokens, times)]), state
