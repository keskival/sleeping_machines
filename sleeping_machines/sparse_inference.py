"""Winner-only native inference with cached key reads (THEORY §414).

The batched path (batched_episodes.batched_logits) evaluates every unit's proposal and then selects one per race.  At
inference only the winner's proposal is needed: the forwarded value, the memory write and the control gates are the
winner's.  The race scores read key + key_read · m from the *stored* memory m (lazy decay is applied only when a slot is
written), so each slot's key read changes only when that slot is written.  Caching it per slot leaves, per head and
layer and event: U score dot products of length P, and the winner's maps (control, input, output, gate) plus one key_read
refresh of the written slot.  Pool size then costs U·P per head instead of about 4·U·P² multiply-adds.

Contract (tests/test_sparse_inference.py): logits equal batched_logits (no forces) within float64 rounding, with identical
race winners, on variable-length episodes, so the same seeds give the same routes.
"""
import math

import torch
from torch.nn import functional as F

from .parallel_stream_language import precise_rotate


class SparseStepper:
    """Incremental winner-only evaluator: one event per lane per call.  The race noise comes from a private generator
    seeded like batched_logits' forked global stream, so a full replay equals batched_logits (and sparse_logits)."""

    @torch.no_grad()
    def __init__(self, model, lanes, seed, deterministic=False):
        self.model, self.n = model, lanes
        self.layers = model._stacked(0)
        D, H, U, P = model.depth, model.heads, model.pool, model.payload
        dtype = model.embedding.weight.dtype
        self.mem = [torch.zeros(lanes, H, U, P, dtype=dtype) for _ in range(D)]
        self.arr = [torch.zeros(lanes, H, U, dtype=torch.float64) for _ in range(D)]
        self.seen = [torch.zeros(lanes, H, U, dtype=torch.bool) for _ in range(D)]
        self.reads = [L['key'].view(H, U, P).expand(lanes, H, U, P).clone() for L in self.layers]
        self.ctx_vals = torch.zeros(lanes, H * P, dtype=dtype)
        self.ctx_arr = torch.zeros(lanes, H, dtype=torch.float64)
        self.has_ctx = torch.zeros(lanes, dtype=torch.bool)
        self.generator = torch.Generator().manual_seed(seed)
        self.deterministic = deterministic          # every clock noise 1: the highest score wins (a labelled variant)

    def _transport(self, value, age, depth, head):
        model = self.model
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    @torch.no_grad()
    def step(self, stamps, marks, active=None):
        """stamps (n,) float64 times, marks (n, content) contents; active (n,) bool lanes that consume this event
        (inactive lanes keep their state).  Returns (n, classes) logits after the event."""
        model, n = self.model, self.n
        D, H, U, P = model.depth, model.heads, model.pool, model.payload
        dtype = model.embedding.weight.dtype
        active = torch.ones(n, dtype=torch.bool) if active is None else active
        lanes = torch.arange(n)
        x = model.embedding.weight[0][None] + model.content(marks.to(dtype))
        arrival = stamps.to(torch.float64)
        read_time = torch.where(self.has_ctx, torch.maximum(arrival, self.ctx_arr.max(-1).values), arrival)
        arrival = read_time
        context = torch.cat([self._transport(self.ctx_vals[:, h * P:(h + 1) * P], read_time - self.ctx_arr[:, h], D - 1, h)
                             for h in range(H)], -1)
        x = torch.where(self.has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                              (model.total_payload,)), x)
        for depth in range(D):
            Lp = self.layers[depth]
            mixed = model.channel_mix[depth](x)
            all_features = F.layer_norm(mixed, (model.total_payload,))
            incoming = mixed.view(n, H, P)
            query = torch.einsum('hpd,ld->lhp', Lp['query'], all_features)
            scores = ((query[:, :, None, :] * self.reads[depth]).sum(-1) / math.sqrt(P) + Lp['clock_bias'].view(H, U)).clamp(-12, 12)
            values, arrivals = [], []
            for head in range(H):
                noise = torch.empty(U, dtype=torch.float64).exponential_(generator=self.generator)   # batched draw order
                if self.deterministic:
                    noise = torch.ones_like(noise)
                times = noise[None, :] / scores[:, head].to(torch.float64).exp()
                first, w = times.min(-1)
                idx = head * U + w
                xh = incoming[:, head]
                m = self.mem[depth][lanes, head, w]
                prev = torch.where(self.seen[depth][lanes, head, w], self.arr[depth][lanes, head, w], arrival)
                controls = torch.einsum('lcp,lp->lc', Lp['control_w'].view(H * U, 2, P)[idx], F.layer_norm(xh, (P,))) \
                    + Lp['control_b'].view(H * U, 2)[idx]
                forget = F.softplus(controls[:, 0]) / math.log(2)
                write = 2 * torch.sigmoid(controls[:, 1])
                age = (arrival - prev).clamp_min(0)
                decay = torch.exp(-age.to(dtype)[:, None] * Lp['rate'].view(H * U, P // 2)[idx] * forget[:, None]).repeat_interleave(2, -1)
                m_new = precise_rotate(m * decay, age[:, None] * Lp['frequency'].view(H * U, P // 2)[idx])
                m_new = m_new + write[:, None] * torch.einsum('lpq,lq->lp', Lp['input'].view(H * U, P, P)[idx], xh)
                y = F.layer_norm(torch.einsum('lpq,lq->lp', Lp['output'].view(H * U, P, P)[idx], m_new) + xh, (P,))
                gate = torch.einsum('lpq,lq->lp', Lp['gate_w'].view(H * U, P, P)[idx], F.gelu(y)) + Lp['gate_b'].view(H * U, P)[idx]
                value = xh + Lp['gain'] * y * torch.sigmoid(gate)
                wl = lanes[active]; ww = w[active]
                self.mem[depth][wl, head, ww] = m_new[active]
                self.arr[depth][wl, head, ww] = arrival[active]
                self.seen[depth][wl, head, ww] = True
                refreshed = Lp['key'].view(H * U, P)[idx] + torch.einsum('lpq,lq->lp', Lp['key_read'].view(H * U, P, P)[idx], m_new)
                self.reads[depth][wl, head, ww] = refreshed[active]
                values.append(value); arrivals.append(arrival + .001 + .010 * first / (1 + first))
            arrivals = torch.stack(arrivals, -1)
            arrival = arrivals.max(-1).values
            x = torch.cat([self._transport(values[h], arrival - arrivals[:, h], depth, h) for h in range(H)], -1)
        self.ctx_vals = torch.where(active[:, None], torch.cat(values, -1), self.ctx_vals)
        self.ctx_arr = torch.where(active[:, None], arrivals, self.ctx_arr)
        self.has_ctx = self.has_ctx | active
        return model.head(x)


@torch.no_grad()
def sparse_logits(model, rows, seed, all_logits=False):
    """Winner-only evaluation of whole episodes (lanes of different lengths), via SparseStepper."""
    dtype = model.embedding.weight.dtype
    n = len(rows)
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    stepper = SparseStepper(model, n, seed)
    out = torch.zeros(n, model.head.out_features, dtype=dtype)
    every = []
    for k in range(T):
        active = lengths > k
        stamps = torch.tensor([float(r['events'][k][0]) if len(r['events']) > k else 0. for r in rows], dtype=torch.float64)
        marks = torch.stack([torch.as_tensor(r['events'][k][1] if len(r['events']) > k else r['events'][0][1], dtype=dtype)
                             for r in rows])
        logits_k = stepper.step(stamps, marks, active)
        if all_logits:
            every.append(torch.where(active[:, None], logits_k, torch.zeros_like(logits_k)))
        out = torch.where((lengths == k + 1)[:, None], logits_k, out)
    return torch.stack(every, 1) if all_logits else out
