"""Prepared CPU inference for tied maps and private temporal state.

Native parity remains unrun. Store one map bank per head, not per receiver;
retain private keys, clocks, rates, frequencies, memories and cached key reads.
No training/online mutation or concurrent serving API. Imports are stdlib only.
"""
import copy
import math

from .prepacked_sparse_inference import _model_signature, _tensor_signature

SHARED = ('input', 'output', 'gate', 'control', 'key_read')
PRIVATE = ('key', 'clock_bias', 'raw_rate', 'frequency')


def sharing_signature(model):
    """Require actual module tying; numerical equality of untied maps is insufficient."""
    identities = []
    private_ids = {name: [] for name in PRIVATE}
    for depth in range(model.depth):
        layer_gain = None
        for head in range(model.heads):
            pool = model.units[depth][head][0]
            if len(pool) != model.pool:
                raise ValueError('Receiver shape differs from declared pool')
            first = pool[0]
            for unit in pool:
                for name in SHARED:
                    if getattr(unit, name) is not getattr(first, name):
                        raise ValueError('Compact inference requires tied modules: ' + name)
                identities.append(tuple(id(getattr(unit, name)) for name in SHARED + PRIVATE))
                for name in PRIVATE:
                    private_ids[name].append(id(getattr(unit, name)))
                gain = float(unit.gain)
                if layer_gain is None:
                    layer_gain = gain
                if gain != layer_gain:
                    raise ValueError('Layer gain must agree with the original stacked producer')
    if any(len(values) != len(set(values)) for values in private_ids.values()):
        raise ValueError('Keys/clocks/timescales must retain distinct parameter objects')
    return tuple(identities)


def _runtime():
    import torch
    from .parallel_stream_language import precise_rotate
    return torch, precise_rotate


def _pack(torch, model):
    F = torch.nn.functional
    layers = []
    for depth in range(model.depth):
        units = [u for h in range(model.heads) for u in model.units[depth][h][0]]
        representatives = [model.units[depth][h][0][0] for h in range(model.heads)]
        layers.append(dict(
            key=torch.stack([u.key for u in units]),
            clock_bias=torch.stack([u.clock_bias for u in units]),
            rate=F.softplus(torch.stack([u.raw_rate for u in units])) + 1e-6,
            frequency=torch.stack([u.frequency for u in units]).to(torch.float64),
            query=torch.stack([model.queries[depth][h].weight for h in range(model.heads)]),
            input=torch.stack([u.input.weight for u in representatives]),
            output=torch.stack([u.output.weight for u in representatives]),
            key_read=torch.stack([u.key_read.weight for u in representatives]),
            gate_w=torch.stack([u.gate.weight for u in representatives]),
            gate_b=torch.stack([u.gate.bias for u in representatives]),
            control_w=torch.stack([u.control.weight for u in representatives]),
            control_b=torch.stack([u.control.bias for u in representatives]), gain=units[0].gain))
    return tuple(layers)


def _compact_logits(torch, precise_rotate, model, layers, rows, seed, all_logits=False):
    """Same temporal program; broadcast shared maps instead of gathering copies."""
    F = torch.nn.functional
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n = len(rows)
    dtype = model.embedding.weight.dtype
    lengths = torch.tensor([len(r['events']) for r in rows])
    T = int(lengths.max())
    mem = [torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)]
    arr = [torch.zeros(n, H, U, dtype=torch.float64) for _ in range(D)]
    seen = [torch.zeros(n, H, U, dtype=torch.bool) for _ in range(D)]
    reads = [L['key'].view(H, U, P).expand(n, H, U, P).clone() for L in layers]
    stamp_rows = torch.zeros(n, T, dtype=torch.float64)
    mark_rows = []
    for lane, row in enumerate(rows):
        events = row['events']
        stamp_rows[lane, :len(events)] = torch.tensor([float(e[0]) for e in events], dtype=torch.float64)
        marks = [torch.as_tensor(e[1], dtype=dtype) for e in events]
        mark_rows.append(torch.stack(marks + [marks[0]] * (T - len(events))))
    mark_rows = torch.stack(mark_rows)
    ctx_vals = torch.zeros(n, H * P, dtype=dtype)
    ctx_arr = torch.zeros(n, H, dtype=torch.float64)
    has_ctx = torch.zeros(n, dtype=torch.bool)
    out = torch.zeros(n, model.head.out_features, dtype=dtype)
    every = []
    lanes = torch.arange(n)

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for k in range(T):
            active = lengths > k
            x = model.embedding.weight[0][None] + model.content(mark_rows[:, k])
            arrival = stamp_rows[:, k]
            read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
            arrival = read_time
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h)
                                 for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                          (model.total_payload,)), x)
            for depth in range(D):
                L = layers[depth]
                mixed = model.channel_mix[depth](x)
                all_features = F.layer_norm(mixed, (model.total_payload,))
                incoming = mixed.view(n, H, P)
                query = torch.einsum('hpd,ld->lhp', L['query'], all_features)
                scores = ((query[:, :, None, :] * reads[depth]).sum(-1) / math.sqrt(P)
                          + L['clock_bias'].view(H, U)).clamp(-12, 12)
                values, arrivals = [], []
                for head in range(H):
                    noise = torch.empty(U, dtype=torch.float64).exponential_()
                    times = noise[None, :] / scores[:, head].to(torch.float64).exp()
                    first, w = times.min(-1)
                    idx = head * U + w
                    xh = incoming[:, head]
                    m = mem[depth][lanes, head, w]
                    prev = torch.where(seen[depth][lanes, head, w], arr[depth][lanes, head, w], arrival)
                    # Expanded stride-zero views preserve the producer's batched
                    # contractions; no [n,P,P] winner-map gather is constructed.
                    controls = torch.einsum('lcp,lp->lc', L['control_w'][head].expand(n, -1, -1),
                                            F.layer_norm(xh, (P,))) + L['control_b'][head]
                    forget = F.softplus(controls[:, 0]) / math.log(2)
                    write = 2 * torch.sigmoid(controls[:, 1])
                    age = (arrival - prev).clamp_min(0)
                    decay = torch.exp(-age.to(dtype)[:, None] * L['rate'].view(H * U, P // 2)[idx]
                                      * forget[:, None]).repeat_interleave(2, -1)
                    m_new = precise_rotate(m * decay, age[:, None] * L['frequency'].view(H * U, P // 2)[idx])
                    m_new = m_new + write[:, None] * torch.einsum('lpq,lq->lp', L['input'][head].expand(n, -1, -1), xh)
                    y = F.layer_norm(torch.einsum('lpq,lq->lp', L['output'][head].expand(n, -1, -1), m_new) + xh, (P,))
                    gate = torch.einsum('lpq,lq->lp', L['gate_w'][head].expand(n, -1, -1), F.gelu(y)) + L['gate_b'][head]
                    value = xh + L['gain'] * y * torch.sigmoid(gate)
                    mem[depth][lanes[active], head, w[active]] = m_new[active]
                    arr[depth][lanes[active], head, w[active]] = arrival[active]
                    seen[depth][lanes[active], head, w[active]] = True
                    refreshed = L['key'].view(H * U, P)[idx] + torch.einsum(
                        'lpq,lq->lp', L['key_read'][head].expand(n, -1, -1), m_new)
                    reads[depth][lanes[active], head, w[active]] = refreshed[active]
                    values.append(value)
                    arrivals.append(arrival + .001 + .010 * first / (1 + first))
                arrivals = torch.stack(arrivals, -1)
                arrival = arrivals.max(-1).values
                x = torch.cat([transport(values[h], arrival - arrivals[:, h], depth, h) for h in range(H)], -1)
            ctx_vals = torch.where(active[:, None], torch.cat(values, -1), ctx_vals)
            ctx_arr = torch.where(active[:, None], arrivals, ctx_arr)
            has_ctx = has_ctx | active
            last = lengths == k + 1
            logits = model.head(x)
            if all_logits:
                every.append(torch.where(active[:, None], logits, torch.zeros_like(logits)))
            out = torch.where(last[:, None], logits, out)
    return torch.stack(every, 1) if all_logits else out


class CompactTiedSparseWorker:
    """Own an immutable one-source CPU snapshot; private .data writes unsupported."""
    def __init__(self, model):
        if model.sources != 1 or getattr(model, '_fast_layers', None) is not None:
            raise ValueError('One-source quiescent model required')
        sharing = sharing_signature(model)
        torch, rotate = _runtime()
        parameters = list(model.parameters())
        if not parameters or any(p.device.type != 'cpu' for p in parameters):
            raise ValueError('CPU parameters required')
        if model.embedding.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('Float32/64 required')
        before = _model_signature(model)
        with torch.inference_mode(False), torch.no_grad():
            self._model = copy.deepcopy(model).eval()
            self._model.requires_grad_(False)
            sharing_signature(self._model)
            self._layers = _pack(torch, self._model)
        if _model_signature(model) != before or sharing_signature(model) != sharing:
            raise ValueError('Source changed during preparation')
        self._torch, self._rotate = torch, rotate
        self._signature = self._fingerprint()
        self._calls = 0

    def _fingerprint(self):
        layers = tuple(tuple((name, _tensor_signature(value) if self._torch.is_tensor(value) else value)
                             for name, value in layer.items()) for layer in self._layers)
        return _model_signature(self._model), sharing_signature(self._model), layers

    def evaluate(self, rows, seed, all_logits=False):
        if self._fingerprint() != self._signature:
            raise RuntimeError('Private compact snapshot changed; create a new worker')
        with self._torch.no_grad():
            output = _compact_logits(self._torch, self._rotate, self._model, self._layers, rows, seed, all_logits)
        self._calls += 1
        return output

    def accounting(self):
        tensors = list(self._model.parameters()) + list(self._model.buffers())
        packed = [value for layer in self._layers for value in layer.values() if self._torch.is_tensor(value)]
        return dict(prepared_stack_builds=1, completed_evaluation_calls=self._calls,
                    private_snapshot_tensor_payload_bytes=sum(t.numel() * t.element_size() for t in tensors),
                    retained_packed_tensor_payload_bytes=sum(t.numel() * t.element_size() for t in packed),
                    heavy_matrix_banks=self._model.depth * self._model.heads,
                    available_slots=self._model.depth * self._model.heads * self._model.pool,
                    explicit_winner_matrix_gather_payload_bytes_per_call=0,
                    per_call_unit_parameter_stacking_payload_bytes=0,
                    scope='Tensor payloads and explicit construction only; input/state/cache, metadata, allocator, internal kernel temporaries and physical traffic extra')
