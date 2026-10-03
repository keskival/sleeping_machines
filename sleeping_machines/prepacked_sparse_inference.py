"""A fixed-weight CPU worker for the existing winner-only evaluator.

Prepared implementation; native numerical admission remains pending.
One private model snapshot and one unit-matrix stack are retained across calls.
Calls preserve reset-per-episode semantics and the original race/RNG program.
Use one worker per process: the underlying evaluator uses process-global RNG
forking and this interface does not add concurrent serving semantics.
"""
import copy


def _runtime():
    import torch
    from .sparse_inference import sparse_logits
    return torch, sparse_logits


def _tensor_signature(value):
    return (id(value), value._version, tuple(value.shape), value.dtype, value.device)


def _model_signature(model):
    parameters = tuple((name, _tensor_signature(value)) for name, value in model.named_parameters())
    buffers = tuple((name, _tensor_signature(value)) for name, value in model.named_buffers())
    gains = tuple((name, float(module.gain)) for name, module in model.named_modules()
                  if hasattr(module, 'gain'))
    modes = tuple((name, module.training) for name, module in model.named_modules())
    shape = tuple(getattr(model, name) for name in
                  ('sources', 'payload', 'depth', 'heads', 'pool', 'total_payload', 'content_dim', 'classes'))
    return parameters, buffers, gains, modes, shape


class _PreparedView:
    def __init__(self, model, layers):
        self._model, self._layers = model, layers

    def _stacked(self, source):
        if source != 0:
            raise ValueError('Current sparse evaluator uses observed source zero only')
        return self._layers

    def __getattr__(self, name):
        return getattr(self._model, name)


class PrepackedSparseWorker:
    """Own a frozen snapshot; construct a new worker for a new weight version.

    Prepare at a quiescent weight boundary; source versions are checked across
    preparation. Afterwards the caller's model can change independently.
    Private snapshot/packed tensors must not be mutated; ordinary in-place
changes/replacements are checked before each call. Deliberate .data writes are
unsupported. Storage accounting counts tensor payloads, not allocator/RSS or
physical memory traffic. Training/online updates are outside this interface.
"""
    def __init__(self, model):
        torch, evaluate = _runtime()
        if getattr(model, '_fast_layers', None) is not None:
            raise ValueError('Prepare outside a live training chunk')
        if model.sources != 1:
            raise ValueError('Admission currently covers one-source language models')
        parameters = list(model.parameters())
        if not parameters or any(p.device.type != 'cpu' for p in parameters):
            raise ValueError('Current evaluator/admission is CPU only')
        if model.embedding.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('Current admission covers float32/float64')
        original_signature = _model_signature(model)
        # Retained tensors need version counters even if the caller is in
        # inference_mode. The numerical evaluator still disables gradients.
        with torch.inference_mode(False), torch.no_grad():
            self._model = copy.deepcopy(model).eval()
            self._model.requires_grad_(False)
            # The producer creates new stacked tensors and transforms rates;
            # retained tensors have no parameter gradient graph in this context.
            self._layers = tuple(self._model._stacked(0))
        if _model_signature(model) != original_signature:
            raise ValueError('Source changed during preparation; use a quiescent weight boundary')
        self._torch, self._evaluate = torch, evaluate
        self._view = _PreparedView(self._model, self._layers)
        self._signature = self._fingerprint()
        self._calls = 0

    def _fingerprint(self):
        layers = tuple(tuple((name, _tensor_signature(value) if self._torch.is_tensor(value) else value)
                             for name, value in layer.items()) for layer in self._layers)
        return _model_signature(self._model), layers

    def evaluate(self, rows, seed, all_logits=False):
        if self._fingerprint() != self._signature:
            raise RuntimeError('Private inference snapshot changed; construct a new worker')
        output = self._evaluate(self._view, rows, seed, all_logits=all_logits)
        self._calls += 1
        return output

    def accounting(self):
        parameters = list(self._model.parameters())
        buffers = list(self._model.buffers())
        packed = [value for layer in self._layers for value in layer.values()
                  if self._torch.is_tensor(value)]
        return dict(prepared_stack_builds=1, completed_evaluation_calls=self._calls,
                    private_snapshot_tensor_payload_bytes=sum(t.numel()*t.element_size() for t in parameters+buffers),
                    retained_packed_tensor_payload_bytes=sum(t.numel()*t.element_size() for t in packed),
                    per_call_unit_parameter_stacking_payload_bytes=0,
                    scope='Tensor payloads only; original model, state/cache, gathers, metadata checks, temporaries and RSS/traffic extra')
