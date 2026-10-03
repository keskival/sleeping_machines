"""Lifecycle contracts with a standard-library fake runtime, not native parity."""
from contextlib import nullcontext
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys


class Tensor:
    def __init__(self, value):
        self.value, self._version = value, 0
        self.shape, self.dtype, self.device = (2,), 'float32', SimpleNamespace(type='cpu')

    def numel(self):
        return 2

    def element_size(self):
        return 4


class Model:
    def __init__(self):
        self.sources, self.payload, self.depth, self.heads, self.pool = 1, 2, 1, 1, 2
        self.total_payload, self.content_dim, self.classes = 2, 2, 2
        self.embedding = SimpleNamespace(weight=Tensor(3.))
        self.gain, self.stacks, self.training = .25, 0, True

    def parameters(self):
        return iter([self.embedding.weight])

    def named_parameters(self):
        return iter([('embedding.weight', self.embedding.weight)])

    def buffers(self):
        return iter([])

    def named_buffers(self):
        return iter([])

    def named_modules(self):
        return iter([('', self)])

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, enabled):
        assert enabled is False
        return self

    def _stacked(self, source):
        assert source == 0
        self.stacks += 1
        return [dict(key=Tensor(self.embedding.weight.value), gain=self.gain)]


def main():
    ns = runpy.run_path(str(Path(__file__).resolve().parents[1]/'sleeping_machines/prepacked_sparse_inference.py'))
    worker_class = ns['PrepackedSparseWorker']
    calls = []
    def evaluate(view, rows, seed, all_logits=False):
        calls.append((seed, all_logits))
        first, second = view._stacked(0), view._stacked(0)
        assert first is second
        return first[0]['key'].value+rows[0]+view.gain
    fake = SimpleNamespace(float32='float32', float64='float64', no_grad=nullcontext,
                           inference_mode=lambda enabled: nullcontext(),
                           is_tensor=lambda value: isinstance(value, Tensor))
    worker_class.__init__.__globals__['_runtime'] = lambda: (fake, evaluate)
    source = Model()
    worker = worker_class(source)
    assert source.training and source.stacks == 0
    assert not worker._model.training and worker._model.stacks == 1
    assert worker.evaluate([2.], 11) == 5.25
    source.embedding.weight.value = 17.
    source.embedding.weight._version += 1
    source.gain = .9
    assert worker.evaluate([2.], 13, all_logits=True) == 5.25
    assert worker._model.stacks == 1
    assert calls == [(11, False), (13, True)]
    assert worker.accounting()['per_call_unit_parameter_stacking_payload_bytes'] == 0
    assert worker.accounting()['private_snapshot_tensor_payload_bytes'] == 8
    assert worker.accounting()['retained_packed_tensor_payload_bytes'] == 8
    for mutate in (
        lambda w: setattr(w._model, 'gain', .4),
        lambda w: setattr(w._model.embedding.weight, '_version', 1),
        lambda w: setattr(w._model.embedding, 'weight', Tensor(3.)),
        lambda w: setattr(w._layers[0]['key'], '_version', 1),
        lambda w: setattr(w._model, 'pool', 3),
        lambda w: setattr(w._model, 'training', True),
    ):
        other = worker_class(Model())
        mutate(other)
        try:
            other.evaluate([2.], 11)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Mutated private snapshot was accepted')
    for configure in (lambda m: setattr(m, 'sources', 2), lambda m: setattr(m, '_fast_layers', {})):
        bad = Model()
        configure(bad)
        try:
            worker_class(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported snapshot accepted')
    globals_ = worker_class.__init__.__globals__
    copier = globals_['copy']
    def changing_source(model):
        result = copier.deepcopy(model)
        model.embedding.weight._version += 1
        return result
    globals_['copy'] = SimpleNamespace(deepcopy=changing_source)
    try:
        try:
            worker_class(Model())
        except ValueError as error:
            assert 'Source changed during preparation' in str(error)
        else:
            raise AssertionError('Changing source snapshot admitted')
    finally:
        globals_['copy'] = copier
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(fake_lifecycle_contracts='passed', original_model_isolated=True,
               retained_stack_reused=True, mutation_guards=True, native_parity='pending'))


if __name__ == '__main__':
    main()
