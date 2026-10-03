"""Stdlib alias/lifecycle/packing-ledger contracts, never native numerical parity."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import sleeping_machines.compact_tied_inference as compact
from compact_tied_resource_geometry import compact_accounting


class Tensor:
    def __init__(self, value=1., shape=(2,)):
        self.value, self.shape, self._version = value, shape, 0
        self.dtype, self.device = 'float32', SimpleNamespace(type='cpu')

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def element_size(self):
        return 8 if self.dtype == 'float64' else 4

    def to(self, dtype):
        result = copy.copy(self)
        result.dtype = dtype
        return result

    def __add__(self, scalar):
        return copy.copy(self)


def shape_model(payload, depth, heads, pool, scalar_bytes):
    def tensor(shape):
        t = Tensor(shape=shape)
        t.dtype = 'float64' if scalar_bytes == 8 else 'float32'
        return t
    units = []
    for _ in range(depth):
        groups = []
        for _ in range(heads):
            shared = {name: SimpleNamespace(weight=tensor((payload, payload))) for name in compact.SHARED}
            shared['gate'].bias = tensor((payload,))
            shared['control'].weight = tensor((2, payload))
            shared['control'].bias = tensor((2,))
            groups.append([[SimpleNamespace(**shared, key=tensor((payload,)), clock_bias=tensor(()),
                raw_rate=tensor((payload // 2,)), frequency=tensor((payload // 2,)), gain=.5)
                for _ in range(pool)]])
        units.append(groups)
    queries = [[SimpleNamespace(weight=tensor((payload, heads * payload))) for _ in range(heads)]
               for _ in range(depth)]
    return SimpleNamespace(payload=payload, depth=depth, heads=heads, pool=pool, units=units, queries=queries)


def shape_stack(tensors):
    assert all(t.shape == tensors[0].shape and t.dtype == tensors[0].dtype for t in tensors)
    return Tensor(shape=(len(tensors),) + tensors[0].shape).to(tensors[0].dtype)


class Model:
    def __init__(self, pool=4):
        self.sources, self.depth, self.heads, self.pool, self.payload = 1, 1, 1, pool, 2
        self.total_payload, self.content_dim, self.classes, self.training = 2, 2, 2, True
        self.embedding = SimpleNamespace(weight=Tensor(3.))
        shared = {name: SimpleNamespace(weight=Tensor(), bias=Tensor()) for name in compact.SHARED}
        units = [SimpleNamespace(**shared, **{name: Tensor() for name in compact.PRIVATE}, gain=.5)
                 for _ in range(pool)]
        self.units = [[[units]]]

    def named_parameters(self):
        result = [('embedding.weight', self.embedding.weight)]
        seen = {id(self.embedding.weight)}
        for i, unit in enumerate(self.units[0][0][0]):
            for name in compact.PRIVATE:
                value = getattr(unit, name)
                if id(value) not in seen:
                    result.append((f'u{i}.{name}', value)); seen.add(id(value))
            for name in compact.SHARED:
                for field in ('weight', 'bias'):
                    value = getattr(getattr(unit, name), field)
                    if id(value) not in seen:
                        result.append((f'u{i}.{name}.{field}', value)); seen.add(id(value))
        return iter(result)

    def parameters(self):
        return iter(value for _, value in self.named_parameters())

    def named_buffers(self):
        return iter([])

    def buffers(self):
        return iter([])

    def named_modules(self):
        return iter([('', self)] + [(f'u{i}', unit) for i, unit in enumerate(self.units[0][0][0])])

    def eval(self):
        self.training = False
        for unit in self.units[0][0][0]:
            unit.training = False
        return self

    def requires_grad_(self, enabled):
        assert enabled is False
        return self


def rejects(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid compact worker accepted')


def main(manifest_name=None):
    cases = 0
    for payload in (2, 8, 32, 64, 96):
        for pool in (1, 2, 4, 8, 32):
            for scalar in (4, 8):
                r = compact_accounting(payload, 4, 2, pool, 3, scalar)
                # Independent enumeration of actual compact dictionary shapes.
                private = 8 * pool * (scalar * (payload + 1 + payload // 2) + 8 * payload // 2)
                weights = 8 * scalar * (4 * payload ** 2 + 3 * payload + 2)
                queries = 8 * scalar * payload * (2 * payload)
                assert r['compact_final_packed_tensor_bytes'] == private + weights + queries
                assert r['compact_heavy_map_banks'] == 8 and r['available_slots'] == 8 * pool
                assert r['selected_writes_per_position'] == 8 and r['scored_keys_per_position'] == 8 * pool
                assert r['unit_memory_bytes'] == r['cached_key_read_bytes'] == 3 * 8 * pool * payload * scalar
                assert (r['avoided_duplicate_packed_tensor_bytes'] == 0) == (pool == 1)
                # Execute the actual packer against shape-only tensors, without
                # importing torch or claiming a numerical contraction result.
                shaped = shape_model(payload, 4, 2, pool, scalar)
                compact.sharing_signature(shaped)
                runtime = SimpleNamespace(stack=shape_stack, float64='float64',
                    nn=SimpleNamespace(functional=SimpleNamespace(softplus=lambda t: t)))
                bank = compact._pack(runtime, shaped)
                byte_total = sum(t.numel() * t.element_size() for layer in bank for t in layer.values()
                                 if isinstance(t, Tensor))
                assert byte_total == r['compact_final_packed_tensor_bytes']
                assert all(layer['input'].shape == (2, payload, payload) for layer in bank)
                assert all(layer['key'].shape == (2 * pool, payload) for layer in bank)
                cases += 1
    for values in ((8, 2, 2, True, 3), (8, 2, 2, 4, 0), (7, 2, 2, 4, 3)):
        rejects(lambda: compact_accounting(*values))
    m = Model()
    for unit in m.units[0][0][0]:
        unit.training = True
    signature = compact.sharing_signature(m)
    assert compact.sharing_signature(copy.deepcopy(m)) != signature
    invalid = copy.deepcopy(m)
    invalid.units[0][0][0][1].input = copy.deepcopy(invalid.units[0][0][0][0].input)
    rejects(lambda: compact.sharing_signature(invalid))
    invalid = copy.deepcopy(m)
    invalid.units[0][0][0][1].key = invalid.units[0][0][0][0].key
    rejects(lambda: compact.sharing_signature(invalid))
    invalid = copy.deepcopy(m)
    invalid.units[0][0][0][1].gain = .7
    rejects(lambda: compact.sharing_signature(invalid))
    fake = SimpleNamespace(float32='float32', float64='float64', no_grad=nullcontext,
        inference_mode=lambda _: nullcontext(), is_tensor=lambda t: isinstance(t, Tensor))
    saved = compact._runtime, compact._pack, compact._compact_logits
    builds = []
    calls = []
    def pack(torch, model):
        builds.append(model)
        return (dict(key=Tensor(model.embedding.weight.value), gain=.5),)
    def evaluate(torch, rotate, model, layers, rows, seed, all_logits):
        calls.append((seed, all_logits))
        return layers[0]['key'].value + rows[0]
    compact._runtime = lambda: (fake, None)
    compact._pack, compact._compact_logits = pack, evaluate
    try:
        worker = compact.CompactTiedSparseWorker(m)
        assert m.training and len(builds) == 1
        assert worker.evaluate([2.], 17) == 5.
        m.embedding.weight.value = 9.; m.embedding.weight._version += 1
        assert worker.evaluate([2.], 19, True) == 5.
        assert len(builds) == 1 and calls == [(17, False), (19, True)]
        assert worker.accounting()['private_snapshot_tensor_payload_bytes'] > 8
        assert worker.accounting()['explicit_winner_matrix_gather_payload_bytes_per_call'] == 0
        mutations = (
            lambda w: setattr(w._model.embedding.weight, '_version', w._model.embedding.weight._version + 1),
            lambda w: setattr(w._layers[0]['key'], '_version', 1),
            lambda w: setattr(w._model.units[0][0][0][1], 'input', copy.deepcopy(w._model.units[0][0][0][0].input)),
            lambda w: setattr(w._model.units[0][0][0][1], 'key', w._model.units[0][0][0][0].key),
            lambda w: setattr(w._model, 'pool', 3),
            lambda w: setattr(w._model, 'training', True),
        )
        for mutate in mutations:
            other = compact.CompactTiedSparseWorker(m)
            mutate(other)
            rejects(lambda: other.evaluate([2.], 17))
        copier = compact.copy
        def changing_source(model):
            result = copier.deepcopy(model)
            model.embedding.weight._version += 1
            return result
        compact.copy = SimpleNamespace(deepcopy=changing_source)
        try:
            rejects(lambda: compact.CompactTiedSparseWorker(m))
        finally:
            compact.copy = copier
    finally:
        compact._runtime, compact._pack, compact._compact_logits = saved
    # Actual alias rejection occurs before the lazy numerical import.
    rejects(lambda: compact.CompactTiedSparseWorker(invalid))
    rejected_manifests = 0
    if manifest_name:
        from compact_tied_inference_contracts import SOURCES, run, sha
        prepared = json.loads((ROOT / manifest_name).read_text())
        assert prepared['source_sha256'] == {name: sha(ROOT / name) for name in SOURCES}
        assert sha(ROOT / prepared['queue']) == prepared['queue_sha256']
        assert not (ROOT / prepared['arguments']['out']).exists()
        for changes in (dict(status='failed'), dict(source_sha256={}), dict(queue_sha256='wrong'),
                        dict(parent_sha256='wrong'), dict(weights_sha256='wrong')):
            with tempfile.TemporaryDirectory(prefix='compact-admission-') as directory:
                path = Path(directory) / 'manifest.json'
                args = SimpleNamespace(manifest=str(path), out=str(Path(directory) / 'out.json'), result=None)
                bad = dict(prepared, **changes, arguments=vars(args))
                path.write_text(json.dumps(bad))
                rejects(lambda: run(args))
                rejected_manifests += 1
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='passed', packing_shape_cases=cases, alias_and_lifecycle_contracts='passed',
                          frozen_manifest_rejections=rejected_manifests, native_parity='unrun',
                          p64_D4_H2_U4_FP32=compact_accounting(64, 4, 2, 4, 64))))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
