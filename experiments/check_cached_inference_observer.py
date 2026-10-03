"""Stdlib observer recovery checks on a fake runtime, never a tensor contract."""
from functools import wraps
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class Base:
    def min(self, dim):
        return SimpleNamespace(indices=Fake([min(range(len(self.values)), key=self.values.__getitem__)],
                                            shape=(1,), dtype='int64'))


class Fake(Base):
    def __init__(self, values, shape=(1, 2), dtype='float64'):
        self.values, self.shape, self.dtype = list(values), shape, dtype

    def detach(self):
        return self

    def clone(self):
        return Fake(self.values, self.shape, self.dtype)


def body(value):
    mem = arr = seen = ctx_vals = ctx_arr = has_ctx = value.clone()
    value.min(-1)
    return value


@wraps(body)
def wrapped(value):
    return body(value)


def failure(value):
    value.min(-1)
    raise RuntimeError('Constructed producer failure')


def main():
    observe = runpy.run_path(str(ROOT/'experiments/cached_inference_contracts.py'))['observe']
    runtime = SimpleNamespace(Tensor=Fake, float64='float64')
    original, previous = Fake.min, sys.getprofile()
    assert previous is None and 'min' not in Fake.__dict__
    value = Fake([1., 2.])
    with observe(runtime, wrapped, 1, 2) as tape:
        assert wrapped(value) is value
        assert 'min' in Fake.__dict__
    assert len(tape['races']) == 1
    assert set(tape['state']) == {'mem', 'arr', 'seen', 'ctx_vals', 'ctx_arr', 'has_ctx'}
    assert 'min' not in Fake.__dict__ and Fake.min is original and sys.getprofile() is previous
    try:
        with observe(runtime, failure, 1, 2) as failed:
            failure(value)
    except RuntimeError as error:
        assert str(error) == 'Constructed producer failure'
    else:
        raise AssertionError('Original producer failure masked')
    assert failed['state'] is None and len(failed['races']) == 1
    assert 'min' not in Fake.__dict__ and Fake.min is original and sys.getprofile() is previous
    Fake.min = original
    try:
        with observe(runtime, body, 1, 2):
            body(value)
        assert Fake.__dict__['min'] is original
    finally:
        delattr(Fake, 'min')
    assert sys.getprofile() is previous
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(fake_runtime_observer='passed', inherited_descriptor='restored', exception='retained',
               existing_descriptor='restored', python_profile='restored', native_contracts='pending'))


if __name__ == '__main__':
    main()
