"""Fake-runtime callback recovery only; no Torch/gradient contract claimed."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


class Tensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1,)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Tensor(self.values[index[1]])
        return self.values[index]

    def __ge__(self, value):
        return Tensor([x >= value for x in self.values])

    def any(self):
        return any(self.values)

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def new_full(self, shape, value):
        assert shape == (1,)
        return Tensor([value])


class Base:
    fail = False

    @classmethod
    def apply(cls, scores, proposals, noise, force):
        if cls.fail:
            raise RuntimeError('Original fake producer failure')
        return Tensor([1., 2.]), Tensor([.02]), Tensor([0])


class Race(Base):
    pass


def main():
    context = runpy.run_path(str(Path(__file__).with_name('language_route_fidelity.py')))['intervention']
    batch = SimpleNamespace(LaneRace=Race, linear_write_credit=object())
    helper, original = batch.linear_write_credit, Race.apply
    args = Tensor([0.]), Tensor([[1., 2.], [3., 4.]]), Tensor([.2, .5]), Tensor([-1])
    with context(None, batch, 0, delivery=1, commit=1) as trace:
        value, delay, winner = Race.apply(*args)
        assert value.values == [3., 4.] and delay.values == [.02] and winner.values == [1]
        assert trace['winner'] == 0 and trace['committed'] == trace['delivered'] == 1
    assert 'apply' not in Race.__dict__ and Race.apply == original
    assert batch.linear_write_credit is helper and len(trace['noise_tape']) == 1
    with context(None, batch, 0, commit=1) as commit_only:
        value, _, winner = Race.apply(*args)
        assert value.values == [1., 2.] and winner.values == [1]
        assert commit_only['delivered'] == 0 and commit_only['committed'] == 1
    Race.fail = True
    try:
        with context(None, batch, 0):
            Race.apply(*args)
    except RuntimeError as error:
        assert str(error) == 'Original fake producer failure'
    else:
        raise AssertionError('Original failure masked')
    finally:
        Race.fail = False
    assert 'apply' not in Race.__dict__ and batch.linear_write_credit is helper
    Race.apply = staticmethod(original)
    owned = Race.__dict__['apply']
    try:
        with context(None, batch, 0):
            Race.apply(*args)
        assert Race.__dict__['apply'] is owned
    finally:
        delattr(Race, 'apply')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(dict(fake_callback_recovery='passed', inherited_apply='restored',
               owned_apply='restored', producer_exception='retained', native_contracts='pending'))


if __name__ == '__main__':
    main()
