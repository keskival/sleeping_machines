import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, 'experiments')
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_commit_credit_audit as A  # noqa: E402


def test_instrumented_step_reproduces_the_reference_forward():
    m = C.make_model(SimpleNamespace(payload=8, depth=2, heads=2, pool=2, clock_step=.05, seed=6), fast=False).double()
    events = [((k + 1) * .05, torch.randn(33, dtype=torch.float64, generator=torch.Generator().manual_seed(k))) for k in range(12)]
    m.train()
    with torch.random.fork_rng():
        torch.manual_seed(3); s = m.new_state(); ref = [m.consume_event(0, t, x, s)[0] for t, x in events]
    with torch.random.fork_rng():
        torch.manual_seed(3); s = m.new_state(); log = []; ins = [A.instrumented_step(m, 0, t, x, s, log) for t, x in events]
    torch.testing.assert_close(torch.stack(ins), torch.stack(ref), rtol=0, atol=1e-12)
    races = [r for r in log if 'scores' in r]
    assert len(races) == 12 * 2 * 2 and sum('aliases' in r for r in log) == 12
    torch.nn.functional.cross_entropy(ins[-1][None], torch.tensor([1])).backward()
    assert any(a.grad is not None and a.grad.abs().sum() > 0 for e in log if 'aliases' in e for a in e['aliases'].values())
