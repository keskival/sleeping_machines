"""Existing real-packet/recovery contracts at the calibrated initial clocks."""
from pathlib import Path
import copy
import json
import sys
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C
import dvs_native_contracts as contracts


if __name__ == '__main__':
    started = time.perf_counter()
    args = SimpleNamespace(seed=6, payload=16, depth=2, heads=2, pool=2)
    original = C.BASE_FACTORY(args, False).double()
    calibrated = C.scale_initial_clocks(copy.deepcopy(original), .05)
    rates = []; frequencies = []
    for name, parameter in original.named_parameters():
        other = dict(calibrated.named_parameters())[name]
        if name.endswith('raw_rate') or name == 'transport_rate':
            torch.testing.assert_close(F.softplus(other) + 1e-6, 20 * (F.softplus(parameter) + 1e-6), rtol=1e-12, atol=1e-12)
            rates.append(name)
        elif name.endswith('frequency') or name == 'transport_frequency':
            torch.testing.assert_close(other, 20 * parameter, rtol=0, atol=0)
            frequencies.append(name)
        else:
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
    value = torch.linspace(-1, 1, 16, dtype=torch.float64)
    age = torch.tensor(.05, dtype=torch.float64)
    torch.testing.assert_close(calibrated.transport(value, age, 0, 0),
        original.transport(value, age / .05, 0, 0), rtol=1e-12, atol=1e-12)
    with C.activate():
        contracts.main()
    tag = sys.argv[sys.argv.index('--tag') + 1]
    out = ROOT / 'experiments/results/diagnostics' / (tag + '.json')
    result = json.loads(out.read_text())
    result.update(contracts_passed=6, clock_step_seconds=.05,
        clock_rate_and_rotation_algebra_verified=True,
        non_temporal_initial_parameters_preserved=True,
        wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
