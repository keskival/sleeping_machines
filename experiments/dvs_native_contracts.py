"""Numerical native packet and actual recovery contracts before the real fit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_native_benchmark as B


def equal(first, second):
    if isinstance(first, torch.Tensor): torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for x, y in zip(first, second): equal(x, y)
    else: assert first == second


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    p.add_argument('--data', required=True); p.add_argument('--controls', required=True); a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    started = time.perf_counter(); torch.set_num_threads(1)
    def args(tag):
        return B.parser().parse_args(['--tag', tag, '--data', a.data, '--controls', a.controls,
            '--fit', '4', '--dev', '2', '--epochs', '2', '--update-targets', '2'])
    config = args('contract'); fitting, _, metadata = B.load(config); row = fitting[0]
    ref = B.make_model(config, False).double(); fast = B.make_model(config).double()
    fast.load_state_dict(ref.state_dict()); outputs = []; gradients = []
    for model in (ref, fast):
        model.zero_grad(); z, _ = B.predict(model, row, 91, True)
        F.cross_entropy(z[None], torch.tensor([row['target']])).backward()
        outputs.append(z.detach()); gradients.append({n: None if v.grad is None else v.grad.clone() for n, v in model.named_parameters()})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-9, atol=1e-10)
    for name in gradients[0]:
        if gradients[0][name] is None: assert gradients[1][name] is None
        else: torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-8, atol=1e-9)
    assert fast.content.weight.grad.norm() > 0 and fast.head.weight.grad.norm() > 0
    assert all(layer.weight.grad is not None and layer.weight.grad.norm() > 0 for depth in fast.queries for layer in depth)
    opt = torch.optim.Adam(fast.parameters(), lr=.003)
    opt.step(); taught, _ = B.predict(fast, row, 81, True); inferred, _ = B.predict(fast, row, 81, False)
    torch.testing.assert_close(taught, inferred, rtol=1e-9, atol=1e-10)
    changed = copy.deepcopy(row); changed.update(target=(row['target'] + 1) % 11, identity='different', index=999)
    same, _ = B.predict(fast, changed, 81, False)
    torch.testing.assert_close(inferred, same, rtol=0, atol=0)
    # Prefix outputs precede the query and must be unaffected by future content.
    def prefix(model, events):
        state = model.new_state(); result = []
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(101); model.eval()
            for t, x in events: result.append(model.consume_event(0, t, x, state)[0])
        return torch.stack(result)
    original = prefix(fast, row['events']); alternate = copy.deepcopy(row['events']); alternate[-1][1][0] += 17
    future = prefix(fast, alternate); torch.testing.assert_close(original[:-1], future[:-1], rtol=0, atol=0)
    with tempfile.TemporaryDirectory(prefix='dvs-native-contract-') as temp:
        continuous = args('continuous'); B.run(continuous, temp)
        recovered = args('recovered'); recovered.stop_after_updates = 1; B.run(recovered, temp)
        recovered.stop_after_updates = None; recovered.resume = True; B.run(recovered, temp)
        x = torch.load(Path(temp) / 'continuous.progress.pt', weights_only=False)
        y = torch.load(Path(temp) / 'recovered.progress.pt', weights_only=False)
        for name in ('online_model', 'optimizer', 'best_state', 'best', 'cursor', 'torch_rng'): equal(x[name], y[name])
        first = json.loads((Path(temp) / 'continuous.json').read_text()); second = json.loads((Path(temp) / 'recovered.json').read_text())
        for name in ('final', 'activity', 'work', 'work_samples', 'selected_epoch'): equal(first[name], second[name])
        assert first['activity']['targets'] == 8 and first['work']['optimizer_updates'] == 4
    names = {**B.sources(), 'experiments/dvs_native_contracts.py': B.sha(Path(__file__))}
    result = dict(status='completed', args=vars(a), contracts_passed=5, full_native_shape=dict(payload=16, depth=2, heads=2, pool=2),
        data=metadata, source_sha256=names, wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Actual real-packet forward/parameter gradients, trained teacher/inference identity, no label/identity/index input, '
              'prefix causality, continuous versus interrupted-driver model/Adam/work recovery. Not gesture quality evidence.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
