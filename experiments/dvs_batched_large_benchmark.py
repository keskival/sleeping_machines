"""Large batched DVS program driver: regularization, clip, coarse packets and tied pools on the exact batched path.

Composes dvs_batched_reg_benchmark (weight decay, input noise, clip, coarse bins; batched training/evaluation, §§405-408)
with tied receiver maps (§398 capacity-exposure: maps shared across each pool; keys, clock biases, timescales and memories
private).  --tie-pools applies dvs_tied_pool_benchmark.tie_pools to the model; all other options pass through.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_batched_le_benchmark as BL  # noqa: E402
import dvs_batched_reg_benchmark as RG  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
import dvs_tied_pool_benchmark as T  # noqa: E402


def parser():
    p = RG.parser(); p.add_argument('--tie-pools', action='store_true')
    return p


def make_model(a, fast=True):
    model = BL.make_model(a)
    return T.tie_pools(model) if a.tie_pools else model


def sources():
    return {**RG.sources(), 'experiments/dvs_tied_pool_benchmark.py': N.sha(ROOT / 'experiments/dvs_tied_pool_benchmark.py'),
            'experiments/dvs_batched_large_benchmark.py': N.sha(ROOT / 'experiments/dvs_batched_large_benchmark.py')}


if __name__ == '__main__':
    import json
    args = parser().parse_args()
    with RG.activate(args):
        N.make_model, N.parser, N.sources = make_model, parser, sources
        N.run(args)
    out = ROOT / 'experiments/results/dvs_native' / (args.tag + '.json')
    result = json.loads(out.read_text())
    if result.get('status') == 'completed':
        result['regularization'] = dict(weight_decay=args.weight_decay, input_noise=args.input_noise, bins=args.bins,
                                        clip=args.clip, tie_pools=args.tie_pools)
        out.write_text(json.dumps(result, indent=2) + '\n')
