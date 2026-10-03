"""Preserve the actual per-layer residual gains when growing a DVS model.

The frozen legacy driver stores gain as a Python float and resets every old
layer to .5/sqrt(parent_depth).  Its first D2->D4 growth is correct; another
growth changes the older layers' gains.  This sibling reconstructs the actual
parent construction, validates its checkpoint, and changes only that reset.
Closed gates, temporal clocks, sparse writes, channel mixing and replay credit
are retained.  Fresh Adam and the extra race-noise draws remain unchanged.
"""
from contextlib import contextmanager
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_batched_le_benchmark as BL
import dvs_clock_calibrated_benchmark as C
import dvs_grow_depth_benchmark as G
import dvs_native_benchmark as N

SELF = 'experiments/dvs_grow_depth_lineage_benchmark.py'
LEGACY = 'experiments/dvs_grow_depth_benchmark.py'
NOTE = 'experiments/theory/121_depth_growth_lineage_contract.md'
BASE_FORWARD_SOURCE_NAMES = tuple(N.sources())


def canonical(path, root=ROOT):
    path = (Path(root) / path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def layer_gains(model):
    gains = []
    for depth in model.units:
        values = [float(u.gain) for head in depth for pool in head for u in pool]
        if not values or any(v != values[0] for v in values):
            raise ValueError('A layer must have one declared stacked residual gain')
        gains.append(values[0])
    return gains


def checked_gains(values, depth):
    if (not isinstance(values, list) or len(values) != depth or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError('Invalid per-layer residual gains')
    return [float(v) for v in values]


def metadata(parent, depth):
    parent_depth = parent['depth']
    if depth <= parent_depth:
        raise ValueError('Growth requires strictly greater depth')
    return dict(version=1, construction='preserve_actual_parent_layer_gains',
        parent_result=parent['path'], parent_result_sha256=parent['result_sha256'],
        parent_checkpoint=parent['checkpoint_path'],
        parent_checkpoint_sha256=parent['checkpoint_sha256'],
        parent_depth=parent_depth, depth=depth,
        parent_layer_gains=list(parent['layer_gains']),
        layer_gains=list(parent['layer_gains']) + [.5 / math.sqrt(depth)] * (depth - parent_depth),
        ancestors=copy.deepcopy(parent['ancestors']),
        optimizer_ownership='Fresh Adam, matching the legacy growth protocol; parent moments are not restored',
        mechanisms='Legacy closed gates, inherited top transport, temporal races, private sparse writes and channel mixing retained',
        limits='Extra races change the parent noise stream; closed-gate plasticity and exact function preservation are not established')


def resolve_parent(path, root=ROOT, active=None):
    """Read and validate the *actual* ordinary, legacy-grown or new parent.

    Legacy progressive parents are reconstructed with their historical reset,
    not retroactively assigned the desired gains.  Every ancestor is bound to
    immutable result/checkpoint bytes.  Unknown constructors are rejected.
    """
    path = canonical(path, root); full = Path(root) / path
    active = set() if active is None else set(active)
    if path in active:
        raise ValueError('Cyclic growth lineage')
    active.add(path)
    if full.suffix != '.json' or not full.is_file():
        raise ValueError('A completed parent result is required')
    result = json.loads(full.read_text())
    if result.get('status') != 'completed':
        raise ValueError('Parent is not completed')
    args = result.get('args', {}); depth = args.get('depth')
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError('Invalid parent depth')
    source = result.get('source_sha256', {})
    # Forward and factory files must be the recognized frozen implementation.
    # Unrelated historical report/regularization-wrapper hashes do not identify
    # a residual-gain construction and are not used to infer one.
    required = set(BASE_FORWARD_SOURCE_NAMES) | {'experiments/dvs_clock_calibrated_benchmark.py'}
    for name in required:
        target = ROOT / name
        if name not in source or not target.is_file() or N.sha(target) != source[name]:
            raise ValueError('Unresolved parent model source: ' + name)
    is_new = SELF in source
    is_legacy = LEGACY in source and not is_new
    if is_new:
        if source[SELF] != N.sha(ROOT / SELF) or NOTE not in source or source[NOTE] != N.sha(ROOT / NOTE):
            raise ValueError('Changed lineage constructor')
    elif is_legacy and source[LEGACY] != N.sha(ROOT / LEGACY):
        raise ValueError('Changed legacy growth constructor')
    # These other wrappers change representation or sharing; accepting them as
    # ordinary factory parents would guess unrecorded semantics.
    known_wrappers = {'experiments/dvs_batched_benchmark.py',
        'experiments/dvs_batched_le_benchmark.py', 'experiments/dvs_batched_reg_benchmark.py',
        'experiments/dvs_grow_depth_benchmark.py', 'experiments/dvs_batched_grow_benchmark.py', SELF}
    unknown = [name for name in source if name.startswith('experiments/dvs_') and
        name.endswith('_benchmark.py') and name not in required and name not in known_wrappers]
    unknown += [name for name in source if name.startswith('sleeping_machines/') and
        name not in required and name != 'sleeping_machines/batched_episodes.py']
    if unknown:
        raise ValueError('Unsupported parent constructor: ' + ', '.join(unknown))
    ancestor = None
    if is_new or is_legacy:
        if not args.get('parent'):
            raise ValueError('Missing growth lineage')
        ancestor = resolve_parent(args['parent'], root, active)
        for key in ('payload', 'heads', 'pool', 'clock_step'):
            if args.get(key) != ancestor['args'].get(key):
                raise ValueError('Ancestor dimension/clock mismatch: ' + key)
        if depth <= ancestor['depth']:
            raise ValueError('Nonincreasing growth lineage')
        if is_new:
            info = metadata(ancestor, depth)
            if result.get('growth_lineage') != info:
                raise ValueError('Unresolved or mismatched persisted growth metadata')
            gains = checked_gains(info['layer_gains'], depth)
            kind = 'lineage_preserving_growth'
        else:
            # This is what the frozen legacy driver actually did, even for an
            # already-grown ancestor.  Preserve the evidence of that reset.
            gains = [.5 / ancestor['depth'] ** .5] * ancestor['depth']
            gains += [.5 / math.sqrt(depth)] * (depth - ancestor['depth'])
            kind = 'legacy_uniform_old_gain_reset'
    else:
        if args.get('parent') or 'growth_lineage' in result:
            raise ValueError('Unrecognized parent lineage')
        gains = [.5 / math.sqrt(depth)] * depth
        kind = 'ordinary_clock_calibrated_factory'
    checkpoint = full.with_suffix('.progress.pt')
    if not checkpoint.is_file():
        raise ValueError('Missing parent checkpoint')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    settings = {k: v for k, v in args.items() if k not in ('tag', 'resume', 'stop_after_updates')}
    if (ck.get('source_sha256') != source or ck.get('data') != result.get('data') or
            ck.get('settings') != settings or ck.get('best_state') is None):
        raise ValueError('Mismatched parent checkpoint protocol')
    if ck.get('result', {}).get('selected_epoch') != result.get('selected_epoch'):
        raise ValueError('Mismatched selected parent checkpoint')
    if ck.get('best') != result.get('final', {}).get('nll'):
        raise ValueError('Mismatched selected parent loss')
    if is_new and (ck.get('growth_lineage') != info or ck.get('result', {}).get('growth_lineage') != info):
        raise ValueError('Checkpoint does not preserve declared layer gains')
    with torch.random.fork_rng():
        expected = C.make_model(argparse.Namespace(**args), True).state_dict()
    state = ck['best_state']
    if state.keys() != expected.keys() or any(
            not isinstance(state[k], torch.Tensor) or state[k].shape != expected[k].shape or
            state[k].dtype != expected[k].dtype or not bool(torch.isfinite(state[k]).all()) for k in expected):
        raise ValueError('Mismatched parent state structure/dtype or nonfinite state')
    checkpoint_path = canonical(checkpoint, root)
    result_sha, checkpoint_sha = N.sha(full), N.sha(checkpoint)
    node = dict(result=path, result_sha256=result_sha, checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha, kind=kind, depth=depth, layer_gains=gains)
    bindings = {} if ancestor is None else dict(ancestor['bindings'])
    bindings.update({str(full): result_sha, str(checkpoint): checkpoint_sha})
    return dict(path=path, checkpoint_path=checkpoint_path, args=args, depth=depth,
        layer_gains=gains, state=state, result_sha256=result_sha,
        checkpoint_sha256=checkpoint_sha, bindings=bindings,
        ancestors=[node] + ([] if ancestor is None else ancestor['ancestors']))


def grow(model, parent_state, parent_gains):
    parent_gains = checked_gains(parent_gains, len(parent_gains))
    if len(parent_gains) >= model.depth:
        raise ValueError('Growth requires strictly greater depth')
    G.grow(model, parent_state, len(parent_gains))
    for depth, gain in enumerate(parent_gains):
        for head in model.units[depth]:
            for pool in head:
                for unit in pool:
                    unit.gain = gain
    return model


def parser():
    p = BL.parser(); p.add_argument('--parent', required=True)
    return p


def make_model(a, fast=True):
    parent = resolve_parent(a.parent)
    for key in ('payload', 'heads', 'pool', 'clock_step'):
        if parent['args'].get(key) != getattr(a, key):
            raise ValueError('Parent dimension/clock mismatch: ' + key)
    info = metadata(parent, a.depth)
    model = grow(C.make_model(a, fast), parent['state'], parent['layer_gains'])
    if layer_gains(model) != info['layer_gains']:
        raise ValueError('Constructed gains differ from the declared lineage')
    model._depth_growth_lineage = info
    return model


def sources(parent=None):
    names = [SELF, NOTE, 'experiments/dvs_grow_depth_lineage_contracts.py']
    result = {**BL.sources(), **G.sources(), **{n: N.sha(ROOT / n) for n in names}}
    if parent is not None:
        result.update(parent['bindings'])
    return result


@contextmanager
def activate(a):
    parent = resolve_parent(a.parent)
    info = metadata(parent, a.depth)
    old = N.make_model, N.parser, N.sources, N.train_window, N.evaluate, torch.save
    def factory(settings, fast=True):
        model = make_model(settings, fast)
        if model._depth_growth_lineage != info:
            raise ValueError('Parent lineage changed after admission')
        return model
    checkpoint_temp = (ROOT / 'experiments/results/dvs_native' / (a.tag + '.progress.tmp')).resolve()
    def save(payload, destination, *positional, **keywords):
        # Annotate each recovery snapshot, including a crash before N.run
        # returns. Only this run's exact checkpoint destination is intercepted.
        if (isinstance(destination, (str, Path)) and Path(destination).resolve() == checkpoint_temp and
                isinstance(payload, dict) and SELF in payload.get('source_sha256', {}) and
                'settings' in payload and 'cursor' in payload and 'result' in payload):
            payload['growth_lineage'] = copy.deepcopy(info)
            payload['result']['growth_lineage'] = copy.deepcopy(info)
        return old[-1](payload, destination, *positional, **keywords)
    N.make_model, N.parser, N.sources = factory, parser, lambda: sources(parent)
    N.train_window, N.evaluate = BL.train_window, BL.evaluate
    torch.save = save
    try:
        yield info
    finally:
        N.make_model, N.parser, N.sources, N.train_window, N.evaluate, torch.save = old
        if any(N.sha(path) != digest for path, digest in parent['bindings'].items()):
            raise ValueError('An admitted parent artifact changed during the run')


def run(a):
    with activate(a) as info:
        if a.resume:
            checkpoint = ROOT / 'experiments/results/dvs_native' / (a.tag + '.progress.pt')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved.get('growth_lineage') != info or saved.get('result', {}).get('growth_lineage') != info:
                raise ValueError('Recovery checkpoint lost its declared residual gains')
        result = N.run(a)
    # Every snapshot already carries the non-tensor semantics, including one
    # written before an interruption. Verify the returned/persisted artifact.
    directory = ROOT / 'experiments/results/dvs_native'
    checkpoint = directory / (a.tag + '.progress.pt')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('growth_lineage') != info or ck['result'].get('growth_lineage') != info or result.get('growth_lineage') != info:
        raise ValueError('Growth metadata was not persisted')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
