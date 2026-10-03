"""Read-only saved-weight/Adam-moment summaries; no forward, backward or steps."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch


def summary(x):
    x = x.detach().double().flatten()
    assert bool(torch.isfinite(x).all())
    return dict(count=x.numel(), minimum=float(x.min()), median=float(x.median()),
                maximum=float(x.max()), l2=float(x.norm()))


def analyze(path):
    saved = torch.load(path, weights_only=False)
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['checkpoint_sha256']
    assert saved['total_targets'] == metadata['trained_targets']
    unique, seen, aliases = [], {}, {}
    for name, value in saved['model'].items():
        key = (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
            unique.append((name, value))
    groups = saved['optimizer']['param_groups']
    assert len(groups) == 1 and not groups[0]['amsgrad'] and groups[0]['weight_decay'] == 0
    group = groups[0]
    # These frozen models have parameters only (no registered buffers); PyTorch
    # state_dict registration order, storage alias dedup, and Adam list coincide.
    assert len(unique) == len(group['params'])
    rows = []
    for (name, value), identifier in zip(unique, group['params']):
        if '.gate.' not in name and '.output.' not in name:
            continue
        state = saved['optimizer']['state'].get(identifier)
        if state is None:
            rows.append(dict(parameter=name, moments_present=False))
            continue
        assert state['exp_avg'].shape == value.shape == state['exp_avg_sq'].shape
        step = int(state['step'])
        assert step == saved['updates']
        b1, b2 = group['betas']
        m = state['exp_avg'].double() / (1 - b1**step)
        root_v = (state['exp_avg_sq'].double() / (1 - b2**step)).sqrt()
        epsilon = group['eps']
        direction = m / (root_v + epsilon)
        row = dict(parameter=name, moments_present=True, weight=summary(value),
            bias_corrected_rms_gradient=summary(root_v),
            epsilon_dominated_coordinates=int((root_v <= epsilon).sum()),
            epsilon_retention=summary(root_v / (root_v + epsilon)),
            stored_moment_direction=summary(direction),
            lr_times_stored_direction_norm=float(group['lr'] * direction.norm()))
        if name.endswith('gate.bias'):
            row['sigmoid_bias_only'] = summary(value.sigmoid())
        rows.append(row)
    return dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=metadata['checkpoint_sha256'],
        targets=saved['total_targets'], updates=saved['updates'], family=metadata['args']['receiver_sharing'],
        unique_parameters=len(unique), aliases=aliases, rows=rows)


if __name__ == '__main__':
    torch.set_num_threads(1)
    paths=[ROOT / 'experiments/results/aws_language_progress' /
           f'aws_depth8_language_20261002T234100Z_{family}_pilot_s7_milestone001.pt'
           for family in ('private', 'depth')]
    print(json.dumps(dict(scope='Read-only immutable 250k teacher checkpoints. No new training or data reads. Sigmoid bias is NOT the input-dependent gate. Stored-moment direction is NOT a prospective or actual functional optimizer update. No replay-vs-teacher quality comparison.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        families=[analyze(p) for p in paths]), indent=2, allow_nan=False))
