"""Frozen, guarded native SHD contracts/smoke/pilot. No official test access.

CLI accepts only a frozen manifest/stage and exact recovery. Admission checks
precede numerical imports. Owner-host run_safe reservation is required; this
driver is not a standalone host admission mechanism.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from public_speech_packets import CONTENT_DIM, file_sha, load_rows


def require_guard_environment(cfg):
    """Reject default/weakened guard settings and direct Python before imports."""
    required = {'MIN_AVAIL_MB': cfg['min_available_mb'], 'JOB_TIMEOUT_S': cfg['timeout_s'],
                'MEM_CAP_RSS_KB': cfg['rss_cap_kb'], 'MEM_CAP_KB': cfg['vms_cap_kb']}
    if any(os.environ.get(name) != str(value) for name, value in required.items()):
        raise ValueError('Explicit frozen run_safe memory/timeout settings required')
    if any(os.environ.get(name) != '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                   'OPENBLAS_NUM_THREADS', 'TORCH_NUM_THREADS')):
        raise ValueError('Guarded one-thread environment required')
    expected = '/tmp/experiments-runner.lock'
    if os.environ.get('AWS_GYM_SLOT'):
        import socket
        slot = os.environ['AWS_GYM_SLOT']
        if slot not in ('1', '2', '3') or socket.gethostname() != 'ip-172-31-47-132':
            raise ValueError('Bounded slots belong only to the explicitly authorized AWS host')
        expected = '/tmp/experiments-runner.aws-gym-slot' + slot + '.lock'
    if os.readlink('/proc/self/fd/9') != expected:
        raise ValueError('Inherited physical run_safe reservation required')
    import fcntl
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)


def preflight(manifest_path, stage, expected_manifest_sha=None):
    if expected_manifest_sha is not None and file_sha(manifest_path) != expected_manifest_sha:
        raise ValueError('Changed frozen manifest; changed settings require a new queue/tag')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['status'] != 'prepared_unrun' or stage not in ('contracts', 'smoke', 'pilot'):
        raise ValueError('Frozen admission stage required')
    for name, expected in manifest['source_sha256'].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError('Changed frozen source: ' + name)
    cfg = manifest['stages'][stage]
    if cfg['tag'] != Path(cfg['tag']).name:
        raise ValueError('Unique plain tag required')
    for prior in cfg['requires']:
        parent_cfg = manifest['stages'][prior]
        record = json.loads((ROOT / parent_cfg['output']).read_text())
        if (record['status'] != 'completed' or record['stage'] != prior or
                record['source_sha256'] != manifest['source_sha256'] or
                record['config'] != parent_cfg or
                record['manifest_sha256'] != file_sha(manifest_path)):
            raise ValueError('Completed exact-protocol prerequisite required')
        if prior == 'smoke':
            if not record['learning_finite'] or not record['formula_coverage_complete']:
                raise ValueError('Smoke must have finite learning and complete work coverage')
            windows = math.ceil(cfg['fit'] / cfg['lanes']) * cfg['epochs']
            projected = windows * record['max_window_wall_s'] * 2 + record['wall_s'] * 2 + 120
            if projected > cfg['timeout_s']:
                raise ValueError('Measured smoke projects beyond pilot timeout; revise unique queue')
            if record['max_rss_kb'] * 1.5 > cfg['rss_cap_kb']:
                raise ValueError('Measured smoke needs a larger safe RSS reservation')
    if stage != 'contracts':
        data = ROOT / manifest['data']['path']
        if data.name != 'shd_train.h5' or file_sha(data) != manifest['data']['sha256']:
            raise ValueError('Changed official TRAIN file; reject before numerical imports')
    return manifest, cfg


def model_for(cfg, dtype=None):
    torch.manual_seed(cfg['seed'])
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=CONTENT_DIM, classes=20,
        payload=cfg['payload'], depth=cfg['depth'], heads=cfg['heads'], pool=cfg['pool'])
    return model.to(dtype=dtype) if dtype is not None else model


def forward(model, rows, seed, mode, credit):
    fn = compiled_logits if mode == 'compiled' else batched_logits
    return fn(model, rows, seed, route_credit=credit)


def update(model, optimizer, rows, cfg, seed, trace=False):
    optimizer.zero_grad(set_to_none=True)
    stages, box = {}, {}
    def call(name, fn):
        if trace:
            stages[name] = capture(fn)
        else:
            fn()
    model.train()
    def objective():
        # Operator accounting runs the contracted eager mathematical program;
        # opaque fused compiler kernels cannot supply a complete ATen ledger.
        logits = forward(model, rows, seed, 'eager' if trace else cfg['backend'], cfg['credit'])
        box['loss'] = F.cross_entropy(logits, torch.tensor([r['target'] for r in rows]))
        if not torch.isfinite(box['loss']):
            raise FloatingPointError('Nonfinite SHD objective')
    call('forward_and_mean_loss', objective)
    call('backward', lambda: box['loss'].backward())
    call('clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'], error_if_nonfinite=True))
    call('adam', optimizer.step)
    if trace and not all(s['formula_coverage_complete'] for s in stages.values()):
        raise ValueError('Incomplete full-step arithmetic coverage')
    return float(box['loss'].detach()), stages


def evaluate(model, rows, cfg):
    model.eval()
    losses, predictions, labels = [], [], []
    with torch.no_grad():
        for begin in range(0, len(rows), cfg['lanes']):
            batch = rows[begin:begin + cfg['lanes']]
            logits = forward(model, batch, 314159, cfg['backend'], None)
            y = torch.tensor([r['target'] for r in batch])
            losses.extend(F.cross_entropy(logits, y, reduction='none').tolist())
            predictions.extend(logits.argmax(-1).tolist())
            labels.extend(y.tolist())
    return dict(targets=len(rows), nll=sum(losses) / len(losses),
        accuracy=sum(a == b for a, b in zip(predictions, labels)) / len(labels),
        per_target_nll=losses, predictions=predictions, labels=labels)


def equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    else:
        assert first == second


def train(cfg, fit, dev, checkpoint, resume=False, stop_updates=None, trace=True, identity=None,
          attempt_started=None):
    """Actual pilot loop, also used in the save/recovery contract."""
    attempt_started = time.perf_counter() if attempt_started is None else attempt_started
    prior_wall = 0.
    prior_peak_rss = 0
    model = model_for(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    cursor = dict(epoch=0, window=0, updates=0)
    result = dict(curve=[], work_samples=[], window_size_counts={}, window_wall_s=[],
                  initial_fit=evaluate(model, fit, cfg), initial_dev=evaluate(model, dev, cfg),
                  fit_queries=0, fit_packets=0, fit_padded_packets=0, fit_raw_spikes=0)
    best, best_state = math.inf, None
    if resume:
        saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
        equal(saved['config'], cfg)
        equal(saved['identity'], identity)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        cursor, result, best, best_state = saved['cursor'], saved['result'], saved['best'], saved['best_state']
        prior_wall = saved['wall_s']
        prior_peak_rss = saved['max_rss_kb']
        torch.set_rng_state(saved['rng'])
    while cursor['epoch'] < cfg['epochs']:
        order = list(range(len(fit)))
        random.Random(cfg['seed'] + 1000 + cursor['epoch']).shuffle(order)
        batches = [order[i:i + cfg['lanes']] for i in range(0, len(order), cfg['lanes'])]
        while cursor['window'] < len(batches):
            indices = batches[cursor['window']]
            batch = [fit[i] for i in indices]
            use_trace = trace and not any(s['targets'] == len(batch) for s in result['work_samples'])
            started = time.perf_counter()
            loss, stages = update(model, optimizer, batch, cfg,
                100000 + cfg['seed'] * 1000 + cursor['updates'], use_trace)
            result['window_wall_s'].append(time.perf_counter() - started)
            if use_trace:
                result['work_samples'].append(dict(targets=len(batch), stages=stages))
            size = str(len(batch))
            result['window_size_counts'][size] = result['window_size_counts'].get(size, 0) + 1
            result['fit_queries'] += len(batch)
            result['fit_packets'] += sum(len(r['events']) for r in batch)
            result['fit_padded_packets'] += len(batch) * max(len(r['events']) for r in batch)
            result['fit_raw_spikes'] += sum(r['raw_spikes'] for r in batch)
            cursor['window'] += 1
            cursor['updates'] += 1
            if cursor['window'] == len(batches):
                score = evaluate(model, dev, cfg)
                result['curve'].append(dict(epoch=cursor['epoch'] + 1, dev=score))
                if score['nll'] < best:
                    best, best_state = score['nll'], copy.deepcopy(model.state_dict())
                    result['selected_epoch'] = cursor['epoch'] + 1
                cursor.update(epoch=cursor['epoch'] + 1, window=0)
            saved = dict(config=cfg, identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                cursor=cursor, result=result, best=best, best_state=best_state, rng=torch.get_rng_state(),
                wall_s=prior_wall + time.perf_counter() - attempt_started,
                max_rss_kb=max(prior_peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(saved, temporary)
            temporary.replace(checkpoint)
            if stop_updates is not None and cursor['updates'] >= stop_updates:
                return result, saved
            if cursor['window'] == 0:
                break
    model.load_state_dict(best_state)
    result['final_dev'] = evaluate(model, dev, cfg)
    result['final_fit'] = evaluate(model, fit, cfg)
    result['parameters'] = sum(p.numel() for p in model.parameters())
    result['prior_attempt_wall_s'] = prior_wall
    result['prior_peak_rss_kb'] = prior_peak_rss
    return result, saved


def contracts(cfg):
    from public_speech_packets import packetize
    rows = [dict(index=j, target=j, raw_spikes=3,
        events=packetize([0, 7000, 17000], [j, 699, j], 16000)) for j in range(3)]
    comparisons = []
    for dtype, atol, rtol in ((torch.float64, 1e-9, 1e-8), (torch.float32, 2e-4, 2e-4)):
        first, second = model_for(cfg, dtype), model_for(cfg, dtype)
        first.train(); second.train()
        z = forward(first, rows, 99, 'eager', 'linear')
        other = forward(second, rows, 99, 'compiled', 'linear')
        torch.testing.assert_close(z, other, atol=atol, rtol=rtol)
        targets = torch.tensor([r['target'] for r in rows])
        F.cross_entropy(z, targets).backward()
        F.cross_entropy(other, targets).backward()
        count = 0
        for (name, a), (other_name, b) in zip(first.named_parameters(), second.named_parameters()):
            assert name == other_name and (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=atol, rtol=rtol)
                if not torch.isfinite(a.grad).all():
                    raise ValueError('Nonfinite contracted gradient')
                count += 1
        with torch.no_grad():
            none = forward(first, rows, 99, 'eager', None)
            linear = forward(first, rows, 99, 'eager', 'linear')
            torch.testing.assert_close(none, linear, atol=0, rtol=0)
        comparisons.append(dict(dtype=str(dtype), every_parameter_gradient=True, gradient_tensors=count))
    recoveries = []
    with tempfile.TemporaryDirectory(prefix='public-speech-contract-') as temp:
        for backend in ('eager', 'compiled'):
            small = dict(cfg, epochs=2, lanes=2, backend=backend)
            continuous_path = Path(temp) / (backend + '_continuous.pt')
            resume_path = Path(temp) / (backend + '_recovery.pt')
            continuous, original = train(small, rows, rows[:2], continuous_path, trace=False)
            train(small, rows, rows[:2], resume_path, stop_updates=1, trace=False)
            recovered, resumed = train(small, rows, rows[:2], resume_path, resume=True, trace=False)
            for name in ('model', 'optimizer', 'best_state', 'cursor', 'rng'):
                equal(original[name], resumed[name])
            for name in ('curve', 'final_dev', 'final_fit', 'fit_queries', 'selected_epoch'):
                equal(continuous[name], recovered[name])
            recoveries.append(backend)
    return dict(comparisons=comparisons, linear_credit_bitwise_forward_invariance=True,
                actual_driver_adam_and_rng_recovery=recoveries, synthetic_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--stage', required=True, choices=('contracts', 'smoke', 'pilot'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest, cfg = preflight(args.manifest, args.stage, args.manifest_sha256)
    if manifest['admission']['reject_unshared_container'] and Path('/.dockerenv').exists():
        raise ValueError('Workspace container has no physical-host reservation; execute on the owning host')
    require_guard_environment(cfg)
    out = ROOT / cfg['output']
    if out.exists():
        raise ValueError('Preserve completed result')
    checkpoint = out.with_suffix('.progress.pt')
    if args.stage == 'contracts' and args.resume:
        raise ValueError('Contracts do not have a recovery checkpoint')
    if checkpoint.exists() != args.resume:
        raise ValueError('Explicit exact recovery required iff a checkpoint exists')
    # All numerical imports are below admission checks. No model execution in
    # the workspace container is authorized by a free container-local lock.
    global torch, F, fast_class, AddressedEventHeads, batched_logits, compiled_logits, capture
    import torch
    from torch.nn import functional as F
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.compiled_episodes import compiled_logits
    from race_language_screen import capture
    from torch._inductor import config as inductor_config
    from torch._dynamo import config as dynamo_config
    torch.set_num_threads(1)
    inductor_config.compile_threads = 1
    dynamo_config.cache_size_limit = 64
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = dict(stage=args.stage, config=cfg, source_sha256=manifest['source_sha256'],
        manifest_sha256=file_sha(args.manifest), status='running',
        hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1, device='cpu'),
        official_test_read=False, supremacy_claim=False)
    if args.stage == 'contracts':
        result['contracts'] = contracts(cfg)
    else:
        fit, dev, data = load_rows(ROOT / manifest['data']['path'], cfg['fit'], cfg['dev'],
                                   cfg['packet_us'], manifest['data']['sha256'])
        identity = dict(manifest_sha256=args.manifest_sha256, source_sha256=manifest['source_sha256'], data=data)
        values, saved = train(cfg, fit, dev, checkpoint, args.resume, identity=identity, attempt_started=started)
        result.update(values, data=data)
        result['class_coverage'] = {name: {str(label): sum(r['target'] == label for r in rows)
            for label in range(20)} for name, rows in (('fit', fit), ('dev', dev))}
        total = 0.
        for size, count in values['window_size_counts'].items():
            sample = next(s for s in values['work_samples'] if s['targets'] == int(size))
            total += count * sum(s['arithmetic_flops'] + s['special_function_evaluations']
                                 for s in sample['stages'].values())
        packets, races = values['fit_packets'], values['fit_packets'] * cfg['depth'] * cfg['heads']
        padded_races = values['fit_padded_packets'] * cfg['depth'] * cfg['heads']
        result['work'] = dict(whole_fit_unit_special_flops_estimate=total,
            fit_flops_per_query_estimate=total / values['fit_queries'], raw_spikes=values['fit_raw_spikes'],
            emitted_packets_and_queries=packets, padded_computed_packet_positions=values['fit_padded_packets'],
            key_scores=races * cfg['pool'], padded_key_scores=padded_races * cfg['pool'],
            selected_updates=races, candidate_value_proposals=races * cfg['pool'],
            padded_candidate_value_proposals=padded_races * cfg['pool'],
            available_receivers=cfg['depth'] * cfg['heads'] * cfg['pool'],
            inference_work=None, energy_joules=None,
            scope='All-proposal training emulator INCLUDING padding and mean-loss/backward/clip/Adam; first EAGER window of each size extrapolated to contracted compiled steps. '
                  'HDF5 decoding, packet preprocessing, serialization, integer/RNG/traffic work excluded from arithmetic, included in wall. '
                  'Inference/energy remain unmeasured; proposal maps and all scored keys are not free.')
        result['learning_finite'] = all(math.isfinite(v) for v in (values['final_dev']['nll'], values['final_fit']['nll']))
        result['small_fit_learning_passed'] = values['final_fit']['nll'] < values['initial_fit']['nll'] - .01
        result['formula_coverage_complete'] = all(s['formula_coverage_complete']
            for sample in values['work_samples'] for s in sample['stages'].values())
        result['max_window_wall_s'] = max(values['window_wall_s'])
        result['checkpoint'] = str(checkpoint.relative_to(ROOT))
        result['checkpoint_sha256'] = file_sha(checkpoint)
        result['selected_weights_field'] = 'best_state'
        result['recovery'] = dict(resumed=args.resume, recorded_attempt_wall_accumulated=True,
            retained_updates_counted=True, uncheckpointed_discarded_work_unknown=args.resume,
            scope='On interruption, work/time after the last durable snapshot is unknown and excluded')
        result['scope'] = 'Native speech admission only; small reused train-speaker DEV, no official test or competitive win'
    result.update(status='completed', wall_s=result.get('prior_attempt_wall_s', 0.) + time.perf_counter() - started,
                  max_rss_kb=max(result.get('prior_peak_rss_kb', 0), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('stage', 'status', 'wall_s', 'max_rss_kb')}), flush=True)


if __name__ == '__main__':
    main()
