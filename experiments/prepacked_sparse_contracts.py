"""Prepared immutable-weight inference admission; guarded queue only.

Help and producer/weight admission use stdlib only. Native contracts are
pending, including trained pilot float32 and promoted-float64 trajectories.
No fit, optimizer update, heldout rescore or speed/energy benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    start = time.perf_counter()
    parent_path, output = ROOT/args.result, ROOT/args.out
    if output.exists():
        raise ValueError('Preserve existing output')
    parent = json.loads(parent_path.read_text())
    if parent.get('status') != 'completed' or not parent.get('final_weights'):
        raise ValueError('Completed fit/admission pilot with actual checkpoint required')
    checkpoint = ROOT/parent['final_weights']
    if not checkpoint.is_file():
        raise ValueError('Actual weights required before runtime imports')
    config = parent['args']
    if config.get('skip_init_from', 0) or config.get('tie_pools', False):
        raise ValueError('Current producer admission covers nongrown, untied models')
    producer = ['sleeping_machines/batched_episodes.py', 'sleeping_machines/fast_native_core.py',
                'sleeping_machines/addressed_event_heads.py']
    for name in producer:
        if sha(ROOT/name) != parent['source_sha256'][name]:
            raise ValueError('Changed numerical producer/factory: '+name)
    names = producer+['sleeping_machines/prepacked_sparse_inference.py',
                      'sleeping_machines/sparse_inference.py', 'sleeping_machines/parallel_stream_language.py',
                      'experiments/prepacked_sparse_contracts.py', 'experiments/cached_inference_contracts.py',
                      'experiments/sparse_inference_accounting.py', 'experiments/language_batched_benchmark.py',
                      'experiments/e120_shared_tasks.py']
    sources = {name: sha(ROOT/name) for name in names}
    parent_hash, checkpoint_hash = sha(parent_path), sha(checkpoint)
    admission = json.loads((ROOT/args.manifest).read_text())
    if admission['status'] != 'prepared_unrun' or admission['source_sha256'] != sources:
        raise ValueError('Changed prepared source bundle; preserve original manifest')
    if admission['arguments'] != vars(args):
        raise ValueError('Changed declared diagnostic arguments')
    if admission['parent_result_sha256'] != parent_hash or admission['weights_sha256'] != checkpoint_hash:
        raise ValueError('Changed declared actual producer evidence')
    if sha(ROOT/admission['queue']) != admission['queue_sha256']:
        raise ValueError('Changed one-job queue')
    manifest_hash = sha(ROOT/args.manifest)
    import numpy as np
    import torch
    import language_batched_benchmark as language
    from cached_inference_contracts import pair
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.sparse_inference import sparse_logits
    from sleeping_machines.prepacked_sparse_inference import PrepackedSparseWorker
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    cases, lifecycle = [], []
    factory = fast_class(AddressedEventHeads)

    def check(model, rows, label):
        # Construction within inference_mode must produce versioned private tensors.
        with torch.inference_mode():
            worker = PrepackedSparseWorker(model)
        def prepared(unused_model, rows, seed, all_logits=False):
            return worker.evaluate(rows, seed, all_logits)
        # Existing observer captures the actual sparse producer frame/mins.
        prepared.__wrapped__ = getattr(sparse_logits, '__wrapped__', sparse_logits)
        for seed in (314159, 322078):
            result = pair(torch, batched_logits, prepared, model, rows, seed, label)
            result['worker'] = worker.accounting()
            result['prepared_forward_calls'] = 2
            cases.append(result)
        # Independent weight ownership: updates to the caller's model cannot
        # alter the worker. Mutating the private worker must be rejected.
        before = worker.evaluate(rows, 314159, True).clone()
        original = model.embedding.weight.clone()
        model.embedding.weight.add_(.1)
        try:
            assert torch.equal(before, worker.evaluate(rows, 314159, True))
        finally:
            model.embedding.weight.copy_(original)
        assert torch.equal(model.embedding.weight, original)
        worker._model.embedding.weight.add_(.1)
        try:
            worker.evaluate(rows, 314159, True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Changed private weight version accepted')
        stats = worker.accounting()
        assert stats['prepared_stack_builds'] == 1 and stats['completed_evaluation_calls'] == 6
        assert stats['per_call_unit_parameter_stacking_payload_bytes'] == 0
        lifecycle.append(dict(label=label, original_model_isolated=True, private_mutation_rejected=True, **stats))

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for pool in (1, 3, 4):
            for dtype in (torch.float32, torch.float64):
                torch.manual_seed(146+pool)
                model = factory(sources=1, content_dim=5, classes=5, payload=8, depth=2,
                                heads=2, pool=pool).to(dtype).eval()
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter)*.1)
                generator = np.random.default_rng(146+pool)
                rows = [dict(events=[(float(t)+float(generator.random()), generator.normal(size=5))
                                     for t in range(length)]) for length in (9, 3, 7)]
                check(model, rows, 'synthetic')
        if all(case['passed'] for case in cases):
            symbols = language.load_text(0, 128)
            data_hash = hashlib.sha256(symbols.tobytes()).hexdigest()
            rows = [language.rows_of(symbols, [offset], length)[0]
                    for offset, length in ((0, 32), (39, 17), (78, 24))]
            weights = torch.load(checkpoint, weights_only=True, map_location='cpu')
            for dtype in (torch.float32, torch.float64):
                model = factory(sources=1, content_dim=27, classes=27, payload=config['payload'],
                                depth=config['depth'], heads=config['heads'], pool=config['pool']).to(dtype).eval()
                model.load_state_dict(weights)
                check(model, rows, 'actual_trained_FIT')
        else:
            data_hash = None
    assert torch.equal(rng, torch.get_rng_state())
    assert sha(parent_path) == parent_hash and sha(checkpoint) == checkpoint_hash
    assert sha(ROOT/args.manifest) == manifest_hash
    assert sha(ROOT/admission['queue']) == admission['queue_sha256']
    for name, digest in sources.items():
        assert sha(ROOT/name) == digest, name
    passed = len(cases) == 16 and all(case['passed'] for case in cases)
    # Pair contracts make four producer calls; each model lifecycle adds two
    # successful worker calls (the rejected mutation performs no forward).
    executed = sum(4*len(c['lengths'])*max(c['lengths']) for c in cases)
    active = sum(4*sum(c['lengths']) for c in cases)
    for index in range(0, len(cases), 2):
        case = cases[index]
        executed += 2*len(case['lengths'])*max(case['lengths'])
        active += 2*sum(case['lengths'])
    result = dict(status='completed' if passed else 'failed_contract', contracts_admitted=passed,
                  args=vars(args), parent_sha256=parent_hash, weights=parent['final_weights'],
                  admission_manifest_sha256=manifest_hash,
                  weights_sha256=checkpoint_hash, source_sha256=sources, cases=cases, lifecycle=lifecycle,
                  parent_training_presentations=parent['fitting_chars'],
                  parent_is_admission_pilot=bool(config.get('max_windows', 0)),
                  parent_has_completed_test_quality='test_bpc' in parent,
                  protocol=dict(data='text8 FIT[0:128], fixed spans, no labels/heldout loss', data_sha256=data_hash,
                                original_and_promoted_precision=True, synthetic_gate_first=True,
                                reference='Original batched producer; actual every-winner/state/cache/RNG observer'),
                  execution=dict(forward_calls=4*len(cases)+2*len(lifecycle),
                                 evaluated_lane_positions=executed, active_lane_positions=active,
                                 backward_calls=0, optimizer_updates=0),
                  hardware=dict(device='cpu', threads=1, torch=torch.__version__),
                  wall_s=time.perf_counter()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Paid parity/lifecycle admission; no heldout quality assignment, physical traffic, speed or energy claim')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as file:
        file.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], cases=len(cases), native_admitted=passed)))
    if not passed:
        raise RuntimeError('Prepared-worker admission failed; retain diagnostic output')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', required=True)
    run(parser.parse_args())
