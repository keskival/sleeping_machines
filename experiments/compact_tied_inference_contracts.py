"""Prepared compact tied inference parity. Guarded one-job queue only.

Help/source admission are stdlib. Synthetic diagnostics do not inherit any
trained quality score. Optional actual tied weights need a new frozen manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]

SOURCES = (
    'sleeping_machines/compact_tied_inference.py', 'sleeping_machines/prepacked_sparse_inference.py',
    'sleeping_machines/sparse_inference.py', 'sleeping_machines/batched_episodes.py',
    'sleeping_machines/fast_native_core.py', 'sleeping_machines/addressed_event_heads.py',
    'sleeping_machines/parallel_stream_language.py', 'sleeping_machines/sparse_race_language.py',
    'sleeping_machines/parallel_head_race_language.py', 'experiments/cached_inference_contracts.py',
    'experiments/sparse_inference_accounting.py', 'experiments/compact_tied_resource_geometry.py',
    'experiments/compact_tied_inference_contracts.py', 'experiments/language_batched_benchmark.py',
    'experiments/e120_shared_tasks.py', 'experiments/dvs_tied_pool_benchmark.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    begin = time.perf_counter()
    manifest_path, output = ROOT / args.manifest, ROOT / args.out
    if output.exists():
        raise ValueError('Preserve existing output')
    manifest = json.loads(manifest_path.read_text())
    sources = {name: sha(ROOT / name) for name in SOURCES}
    if (manifest['status'] != 'prepared_unrun' or manifest['source_sha256'] != sources
            or manifest['arguments'] != vars(args) or sha(ROOT / manifest['queue']) != manifest['queue_sha256']):
        raise ValueError('Changed prepared source/arguments/queue')
    parent, weights = None, None
    if args.result:
        parent_path = ROOT / args.result
        parent = json.loads(parent_path.read_text())
        if (parent.get('status') != 'completed' or not parent['args'].get('tie_pools')
                or parent['args'].get('skip_init_from', 0)):
            raise ValueError('Actual completed nongrown tied-map parent required')
        weights = ROOT / parent['final_weights']
        if sha(parent_path) != manifest['parent_sha256'] or sha(weights) != manifest['weights_sha256']:
            raise ValueError('Changed actual parent/weights')
        for name, digest in parent['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Changed actual numerical producer: ' + name)
    elif manifest.get('parent_sha256') is not None or manifest.get('weights_sha256') is not None:
        raise ValueError('Synthetic admission cannot inherit quality parents')
    manifest_hash = sha(manifest_path)
    import numpy as np
    import torch
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.prepacked_sparse_inference import PrepackedSparseWorker
    from sleeping_machines.sparse_inference import sparse_logits
    from sleeping_machines.compact_tied_inference import (CompactTiedSparseWorker, SHARED, _compact_logits)
    from cached_inference_contracts import pair
    from compact_tied_resource_geometry import compact_accounting
    torch.set_num_threads(1)
    caller_rng = torch.get_rng_state().clone()
    cases, lifecycle = [], []
    execution = dict(forward_calls=0, evaluated_lane_positions=0, active_lane_positions=0,
                     backward_calls=0, optimizer_updates=0)
    factory = fast_class(AddressedEventHeads)
    def tie(model):
        for depth in model.units:
            for head in depth:
                for pool in head:
                    for unit in list(pool)[1:]:
                        for name in SHARED:
                            setattr(unit, name, getattr(pool[0], name))
        return model
    def charged(rows, calls):
        execution['forward_calls'] += calls
        execution['evaluated_lane_positions'] += calls * len(rows) * max(len(r['events']) for r in rows)
        execution['active_lane_positions'] += calls * sum(len(r['events']) for r in rows)
    def check(model, rows, label):
        with torch.inference_mode():
            ordinary = PrepackedSparseWorker(model)
            compact = CompactTiedSparseWorker(model)
        def old(unused, rows, seed, all_logits=False):
            return ordinary.evaluate(rows, seed, all_logits)
        old.__wrapped__ = getattr(sparse_logits, '__wrapped__', sparse_logits)
        def new(unused, rows, seed, all_logits=False):
            return compact.evaluate(rows, seed, all_logits)
        new.__wrapped__ = _compact_logits
        for function, backend in ((old, 'expanded_prepacked'), (new, 'compact_tied')):
            for seed in (311, 733):
                result = pair(torch, batched_logits, function, model, rows, seed, label)
                result['backend'] = backend
                cases.append(result)
                charged(rows, 4)
        expected = compact_accounting(model.payload, model.depth, model.heads, model.pool, len(rows),
                                      model.embedding.weight.element_size())
        stats = compact.accounting()
        assert stats['retained_packed_tensor_payload_bytes'] == expected['compact_final_packed_tensor_bytes']
        assert ordinary.accounting()['retained_packed_tensor_payload_bytes'] == expected['final_stacked_tensor_bytes_per_call']
        before = compact.evaluate(rows, 311, True).clone()
        charged(rows, 1)
        saved = model.embedding.weight.clone()
        model.embedding.weight.add_(.1)
        try:
            assert torch.equal(before, compact.evaluate(rows, 311, True))
            charged(rows, 1)
        finally:
            model.embedding.weight.copy_(saved)
        compact._model.units[0][0][0][0].input.weight.add_(.1)
        try:
            compact.evaluate(rows, 311, True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Changed shared private map accepted')
        assert compact.accounting()['prepared_stack_builds'] == 1
        lifecycle.append(dict(label=label, dtype=str(model.embedding.weight.dtype), pool=model.pool,
                              source_isolated=True, shared_map_mutation_rejected=True, accounting=stats,
                              expected_tensor_payload=expected))
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for pool in (1, 3, 4, 8):
            for dtype in (torch.float32, torch.float64):
                torch.manual_seed(148 + pool)
                model = tie(factory(sources=1, content_dim=5, classes=5, payload=8, depth=2,
                                    heads=2, pool=pool)).to(dtype).eval()
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * .1)
                generator = np.random.default_rng(148 + pool)
                rows = [dict(events=[(float(t) + float(generator.random()), generator.normal(size=5))
                                     for t in range(length)]) for length in (13, 5, 9)]
                check(model, rows, 'synthetic_tied')
        data_hash = None
        if parent is not None and all(case['passed'] for case in cases):
            import language_batched_benchmark as language
            symbols = language.load_text(0, 128)
            data_hash = hashlib.sha256(symbols.tobytes()).hexdigest()
            rows = [language.rows_of(symbols, [offset], length)[0]
                    for offset, length in ((0, 32), (39, 17), (78, 24))]
            saved = torch.load(weights, weights_only=True, map_location='cpu')
            config = parent['args']
            for dtype in (torch.float32, torch.float64):
                model = tie(factory(sources=1, content_dim=27, classes=27, payload=config['payload'],
                                    depth=config['depth'], heads=config['heads'], pool=config['pool'])).to(dtype).eval()
                model.load_state_dict(saved)
                check(model, rows, 'actual_trained_tied_FIT')
    expected_cases = 32 + (8 if parent is not None else 0)
    passed = len(cases) == expected_cases and all(case['passed'] for case in cases)
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert sha(manifest_path) == manifest_hash and sha(ROOT / manifest['queue']) == manifest['queue_sha256']
    assert {name: sha(ROOT / name) for name in SOURCES} == sources
    if parent is not None:
        assert sha(parent_path) == manifest['parent_sha256'] and sha(weights) == manifest['weights_sha256']
    result = dict(status='completed' if passed else 'failed_contract', contracts_admitted=passed,
        args=vars(args), source_sha256=sources, manifest_sha256=manifest_hash,
        parent_sha256=manifest.get('parent_sha256'), weights_sha256=manifest.get('weights_sha256'),
        cases=cases, lifecycle=lifecycle, execution=execution, fit_data_sha256=data_hash,
        hardware=dict(device='cpu', threads=1, torch=torch.__version__),
        wall_s=time.perf_counter() - begin, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Paid parity/layout/lifecycle diagnostic; synthetic cases have no trained quality. Actual tied FIT cases do not replace heldout rescoring or measured service resources.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], cases=len(cases), native_admitted=passed)))
    if not passed:
        raise RuntimeError('Retained compact tied inference contracts failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--result')
    run(parser.parse_args())
