"""Source-frozen paired saved-quality rescore. Guarded one-job queue only.

Admission/help use stdlib. No training, online update, compiler or speed claim.
The actual trained FP32/64 state/winner/cache contracts must already have passed.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from sparse_rescore_protocol import (RESCORE_SOURCES, admit_contracts, parent_quality,
                                     sha, source_hashes, window_plan)


def run(args):
    started = time.perf_counter()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    output = ROOT / manifest['output']
    if output.exists() or manifest['status'] != 'prepared_unrun':
        raise ValueError('Unique unused prepared job required')
    sources = source_hashes(RESCORE_SOURCES)
    if sources != manifest['source_sha256'] or sha(ROOT / manifest['queue']) != manifest['queue_sha256']:
        raise ValueError('Changed numerical sources or queue')
    parent_path = ROOT / manifest['parent']
    parent = json.loads(parent_path.read_text())
    weights = ROOT / parent['final_weights']
    parent_hash, weights_hash = sha(parent_path), sha(weights)
    if parent_hash != manifest['parent_sha256'] or weights_hash != manifest['weights_sha256']:
        raise ValueError('Changed completed fit or actual weights')
    for name, digest in parent['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed fit producer: ' + name)
    a = parent['args']
    if a.get('skip_init_from', 0) or a.get('tie_pools', False):
        raise ValueError('Current admission covers nongrown, untied models')
    split, length, segment = (manifest[name] for name in ('split', 'length', 'segment'))
    saved_bpc = parent_quality(parent, split, segment)
    if length not in (8192, a[split]):
        raise ValueError('Only prepared prefix or full saved split admitted')
    lanes = max(1, a['lanes'] * a['segment'] // segment)
    plan = window_plan(length, segment, lanes)
    if {k: v for k, v in plan.items() if k != 'batches'} != manifest['execution_plan']:
        raise ValueError('Changed evaluation boundary or work')
    required = manifest['required_contracts']
    contract_manifest_path = ROOT / required['manifest']
    if sha(contract_manifest_path) != required['manifest_sha256']:
        raise ValueError('Changed required contract manifest')
    contract_manifest = json.loads(contract_manifest_path.read_text())
    if (contract_manifest['source_sha256'] != {k: sources[k] for k in contract_manifest['source_sha256']}
            or sha(ROOT / contract_manifest['queue']) != contract_manifest['queue_sha256']):
        raise ValueError('Changed contract producer or queue')
    contracts_path = ROOT / required['output']
    contracts = json.loads(contracts_path.read_text())
    contracts_hash, manifest_hash = sha(contracts_path), sha(manifest_path)
    admit_contracts(contracts, dict(contract_manifest, manifest_sha256=required['manifest_sha256'],
                                   parent_pool=a['pool']), parent_hash, weights_hash)
    prefix_path, prefix_hash = None, None
    if length == a[split]:
        prefix = manifest['required_prefix']
        prefix_path = ROOT / prefix['output']
        prefix_hash = sha(prefix_path)
        prefix_result = json.loads(prefix_path.read_text())
        if (prefix_result.get('status') != 'completed' or prefix_result.get('length') != 8192
                or prefix_result.get('split') != 'dev' or prefix_result.get('segment') != segment
                or prefix_result.get('parent_sha256') != parent_hash
                or prefix_result.get('weights_sha256') != weights_hash
                or prefix_result.get('source_sha256') != sources
                or prefix_result.get('admission_manifest_sha256') != prefix['manifest_sha256']):
            raise ValueError('Matching DEV prefix admission must pass before full scoring')
        if sha(ROOT / prefix['manifest']) != prefix['manifest_sha256']:
            raise ValueError('Changed prefix protocol')
    # No torch/numpy import, model allocation or dataset access before admission.
    import numpy as np
    import torch
    from torch.nn import functional as F
    import language_batched_benchmark as language
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.prepacked_sparse_inference import PrepackedSparseWorker
    from sparse_inference_accounting import cache_accounting
    torch.set_num_threads(1)
    caller_rng = torch.get_rng_state().clone()
    bits = [0., 0.]
    maximum_logit_error = 0.
    completed_calls = 0
    failed = None
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27,
                    payload=a['payload'], depth=a['depth'], heads=a['heads'], pool=a['pool']).eval()
        model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
        worker = PrepackedSparseWorker(model)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        offset = parent['protocol'][split][0]
        symbols = language.load_text(offset, length)
        data_hash = __import__('hashlib').sha256(symbols.tobytes()).hexdigest()
        for batch in plan['batches']:
            rows = language.rows_of(symbols, batch, segment)
            reference = batched_logits(model, rows, 314159, all_logits=True)
            candidate = worker.evaluate(rows, 314159, all_logits=True)
            completed_calls += 1
            if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()):
                failed = dict(kind='nonfinite_logits', batch=completed_calls)
                break
            error = float((reference - candidate).abs().max())
            maximum_logit_error = max(maximum_logit_error, error)
            if not torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4):
                failed = dict(kind='paired_logits', batch=completed_calls, maximum_absolute_difference=error)
                break
            targets = torch.tensor(np.stack([symbols[s + 1:s + segment + 1] for s in batch])).long()
            for backend, logits in enumerate((reference, candidate)):
                ce = F.cross_entropy(logits.reshape(-1, 27), targets.reshape(-1), reduction='none').view(len(batch), segment)
                for lane, start in enumerate(batch):
                    part = ce[lane] if start == 0 else ce[lane, segment // 2:]
                    bits[backend] += float(part.sum()) / math.log(2)
            if completed_calls % 10 == 0:
                print(json.dumps(dict(paired_batches=completed_calls, total_batches=plan['backend_calls'])), flush=True)
        untouched = all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
        if not untouched or any(p.grad is not None for p in model.parameters()):
            failed = dict(kind='model_mutation_or_gradient')
        full = length == a[split]
        bpcs = [b / plan['scored_targets'] for b in bits] if failed is None else None
        if bpcs is not None and (abs(bpcs[0] - bpcs[1]) > 1e-5
                or (full and any(abs(bpc - saved_bpc) > 1e-5 for bpc in bpcs))):
            failed = dict(kind='quality_reproduction', paired_bpc=bpcs, saved_bpc=saved_bpc,
                          full_split=full, absolute_tolerance=1e-5)
        accounting = worker.accounting()
    if not torch.equal(caller_rng, torch.get_rng_state()):
        failed = dict(kind='caller_rng_changed')
    for path, digest in ((parent_path, parent_hash), (weights, weights_hash), (manifest_path, manifest_hash),
                         (contracts_path, contracts_hash), (contract_manifest_path, required['manifest_sha256'])):
        if sha(path) != digest:
            raise RuntimeError('Evidence changed during diagnostic: ' + str(path))
    if source_hashes(RESCORE_SOURCES) != sources or sha(ROOT / manifest['queue']) != manifest['queue_sha256']:
        raise RuntimeError('Source or queue changed during diagnostic')
    if prefix_path is not None and sha(prefix_path) != prefix_hash:
        raise RuntimeError('Prefix evidence changed during diagnostic')
    actual_positions = sum(len(b) * segment for b in plan['batches'][:completed_calls])
    result = dict(status='completed' if failed is None else 'failed_contract', failure=failed,
        parent_sha256=parent_hash, weights_sha256=weights_hash, source_sha256=sources,
        admission_manifest_sha256=manifest_hash, contracts_sha256=contracts_hash,
        prefix_sha256=prefix_hash,
        split=split, length=length, segment=segment, lanes=lanes, seed=314159,
        saved_bpc=saved_bpc, eager_bpc=bpcs[0] if bpcs else None, sparse_bpc=bpcs[1] if bpcs else None,
        reproduces_completed_full_split_quality=failed is None and full,
        maximum_absolute_logit_difference=maximum_logit_error, data_sha256=data_hash,
        execution=dict(plan=manifest['execution_plan'], paired_batches_completed=completed_calls,
                       actual_forward_calls=2 * completed_calls, actual_evaluated_positions=2 * actual_positions,
                       backward_calls=0, optimizer_updates=0),
        worker=accounting, state_cache_payload=cache_accounting(a['payload'], a['depth'], a['heads'], a['pool'], lanes),
        protocol='Original E64 exclusive-stop windows, saved lane grouping/seed and FP32 weights; paired eager and prepacked sparse logits',
        scope='Paid paired rescore, not an isolated timing, FLOP, traffic or energy benchmark. Every-race/final-state/cache checks apply to prerequisite FIT spans; full split compares every logit and aggregate quality.',
        hardware=dict(device='cpu', threads=1, torch=torch.__version__),
        wall_s=time.perf_counter() - started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], sparse_bpc=result['sparse_bpc'], full_quality=result['reproduces_completed_full_split_quality'])))
    if failed is not None:
        raise RuntimeError('Retained diagnostic failed; sparse quality remains unadmitted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    run(parser.parse_args())
