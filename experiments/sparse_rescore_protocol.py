"""Saved-quality sparse admission and E64 window ledger; standard library only."""
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SOURCES = (
    'sleeping_machines/batched_episodes.py', 'sleeping_machines/fast_native_core.py',
    'sleeping_machines/addressed_event_heads.py', 'sleeping_machines/prepacked_sparse_inference.py',
    'sleeping_machines/sparse_inference.py', 'sleeping_machines/parallel_stream_language.py',
    'experiments/prepacked_sparse_contracts.py', 'experiments/cached_inference_contracts.py',
    'experiments/sparse_inference_accounting.py', 'experiments/language_batched_benchmark.py',
    'experiments/e120_shared_tasks.py',
)
RESCORE_SOURCES = CONTRACT_SOURCES + (
    'experiments/sparse_rescore_protocol.py', 'experiments/trained_sparse_rescore.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(names, root=ROOT):
    return {name: sha(root / name) for name in names}


def window_plan(length, segment, lanes):
    """Exactly retain the producer's exclusive stop and first-window convention."""
    if any(type(x) is not int or x < 1 for x in (length, segment, lanes)):
        raise ValueError('Positive integer dimensions required')
    if segment % 2 or length <= segment + 1:
        raise ValueError('Even segment and a nonempty saved-protocol evaluation required')
    half = segment // 2
    starts = list(range(0, length - segment - 1, half))
    batches = [starts[b:b + lanes] for b in range(0, len(starts), lanes)]
    scored = segment + (len(starts) - 1) * half
    evaluated = len(starts) * segment
    return dict(batches=batches, windows=len(starts), backend_calls=len(batches),
                scored_targets=scored, evaluated_positions_per_backend=evaluated,
                warm_context_positions_per_backend=evaluated - scored,
                paired_evaluated_positions=2 * evaluated,
                first_target=1, last_target=scored,
                omitted_target_positions=length - 1 - scored)


def parent_quality(parent, split, segment):
    a = parent['args']
    if parent.get('status') != 'completed' or a.get('max_windows', 0):
        raise ValueError('A completed quality fit, not an admission pilot, is required')
    if split not in ('dev', 'test') or segment not in (a['segment'], a.get('eval_segment')):
        raise ValueError('Only saved DEV/test protocols are admitted')
    key = split + '_bpc' + ('_eval_segment' if segment != a['segment'] else '')
    score = parent[key]
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Finite saved quality required')
    return score


def admit_contracts(result, manifest, parent_hash, weights_hash):
    """Require every declared case and guard, not just a top-level success flag."""
    if result.get('status') != 'completed' or result.get('contracts_admitted') is not True:
        raise ValueError('Actual trained sparse contracts have not passed')
    for key, expected in (
        ('parent_sha256', parent_hash), ('weights_sha256', weights_hash),
        ('admission_manifest_sha256', manifest['manifest_sha256']),
        ('source_sha256', manifest['source_sha256']), ('args', manifest['arguments']),
    ):
        if result.get(key) != expected:
            raise ValueError('Contract provenance differs: ' + key)
    if result.get('parent_is_admission_pilot') is not False or result.get('parent_has_completed_test_quality') is not True:
        raise ValueError('Contracts must use this completed quality checkpoint')
    expected = {('synthetic', dtype, pool, seed)
                for dtype in ('torch.float32', 'torch.float64') for pool in (1, 3, 4)
                for seed in (314159, 322078)}
    pool = manifest['parent_pool']
    expected |= {('actual_trained_FIT', dtype, pool, seed)
                 for dtype in ('torch.float32', 'torch.float64') for seed in (314159, 322078)}
    observed = []
    checks = {'logits', 'mem', 'arr', 'seen', 'ctx_vals', 'ctx_arr', 'has_ctx', 'cache_invariant'}
    for case in result.get('cases', []):
        observed.append((case.get('label'), case.get('dtype'), case.get('pool'), case.get('seed')))
        if (case.get('passed') is not True or case.get('active_races', 0) <= 0
                or case.get('winner_agreements') != case['active_races']
                or set(case.get('checks', {})) != checks
                or any(value is not True for value in case['checks'].values())):
            raise ValueError('A required trajectory contract failed')
    if len(observed) != 16 or set(observed) != expected:
        raise ValueError('Missing, duplicated or unexpected trajectory cases')
    lifecycle = result.get('lifecycle', [])
    if len(lifecycle) != 8 or any(
        c.get('original_model_isolated') is not True or c.get('private_mutation_rejected') is not True
        or c.get('prepared_stack_builds') != 1 or c.get('completed_evaluation_calls') != 6
        or c.get('per_call_unit_parameter_stacking_payload_bytes') != 0 for c in lifecycle
    ):
        raise ValueError('Required worker lifecycle contracts failed')
    execution = result.get('execution', {})
    if execution.get('backward_calls') != 0 or execution.get('optimizer_updates') != 0:
        raise ValueError('Unexpected numerical learning in admission')

