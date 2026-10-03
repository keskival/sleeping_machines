"""Stdlib coverage, provenance-rejection and import-boundary contracts; no model execution."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from sparse_rescore_protocol import admit_contracts, parent_quality, sha, source_hashes, window_plan


def rejected(function):
    try:
        function()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError('Invalid admission accepted')


def fixture():
    manifest = dict(manifest_sha256='manifest', source_sha256={'source': 'hash'},
                    arguments={'result': 'fit'}, parent_pool=4)
    checks = {k: True for k in ('logits', 'mem', 'arr', 'seen', 'ctx_vals', 'ctx_arr', 'has_ctx', 'cache_invariant')}
    cases = [dict(label=label, dtype=dtype, pool=pool, seed=seed, passed=True,
                  active_races=73, winner_agreements=73, checks=dict(checks))
             for label, pools in (('synthetic', (1, 3, 4)), ('actual_trained_FIT', (4,)))
             for pool in pools for dtype in ('torch.float32', 'torch.float64')
             for seed in (314159, 322078)]
    result = dict(status='completed', contracts_admitted=True, parent_sha256='parent',
                  weights_sha256='weights', admission_manifest_sha256='manifest',
                  source_sha256=manifest['source_sha256'], args=manifest['arguments'],
                  parent_is_admission_pilot=False, parent_has_completed_test_quality=True, cases=cases,
                  lifecycle=[dict(original_model_isolated=True, private_mutation_rejected=True,
                    prepared_stack_builds=1, completed_evaluation_calls=6,
                    per_call_unit_parameter_stacking_payload_bytes=0) for _ in range(8)],
                  execution=dict(backward_calls=0, optimizer_updates=0))
    return result, manifest


def main(manifest_path=None):
    enumerations = 0
    for segment in (2, 4, 8, 128, 256):
        for length in (segment + 2, 2 * segment, 3 * segment + 7, 8192, 1000000):
            for lanes in (1, 3, 32, 64):
                p = window_plan(length, segment, lanes)
                starts = [s for batch in p['batches'] for s in batch]
                assert starts == list(range(0, length - segment - 1, segment // 2))
                # Targets partition consecutively, without re-scoring warm context.
                previous = 0
                for start in starts:
                    first = start + 1 if start == 0 else start + segment // 2 + 1
                    last = start + segment
                    assert first == previous + 1
                    previous = last
                assert previous == p['scored_targets'] == p['last_target']
                assert p['paired_evaluated_positions'] == 2 * len(starts) * segment
                assert p['scored_targets'] + p['omitted_target_positions'] == length - 1
                assert p['backend_calls'] == len(p['batches'])
                enumerations += 1
    for segment, lanes in ((128, 64), (256, 32)):
        assert window_plan(1000000, segment, lanes)['scored_targets'] == 999936
    for dimensions in ((128, 127, 1), (256, 256, 1), (1000, 128, 0), (1000, 128., 1)):
        rejected(lambda: window_plan(*dimensions))
    result, manifest = fixture()
    admit_contracts(result, manifest, 'parent', 'weights')
    mutations = [
        lambda r: r.update(status='failed_contract'),
        lambda r: r.update(parent_sha256='wrong'),
        lambda r: r.update(weights_sha256='wrong'),
        lambda r: r.update(source_sha256={}),
        lambda r: r.update(args={}),
        lambda r: r.update(admission_manifest_sha256='wrong'),
        lambda r: r.update(parent_is_admission_pilot=True),
        lambda r: r.update(parent_has_completed_test_quality=False),
        lambda r: r['cases'].pop(),
        lambda r: r['cases'].__setitem__(0, deepcopy(r['cases'][1])),
        lambda r: r['cases'][0].update(winner_agreements=72),
        lambda r: r['cases'][0]['checks'].update(cache_invariant=False),
        lambda r: r['cases'][0]['checks'].pop('arr'),
        lambda r: r['cases'][0].update(active_races=0, winner_agreements=0),
        lambda r: r['lifecycle'].pop(),
        lambda r: r['lifecycle'][0].update(private_mutation_rejected=False),
        lambda r: r['lifecycle'][0].update(prepared_stack_builds=2),
        lambda r: r['execution'].update(optimizer_updates=1),
    ]
    for mutation in mutations:
        invalid = deepcopy(result)
        mutation(invalid)
        rejected(lambda: admit_contracts(invalid, manifest, 'parent', 'weights'))
    parent = dict(status='completed', args=dict(segment=128, eval_segment=256, max_windows=0),
                  test_bpc=2., test_bpc_eval_segment=1.99)
    assert parent_quality(parent, 'test', 256) == 1.99
    rejected(lambda: parent_quality(dict(parent, args=dict(parent['args'], max_windows=60)), 'test', 256))
    rejected(lambda: parent_quality(dict(parent, test_bpc=float('nan')), 'test', 128))
    rejected(lambda: parent_quality(parent, 'test', 512))
    if manifest_path:
        # The actual prepared ladder has no numerical result. Refuse before runtime imports.
        m = json.loads((ROOT / manifest_path).read_text())
        assert not (ROOT / m['required_contracts']['output']).exists()
        runner = runpy.run_path(str(ROOT / 'experiments/trained_sparse_rescore.py'))['run']
        rejected(lambda: runner(SimpleNamespace(manifest=manifest_path)))
        ladder = json.loads((ROOT / manifest_path).parent.joinpath('manifest.json').read_text())
        assert len(ladder['jobs']) == 7
        for job in ladder['jobs']:
            frozen = json.loads((ROOT / job['manifest']).read_text())
            assert sha(ROOT / job['queue']) == frozen['queue_sha256']
            assert source_hashes(frozen['source_sha256']) == frozen['source_sha256']
            lines = [s for s in (ROOT / job['queue']).read_text().splitlines() if s and not s.startswith('#')]
            assert len(lines) == 1 and not (ROOT / job['output']).exists()
            if job['stage'] == 'full':
                prefix = frozen['required_prefix']
                assert sha(ROOT / prefix['manifest']) == prefix['manifest_sha256']
                rejected(lambda: runner(SimpleNamespace(manifest=job['manifest'])))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='passed', window_enumerations=enumerations,
                          invalid_contracts_rejected=len(mutations), native_runtime='unrun',
                          missing_native_admission_rejected=bool(manifest_path))))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
