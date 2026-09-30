"""Guarded fidelity checks before a bounded eight-block full-KV scaling step."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEM = 'local_packed_episodic_ladder_20260930T222500Z'


def main():
    plan = json.loads((ROOT / f'experiments/queue/{STEM}.json').read_text())
    if plan['status'].startswith('superseded'):
        raise ValueError('Single-head scaling is superseded by the parallel-head campaign')
    status_path = ROOT / f'experiments/queue/{STEM}.status.json'
    live = dict(status='waiting_for_current_campaign', completed=[])

    def save():
        status_path.write_text(json.dumps(live, indent=2) + '\n')

    def validate(path):
        result = json.loads((ROOT / path).read_text())
        if result['status'] != 'completed':
            raise ValueError('Completed result required: ' + path)
        for name, digest in result['source_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Source changed: ' + name)
        return result

    def run(job):
        for name, digest in plan['source_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Preserve prepared experiment sources: ' + name)
        live.update(status='running', current_job=job['tag']); save()
        env = dict(os.environ, **{k: str(v) for k, v in plan['resource_caps'].items()},
                   JOB_TIMEOUT_S=str(job['timeout_s']), WAIT='1', WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash', 'experiments/queue/run_safe.sh', job['queue']],
                       cwd=ROOT, env=env, check=True)
        result = validate(job['result'])
        live['completed'].append(job['result']); save()
        return result

    try:
        save()
        # Let the existing coordinator finish its own publication before any
        # new guarded work. No priority change or competing training process.
        deadline = time.monotonic() + 86400
        while True:
            previous = json.loads((ROOT / plan['previous_status']).read_text())
            if previous['status'] == 'completed_development_campaign':
                break
            if previous['status'] == 'needs_review':
                raise ValueError('Existing campaign needs review; preserve it')
            if time.monotonic() > deadline:
                raise TimeoutError('Previous campaign did not finish within one day')
            time.sleep(5)
        contracts = run(plan['contracts']); run(plan['smoke'])
        checks = contracts['numerical_contracts']
        for name in ('lossless_storage_predictions', 'lossless_gradients_and_updates',
                     'exact_packed_checkpoint_next_update', 'slab_boundary_credit'):
            if not checks[name]:
                raise ValueError('Packing fidelity contract missing: ' + name)
        expected = validate(plan['unpacked_fidelity_reference'])
        replay = run(plan['fidelity'])
        if (replay['initial_dev']['bpc'] != expected['initial_dev']['bpc'] or
            replay['selected_epoch'] != expected['selected_epoch'] or
            replay['final']['dev']['bpc'] != expected['final']['dev']['bpc'] or
            [r['dev']['bpc'] for r in replay['curve']] !=
                [r['dev']['bpc'] for r in expected['curve']]):
            raise ValueError('Packed 2K replay does not preserve the saved learning curve')
        receiver = validate(plan['eight_k_receiver'])
        kv = validate(plan['eight_k_kv'])
        regression = kv['final']['dev']['bpc'] - receiver['final']['dev']['bpc']
        live['eight_k_kv_regression_bpc'] = regression; save()
        if regression <= plan['maximum_eight_k_regression_bpc']:
            run(plan['larger'])
            # Uses the same global report inventory, keeping the fidelity
            # replay outside quality-variant plots as a storage-only control.
            subprocess.run([sys.executable, 'scripts/update_episodic_language_report.py',
                            *live['completed']], cwd=ROOT, check=True)
        else:
            live['larger_deferred'] = '8K regression exceeds the predeclared bounded-data gate'
            save()
        live.update(status='completed_bounded_packed_campaign', current_job=None); save()
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
