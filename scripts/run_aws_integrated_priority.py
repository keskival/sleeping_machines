"""Serial, resumable integrated stages with completed-evidence publication."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/workspace')
STAMP = '20260930T193500Z'
CONTRACT = 'experiments/results/parallel_language/local_sparse_contract_20260930T175000Z.json'
MANIFEST = ROOT / 'experiments/queue/aws_integrated_priority_20260930T193500Z.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    if git('branch', '--show-current') != 'main':
        raise RuntimeError('Use main')
    sys.path.insert(0, str(ROOT / 'experiments'))
    from integrated_language_protocol import sources
    locked = sources(str(ROOT / 'experiments/integrated_language_benchmark.py'))
    plan = json.loads(MANIFEST.read_text())
    if plan['source_sha256'] != locked:
        raise RuntimeError('Integrated sources changed; preserve the source snapshot and review')
    def save():
        plan['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        MANIFEST.write_text(json.dumps(plan, indent=2) + '\n')
    def publish(paths, message):
        subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', message], cwd=ROOT, check=True)
        subprocess.run(['git', 'pull', '--rebase', 'origin', 'main'], cwd=ROOT, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
    previous = None
    for job in plan['jobs']:
        result_path = ROOT / job['result']
        if job.get('status') == 'completed':
            previous = json.loads(result_path.read_text())
            continue
        if sources(str(ROOT / 'experiments/integrated_language_benchmark.py')) != locked:
            raise RuntimeError('Contract-locked sources changed between stages')
        if previous is not None:
            # A measured whole-stage rate includes development and startup,
            # so this extrapolation intentionally leaves a recovery margin.
            rate = previous['wall_s'] / ((previous['args']['fit'] - 1) * previous['args']['epochs'])
            job['timeout_s'] = math.ceil(2 * rate * ((job['fit'] - 1) * 4 + 2_200_000))
        job['status'] = 'running'; save()
        env = dict(os.environ, WAIT='1', WAIT_TIMEOUT_S='172800', MIN_AVAIL_MB='8192',
                   MEM_CAP_KB='6000000', MEM_CAP_RSS_KB='2500000',
                   JOB_TIMEOUT_S=str(job['timeout_s']), PYTHONUNBUFFERED='1')
        queue = ROOT / job['queue']
        checkpoint = result_path.with_suffix('.progress.pt')
        if checkpoint.exists() and not result_path.exists():
            recovery = queue.with_name(queue.stem + '_recovery_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.txt')
            recovery.write_text(queue.read_text().strip() + ' --resume\n')
            queue = recovery
        rc = subprocess.run(['bash', 'experiments/queue/run_safe.sh', str(queue)], cwd=ROOT, env=env).returncode
        if rc:
            job.update(status='needs_review', exit_code=rc); save()
            publish([str(MANIFEST.relative_to(ROOT))], 'Record interrupted integrated stage; preserve recovery artifacts')
            raise RuntimeError(f'Guarded integrated stage stopped: {rc}')
        result = json.loads(result_path.read_text())
        if result['status'] != 'completed' or result['source_sha256'] != locked:
            raise RuntimeError('Incomplete or source-mismatched result')
        job.update(status='completed', wall_s=result['wall_s'],
                   development_bpc=result['final']['dev']['bpc']); save()
        subprocess.run([sys.executable, 'scripts/update_language_report.py', job['result']], cwd=ROOT, check=True)
        publish([str(MANIFEST.relative_to(ROOT))], 'Record completed AWS integrated stage and measured runtime')
        previous = result
        # Require useful learning before spending the long larger-data budget.
        if job['fit'] == 32768 and result['final']['dev']['bpc'] >= 3.120653:
            raise RuntimeError('Width-32 pilot did not improve the matched width-16 result; diagnose before scaling')
    plan['status'] = 'completed'; save()
    publish([str(MANIFEST.relative_to(ROOT))], 'Complete prioritized AWS integrated 10M campaign')


if __name__ == '__main__':
    main()
