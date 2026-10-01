"""Serial immutable matrix recovery with guarded phases and per-result publication."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def validate_result(root, job):
    result = json.loads((root / job['result']).read_text())
    if result.get('status') != 'completed' or result['args']['tag'] != job['tag']:
        raise ValueError('Completed matching result required: ' + job['tag'])
    for name, sha in result.get('source_sha256', {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Result source differs: ' + name)

    def visit(value):
        if isinstance(value, dict):
            if value.get('formula_coverage_complete') is False or value.get('unsupported_floating_operators'):
                raise ValueError('Incomplete operator accounting: ' + job['tag'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite result: ' + job['tag'])
    visit(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    path = (ROOT / args.manifest).resolve()
    plan = json.loads(path.read_text())
    state = path.parent / 'worker_recovery.status.json'
    if state.exists():
        raise ValueError('Preserve existing lifecycle; prepare a new recovery plan')
    live = dict(status='checking', completed=[], current_job=None,
                host=os.uname().nodename, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def save():
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(live, indent=2) + '\n')
        temporary.replace(state)

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()

    def frozen():
        if git('branch', '--show-current') != 'main':
            raise ValueError('Main required')
        for name, sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError('Frozen source changed: ' + name)
        for job in plan['jobs']:
            if hashlib.sha256((ROOT / job['queue']).read_bytes()).hexdigest() != job['queue_sha256']:
                raise ValueError('Queue changed: ' + job['queue'])

    def publish(files, message):
        if not git('status', '--porcelain', '--', *files):
            return
        git('add', '--', *files)
        git('commit', '--only', '-m', message, '--', *files)
        for attempt in range(6):
            fetched = subprocess.run(['git', 'fetch', 'origin'], cwd=ROOT)
            if fetched.returncode == 0:
                rebased = subprocess.run(['git', 'rebase', 'origin/main'], cwd=ROOT)
                if rebased.returncode:
                    raise RuntimeError('Publication rebase requires conflict resolution; evidence committed locally')
                pushed = subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT)
                if pushed.returncode == 0:
                    frozen()
                    return
            time.sleep(min(5 * 2**attempt, 60))
        raise RuntimeError('Publication retries exhausted; evidence committed locally')

    by_tag = {job['tag']: job for job in plan['jobs']}
    save()
    try:
        frozen()
        if shutil.which('nvidia-smi'):
            occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if occupied:
                raise ValueError('GPU occupied')
        process_lines = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines()
        forbidden = ('scripts/run_aws_non_shd.py', 'scripts/run_research_gym.py',
                     'scripts/run_native_research_campaign.py', 'scripts/run_delay_feature_campaign.py',
                     'scripts/run_aws_fast_matrix_after_depth4.py')
        for line in process_lines:
            if any(name in line for name in forbidden):
                raise ValueError('Existing coordinator must finish: ' + line)
        # Complete all contracts and accounting smokes before any pilot.
        ordered = [job for job in plan['jobs'] if job['stage'] != 'pilot']
        ordered += [job for job in plan['jobs'] if job['stage'] == 'pilot']
        for job in ordered:
            frozen()
            for tag in job['requires']:
                validate_result(ROOT, by_tag[tag])
            live.update(status='running', current_job=job['tag']); save()
            if (ROOT / job['result']).exists():
                result = validate_result(ROOT, job)
                reused = True
            else:
                reused = False
                caps = job['resource_caps']
                env = dict(os.environ, **{key: str(value) for key, value in caps.items()},
                           JOB_TIMEOUT_S=str(job['timeout_s']))
                started = time.monotonic()
                rc = subprocess.run(['bash', 'experiments/queue/run_safe.sh', job['queue']], cwd=ROOT, env=env).returncode
                if rc:
                    failure = path.parent / (job['tag'] + '.failure.json')
                    failure.write_text(json.dumps(dict(status='failed', tag=job['tag'], exit_code=rc,
                        wall_s=time.monotonic()-started, resource_caps=caps, queue=job['queue'],
                        source_sha256=plan['source_sha256'],
                        runner_log='experiments/queue/runner_' + Path(job['queue']).stem + '.out'), indent=2) + '\n')
                    publish([str(failure.relative_to(ROOT))], 'Preserve failed AWS matrix stage ' + job['tag'])
                    raise RuntimeError('Guarded prerequisite/stage failed: ' + job['tag'])
                result = validate_result(ROOT, job)
            publish([job['result']], 'Record completed ' + job['tag'])
            peak = result.get('max_rss_kb', 0)
            if job['stage'] == 'smoke' and peak > job['resource_caps']['MEM_CAP_RSS_KB'] * .85:
                raise RuntimeError('Smoke near RSS cap; review pilot allocation: ' + job['tag'])
            live['completed'].append(dict(tag=job['tag'], result=job['result'], reused=reused,
                max_rss_kb=peak, wall_s=result.get('wall_s'))); save()
            print(json.dumps(live['completed'][-1]), flush=True)
        live.update(status='completed', current_job=None); save()
        summary = path.parent / 'completed_summary.json'
        summary.write_text(json.dumps(live, indent=2) + '\n')
        publish([str(summary.relative_to(ROOT))], 'Complete guarded AWS fast matrix recovery')
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
