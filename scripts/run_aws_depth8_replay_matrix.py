"""Serial immutable matrix recovery with guarded phases and per-result publication."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime
import fcntl
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
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    os.chdir(ROOT)
    path = (ROOT / args.manifest).resolve()
    plan = json.loads(path.read_text())
    state = path.parent / 'worker_recovery.status.json'
    if state.exists():
        raise ValueError('Preserve existing lifecycle; prepare a new recovery plan')
    live = dict(status='checking', completed=[], current_job=None,
                host=os.uname().nodename, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if args.jobs > 1 and plan.get('parallel_host') != os.uname().nodename:
        raise ValueError('Parallel protocol must be explicitly assigned to this AWS host')

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
        git('add', '-f', '--', *files)
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
        host_lock = open('/tmp/experiments-runner.lock', 'a') if args.jobs > 1 else None
        if host_lock:
            fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def execute(job, slot):
            caps = job['resource_caps']
            env = dict(os.environ, **{key: str(value) for key, value in caps.items()},
                       JOB_TIMEOUT_S=str(job['timeout_s']))
            descriptors = ()
            if host_lock:
                env.update(AWS_GYM_SLOT=str(slot), AWS_GYM_HOST_LOCK_FD=str(host_lock.fileno()))
                descriptors = (host_lock.fileno(),)
            started = time.monotonic()
            rc = subprocess.run(['bash', 'experiments/queue/run_safe.sh', job['queue']],
                                cwd=ROOT, env=env, pass_fds=descriptors).returncode
            if rc:
                raise RuntimeError(f'Guarded stage {job["tag"]} failed with exit {rc}; wall {time.monotonic()-started:.3f}s')
            return validate_result(ROOT, job)

        def finish(job, result, reused):
            checkpoint = Path(job['result']).with_suffix('.progress.pt')
            files = [job['result']] + ([str(checkpoint)] if checkpoint.exists() else [])
            publish(files, 'Record completed ' + job['tag'])
            peak = result.get('max_rss_kb', 0)
            live['completed'].append(dict(tag=job['tag'], result=job['result'], reused=reused,
                max_rss_kb=peak, wall_s=result.get('wall_s')))
            save(); print(json.dumps(live['completed'][-1]), flush=True)
            if job['stage'] == 'smoke' and (not result.get('small_fit_learning_passed') or peak >= 1000000):
                raise RuntimeError('Depth8 learning/RSS admission failed: ' + job['tag'])
            if job['stage'] == 'smoke' and peak > job['resource_caps']['MEM_CAP_RSS_KB'] * .85:
                raise RuntimeError('Smoke near RSS cap; review pilot allocation: ' + job['tag'])

        done = set()
        try:
            with ThreadPoolExecutor(max_workers=args.jobs) as workers:
                # The phase barrier remains even when cells run in parallel.
                for phase in ('checks', 'pilots'):
                    pending = [job for job in plan['jobs'] if (job['stage'] == 'pilot') == (phase == 'pilots')]
                    active = {}; failure_errors = []; free_slots = set(range(1, args.jobs + 1))
                    while pending or active:
                        if not failure_errors:
                            frozen()
                            for job in list(pending):
                                if not set(job['requires']).issubset(done):
                                    continue
                                if (ROOT / job['result']).exists():
                                    finish(job, validate_result(ROOT, job), True)
                                    done.add(job['tag']); pending.remove(job); continue
                                if not free_slots:
                                    break
                                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
                                reserved = sum(info[0]['resource_caps']['MEM_CAP_RSS_KB'] for info in active.values())
                                if available - reserved - job['resource_caps']['MEM_CAP_RSS_KB'] < 8192 * 1024:
                                    continue
                                slot = min(free_slots); free_slots.remove(slot)
                                active[workers.submit(execute, job, slot)] = (job, slot)
                                pending.remove(job)
                        live.update(status='running_' + phase, active_jobs=[info[0]['tag'] for info in active.values()],
                                    max_parallel_jobs=args.jobs, current_job=None); save()
                        if not active:
                            if failure_errors:
                                raise RuntimeError('; '.join(failure_errors))
                            if pending:
                                raise RuntimeError('Admission blocked by memory or unavailable prerequisite')
                            break
                        ready, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                        for future in ready:
                            job, slot = active.pop(future); free_slots.add(slot)
                            try:
                                finish(job, future.result(), False); done.add(job['tag'])
                            except Exception as error:
                                failure_errors.append(str(error))
                                failure = path.parent / (job['tag'] + '.failure.json')
                                failure.write_text(json.dumps(dict(status='failed', tag=job['tag'], error=str(error),
                                    resource_caps=job['resource_caps'], queue=job['queue'], source_sha256=plan['source_sha256'],
                                    runner_log='experiments/queue/runner_' + Path(job['queue']).stem + '.out'), indent=2) + '\n')
                                publish([str(failure.relative_to(ROOT))], 'Preserve failed AWS matrix stage ' + job['tag'])
                        # On failure, admit no new jobs; finish/publish in-flight
                        # work before stopping. Never abandon guards or trainers.
                        if failure_errors and not active:
                            raise RuntimeError('; '.join(failure_errors))
        finally:
            if host_lock:
                host_lock.close()
        live.update(status='completed', current_job=None); save()
        summary = path.parent / 'completed_summary.json'
        summary.write_text(json.dumps(live, indent=2) + '\n')
        publish([str(summary.relative_to(ROOT))], 'Complete guarded AWS fast matrix recovery')
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save()
        raise


if __name__ == '__main__':
    main()
