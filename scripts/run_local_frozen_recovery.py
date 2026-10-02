"""Bounded serial recovery after a host session loses its prior processes.

Preserves old lifecycles. Every numerical job still enters through run_safe;
this supervisor never signals stale PIDs or edits frozen model dependencies.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_aws_matrix_recovery import validate_result
from scripts.run_state_credit_boundary import identity


def verify(plan):
    if os.uname().nodename != plan['host']:
        raise ValueError('Host-specific recovery required')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'main':
        raise ValueError('Main required')
    for name, digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    for job in plan['jobs']:
        queue = ROOT / job['queue']
        if hashlib.sha256(queue.read_bytes()).hexdigest() != job['queue_sha256']:
            raise ValueError('Frozen queue changed: ' + job['tag'])
        lines = [s for s in queue.read_text().splitlines() if s.strip() and not s.startswith('#')]
        if len(lines) != 1 or lines[0].split()[0] != job['tag']:
            raise ValueError('Exactly one matching job per queue required')
    for process in plan['absent_predecessors']:
        current = identity(process['pid'])
        if current and current['state'] != 'Z' and current['start_ticks'] == process['start_ticks']:
            raise ValueError('Preserved predecessor is still alive; do not insert recovery')


def compare(plan):
    pilots = [validate_result(ROOT, j) for j in plan['jobs'] if j['stage'] == 'pilot']
    if len(pilots) != 2:
        raise ValueError('Exactly two matched write-credit pilots required')
    baseline, teacher = pilots
    if baseline['data_sha256'] != teacher['data_sha256']:
        raise ValueError('Matched data required')
    settings = lambda r: {k: v for k, v in r['args'].items() if k not in ('tag', 'state_credit')}
    if settings(baseline) != settings(teacher) or baseline['args']['state_credit'] != 0 or teacher['args']['state_credit'] != 1:
        raise ValueError('Only added credit may differ')
    gain = teacher['final']['dev']['accuracy'] - baseline['final']['dev']['accuracy']
    nll = baseline['final']['dev']['nll'] - teacher['final']['dev']['nll']
    ratio = teacher['work']['total_training_unit_special_flops'] / baseline['work']['total_training_unit_special_flops']
    return dict(status='completed', kind='exploratory_state_credit_comparison',
        source_results=[j['result'] for j in plan['jobs'] if j['stage'] == 'pilot'],
        accuracy_gain=gain, nll_gain=nll, whole_fitting_work_ratio=ratio,
        followup_gate_passed=gain >= .05 and nll >= .02 and ratio <= 2,
        scope='Single seed, reused development populations; no benchmark or unbiased-gradient claim. No automatic larger fit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    state = path.with_suffix('.status.json')
    if state.exists():
        raise ValueError('Preserve prior recovery lifecycle; use a new plan')
    live = dict(status='checking', completed=[], current_job=None,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    child = None

    def save():
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(live, indent=2) + '\n')
        temporary.replace(state)

    def publish(files, message):
        # Local commits only: no external rebase while another host is fitting.
        if not subprocess.check_output(['git', 'status', '--porcelain', '--', *files], cwd=ROOT):
            return
        subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '--only', '-m', message, '--', *files], cwd=ROOT, check=True)

    def interrupted(signum, frame):
        raise InterruptedError('Supervisor signal ' + str(signum))

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    save()
    try:
        for job in plan['jobs']:
            verify(plan)
            if not set(job['requires']).issubset({r['tag'] for r in live['completed']}):
                raise ValueError('Unmet prerequisites: ' + job['tag'])
            live.update(status='running', current_job=job['tag']); save()
            if not (ROOT / job['result']).exists():
                env = dict(os.environ, **{k: str(v) for k, v in plan['resource_caps'].items()},
                    JOB_TIMEOUT_S=str(job['timeout_s']))
                # Never accept inherited AWS slots on this serial host.
                for key in ('AWS_GYM_SLOT', 'AWS_GYM_HOST_LOCK_FD', 'WAIT', 'AFTER_JOB_HOOK', 'CONTINUE_ON_FAILURE'):
                    env.pop(key, None)
                child = subprocess.Popen(['bash', 'experiments/queue/run_safe.sh', job['queue']], cwd=ROOT, env=env)
                rc = child.wait(); child = None
                if rc:
                    raise RuntimeError(f'Guarded stage {job["tag"]} exited {rc}')
            row = validate_result(ROOT, job)
            verify(plan)
            if job['stage'] == 'smoke' and row['max_rss_kb'] > .85 * plan['resource_caps']['MEM_CAP_RSS_KB']:
                raise ValueError('Smoke near RSS cap; do not admit fits')
            publish([job['result']], 'Record completed ' + job['tag'])
            live['completed'].append(dict(tag=job['tag'], result=job['result'],
                wall_s=row['wall_s'], max_rss_kb=row['max_rss_kb'])); save()
        summary = compare(plan)
        analysis = path.with_name(path.stem + '_analysis.json')
        analysis.write_text(json.dumps(summary, indent=2) + '\n')
        publish([str(analysis.relative_to(ROOT))], 'Compare matched integrated addressed-state credit pilots')
        live.update(status='completed', current_job=None, comparison=summary); save()
    except BaseException as error:
        live.update(status='needs_review', error=str(error)); save()
        raise
    finally:
        if child is not None and child.poll() is None:
            # Let run_safe terminate its own process group and watchdog cleanly.
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                live.update(status='needs_review', error='Guard did not drain within 30 seconds; inspect before any new job')
                save()


if __name__ == '__main__':
    main()
