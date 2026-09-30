"""Run immutable one-job queues serially and gate promotion on development evidence."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    args = parser.parse_args()
    path = ROOT/args.manifest
    plan = json.loads(path.read_text())
    def persist():
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(plan,indent=2)+'\n');temporary.replace(path)
    def stop(reason):
        plan.update(status='needs_improvement',reason=reason,updated_utc=now());persist()
        print('PROMOTION STOPPED: '+reason,flush=True);sys.exit(2)
    for source,digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != digest:
            stop('Source changed before execution: '+source)
    plan.update(status='running',updated_utc=now());persist()
    for job in plan['jobs']:
        if job.get('status') == 'completed':continue
        if job['kind'] == 'official_comparison':
            primary = next(j for j in plan['jobs'] if j['kind']=='primary_development')
            row = json.loads((ROOT/primary['result']).read_text())
            score = row['final']['dev']['bpc']
            if score > plan['promotion']['maximum_1m_development_bpc']:
                stop(f'1M development bpc {score:.4f} exceeds the declared scale-up gate; diagnose before a full run')
            data_rows = [j for j in plan['jobs'] if j['width']==128 and j['kind'] in ('capacity','data')]
            ordered = sorted((j['fit'],json.loads((ROOT/j['result']).read_text())['final']['dev']['bpc']) for j in data_rows)
            if ordered[-1][1] > ordered[0][1]-plan['promotion']['minimum_data_gain_bpc']:
                stop('Larger data did not improve the fixed-width development comparison enough')
        queue = ROOT/job['queue']
        commands = [line for line in queue.read_text().splitlines() if line.strip() and not line.startswith('#')]
        if len(commands) != 1 or commands[0].split()[0] != job['tag']:
            stop('Invalid one-job queue: '+job['queue'])
        plan.update(current_job=job['tag'],updated_utc=now());job.update(status='running',started_utc=now());persist()
        environment = dict(os.environ,**{str(k):str(v) for k,v in plan['resource_caps'].items()},
                           WAIT='1',WAIT_TIMEOUT_S='172800',JOB_TIMEOUT_S=str(job['timeout_s']))
        print(now()+' starting '+job['queue'],flush=True)
        rc = subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=environment).returncode
        if rc:stop(f'Guarded job {job["tag"]} exited {rc}; preserve checkpoint/log and investigate')
        result = json.loads((ROOT/job['result']).read_text())
        if result['status'] != 'completed':stop('Incomplete result: '+job['tag'])
        if job['kind'] != 'official_comparison' and result['protocol']['official_test_read']:
            stop('Development stage touched the official test')
        score = result['final']['dev']['bpc']
        if not math.isfinite(score) or result['initial_dev']['bpc']-score < .25:
            stop('No adequate finite development improvement: '+job['tag'])
        if result['work']['fitting_targets'] != (job['fit']-1)*job['epochs']:
            stop('Fitting budget mismatch: '+job['tag'])
        for row in result['diagnostics']:
            if not all(row['parameter_change_norms'][key]>0 for key in ('input.weight','output.weight','gate.weight')):
                stop('Inactive learned value block: '+job['tag'])
        job.update(status='completed',finished_utc=now(),development_bpc=score,
                   parameters=result['parameters'],training_flops=result['work']['total_training_arithmetic_flops'],
                   wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])
        plan['updated_utc']=now();persist()
        print(json.dumps({k:job[k] for k in ('tag','development_bpc','parameters','training_flops','wall_s')}),flush=True)
    plan.update(status='completed',current_job=None,updated_utc=now());persist()
    print('All scaling stages and the fixed official comparison completed',flush=True)


if __name__ == '__main__':
    main()
