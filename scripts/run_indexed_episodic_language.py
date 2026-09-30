"""Content-indexed eight-depth KV pilot and bounded matched data promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEM='local_indexed_episodic_depth8_20260930T211500Z'


def main():
    plan=json.loads((ROOT/f'experiments/queue/{STEM}.json').read_text())
    status_path=ROOT/f'experiments/queue/{STEM}.status.json'
    live=dict(status='running',completed=[])
    def save():status_path.write_text(json.dumps(live,indent=2)+'\n')
    def validate(path):
        r=json.loads((ROOT/path).read_text())
        if r['status']!='completed':raise ValueError('Completed record required: '+path)
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Source changed: '+name)
        return r
    def run(job):
        for name,digest in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Preserve active experiment sources: '+name)
        live['current_job']=job['tag'];save()
        env=dict(os.environ,**{k:str(v) for k,v in plan['resource_caps'].items()},
            JOB_TIMEOUT_S=str(job['timeout_s']),WAIT='1',WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],cwd=ROOT,env=env,check=True)
        r=validate(job['result'])
        if r['protocol']['official_test_read']:raise ValueError('No official test in development pilot')
        live['completed'].append(job['result']);save();return r
    def publish():
        subprocess.run([sys.executable,'scripts/update_episodic_language_report.py',
                        *plan['preserved_comparisons'],*live['completed']],cwd=ROOT,check=True)
    try:
        validate(plan['required_contracts']);validate(plan['required_smoke'])
        six=validate(plan['six_depth_receiver']);eight=validate(plan['eight_depth_receiver'])
        pilot=run(plan['pilot']);publish()
        depth_gain=six['final']['dev']['bpc']-eight['final']['dev']['bpc']
        regression=pilot['final']['dev']['bpc']-eight['final']['dev']['bpc']
        live.update(depth_gain_bpc=depth_gain,kv_regression_bpc=regression);save()
        if depth_gain>=plan['minimum_depth_gain_bpc'] and regression<=plan['maximum_kv_regression_bpc']:
            # Extra data is warranted by the demonstrated depth intervention and
            # a functioning KV pilot; KV superiority itself remains a question.
            for job in plan['larger_pair']:run(job)
            publish()
        else:
            live['larger_pair']='deferred for diagnosis under predeclared gate';save()
        live.update(status='completed_development_campaign',current_job=None);save()
        # Longer runs need a packed cache, measured workload and a reviewed
        # promotion. Never resume superseded six-depth large queues implicitly.
    except Exception as error:
        live.update(status='needs_review',error=str(error));save();raise


if __name__=='__main__':main()
