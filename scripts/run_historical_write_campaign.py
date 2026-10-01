"""Finite, source-frozen, serial historical write-credit research ladder."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-stem', required=True)
    stem = parser.parse_args().manifest_stem
    if Path(stem).name != stem:
        raise ValueError('Local manifest stem required')
    plan = json.loads((ROOT/f'experiments/queue/{stem}.json').read_text())
    status = ROOT/f'experiments/queue/{stem}.status.json'
    if status.exists():
        raise ValueError('Do not start a second supervisor or overwrite its state')
    live = dict(status='checking_completed_campaign', completed=[], decisions=[], current_job=None)

    def save():
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(live, indent=2)+'\n')
        temporary.replace(status)

    def frozen():
        for name, sha in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
                raise ValueError('Queued source changed: '+name)

    def validate(path):
        row = json.loads((ROOT/path).read_text())
        if row['status'] != 'completed':
            raise ValueError('Completed record required: '+path)
        for name, sha in row['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
                raise ValueError('Result source changed: '+name)
        return row

    def run(job, timeout=None):
        frozen(); live.update(status='running', current_job=job['tag']); save()
        env = dict(os.environ, **{k: str(v) for k, v in plan['resource_caps'].items()},
                   JOB_TIMEOUT_S=str(timeout or job['timeout_s']), WAIT='1', WAIT_TIMEOUT_S='86400')
        subprocess.run(['bash', 'experiments/queue/run_safe.sh', job['queue']],
                       cwd=ROOT, env=env, check=True)
        row = validate(job['result'])
        if 'work' in row:
            traces = list(row['work']['traces'].values())
            audits = [t[k] for t in traces for k in
                      ('forward_and_loss','backward','gradient_normalization','gradient_clipping','optimizer')]
            audits.append(row['work']['inference_trace'])
            if not all(t['formula_coverage_complete'] for t in audits):
                raise ValueError('Incomplete operation accounting: '+job['tag'])
        live['completed'].append(job['result']); save()
        changes = subprocess.check_output(['git','status','--porcelain','--',job['result']],
                                          cwd=ROOT, text=True)
        if changes.strip():
            subprocess.run([sys.executable,'scripts/publish_language_credit_stage.py',job['result']],
                           cwd=ROOT, check=True)
        return row

    quality = lambda r: r['final']['dev']['bpc']
    work = lambda r: r['work']['cpu_emulator']['total_training_unit_special_flops']
    try:
        save(); frozen()
        previous = json.loads((ROOT/plan['prioritized_status']).read_text())
        if previous['status'] != 'completed_bounded_credit_campaign' or previous.get('current_job'):
            raise ValueError('Prior integrated campaign is not finished')
        reference = validate(plan['pilot_reference'])
        validate(plan['contracts'])
        for job in plan['smokes']:
            run(job)
        candidates = {}
        for alpha, job in plan['pilots'].items():
            row = run(job)
            gain = quality(reference)-quality(row)
            live.setdefault('pilot_gains_bpc', {})[alpha] = gain; save()
            if gain >= plan['minimum_pilot_gain_bpc']:
                candidates[alpha] = row
        if not candidates:
            live['decisions'].append('No completed pilot passes .02 bpc gain; stop scaling and retain all results')
            live.update(status='completed_bounded_write_credit_campaign', current_job=None); save(); return
        chosen = min(candidates, key=lambda a: (quality(candidates[a]), work(candidates[a])))
        live['selected_write_credit'] = chosen; save()
        eight = run(plan['eight_k'][chosen])
        control = validate(plan['indexed_control_eight_k'])
        if quality(eight) > quality(control)+plan['maximum_control_regression_bpc']:
            live['decisions'].append('Completed 8K misses indexed-control quality gate; no larger fit')
            live.update(status='completed_bounded_write_credit_campaign', current_job=None); save(); return
        larger = run(plan['thirtytwo_k'][chosen])
        run(plan['repeat_eight_k'][chosen])
        gain = quality(eight)-quality(larger)
        # Three float vectors + position and index ID, all H*D banks.
        args = larger['args']
        extra = (131072-32768)*args['heads']*args['depth']*(3*args['payload']*4+16)
        rss = larger['max_rss_kb']+extra/1024
        timeout = math.ceil(larger['wall_s']*4*1.5+1800)
        live.update(larger_data_gain_bpc=gain, projected_131k_rss_kb=rss,
                    projected_131k_timeout_s=timeout); save()
        if (gain >= plan['minimum_larger_data_gain_bpc'] and rss <= plan['maximum_projected_rss_kb']
                and timeout <= plan['maximum_timeout_s']):
            run(plan['onehundredthirtyone_k'][chosen], timeout)
        else:
            live['decisions'].append('131K deferred by measured quality, memory or duration gate')
        live.update(status='completed_bounded_write_credit_campaign', current_job=None); save()
    except Exception as error:
        live.update(status='needs_review', error=str(error)); save(); raise


if __name__ == '__main__':
    main()
