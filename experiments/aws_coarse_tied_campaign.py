"""Guarded one-job stages, immutable sources, publish each completed numerical run."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PREFIX='aws_coarse_tied_20261002T233600Z'
CONTRACT=ROOT/'experiments/results/diagnostics/aws_coarse_tied_contracts_20261002T233500Z.json'
SUMMARY=ROOT/'experiments/results/diagnostics'/f'{PREFIX}_summary.json'


def command(args):subprocess.run(args,cwd=ROOT,check=True)


def check_sources(result):
    import hashlib
    for name,digest in result['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def finite(v):
        if isinstance(v,float):assert math.isfinite(v)
        elif isinstance(v,dict):
            for value in v.values():finite(value)
        elif isinstance(v,list):
            for value in v:finite(value)
    finite(result)


def publish(paths,message):
    command(['git','add','-f',*[str(p.relative_to(ROOT)) for p in paths]])
    command(['git','commit','-m',message]);command(['git','pull','--rebase']);command(['git','push','origin','main'])


def job(pool,stage,seed,fit,dev,epochs):
    check_sources(json.loads(CONTRACT.read_text()))
    tag=f'{PREFIX}_p{pool}_{stage}_s{seed}'
    queue=ROOT/'experiments/queue'/f'{tag}.txt';assert not queue.exists()
    args=['experiments/aws_coarse_tied_native.py','--tag',tag,'--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json',
        '--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--contracts',str(CONTRACT.relative_to(ROOT)),
        '--fit',str(fit),'--dev',str(dev),'--epochs',str(epochs),'--update-targets','16','--lr','.003','--payload','16','--depth','2','--heads','2',
        '--pool',str(pool),'--bins','4','--clock-step','.25','--seed',str(seed)]
    queue.write_text(tag+' '+' '.join(args)+'\n')
    env=dict(os.environ,MEM_CAP_KB='6000000',MEM_CAP_RSS_KB='2000000',MIN_AVAIL_MB='8192',JOB_TIMEOUT_S='900',WAIT='1',WAIT_TIMEOUT_S='600')
    subprocess.run(['bash','experiments/queue/run_safe.sh',str(queue.relative_to(ROOT))],cwd=ROOT,env=env,check=True)
    out=ROOT/'experiments/results/dvs_native'/f'{tag}.json';result=json.loads(out.read_text());assert result['status']=='completed';check_sources(result)
    for sample in result['work_samples']:
        assert all(s['formula_coverage_complete'] for s in sample['stages'].values())
    publish([queue,out,out.with_suffix('.progress.pt')],f'Record AWS coarse tied pool{pool} {stage} seed{seed}')
    return result


def gate(result,seed):
    name='aws_coarse_native_20261002T212600Z_coarse_matchedclock_s6' if seed==6 else 'aws_coarse_native_replication_20261002T212900Z_coarse_matchedclock_s7'
    path=ROOT/'experiments/results/dvs_native'/f'{name}.json';baseline=json.loads(path.read_text());check_sources(baseline)
    gain=baseline['final']['nll']-result['final']['nll'];accuracy=result['final']['accuracy']-baseline['final']['accuracy']
    ratio=result['work']['whole_fit_unit_special_flops_estimate']/baseline['work']['whole_fit_unit_special_flops_estimate']
    passed=gain>=.05 and accuracy>=-.01 and ratio<=3 and result['parameters']<=15523
    return dict(reference=name,nll_gain=gain,accuracy_gain=accuracy,fitting_work_ratio=ratio,parameters=result['parameters'],passed=passed)


def main():
    started=time.monotonic()
    while not CONTRACT.exists():
        if time.monotonic()-started>600:raise TimeoutError('Numerical contract pending')
        time.sleep(2)
    contract=json.loads(CONTRACT.read_text());assert contract['contracts_passed'];check_sources(contract)
    completed=[]
    for pool in (2,8):
        result=job(pool,'smoke',6,24,8,2)
        assert result['small_fit_learning_passed'] and result['max_rss_kb']<1000000
        completed.append(result['args']['tag'])
    pilots=[]
    for pool in (2,8):
        result=job(pool,'pilot',6,256,192,4);pilots.append(dict(pool=pool,tag=result['args']['tag'],final=result['final'],gate=gate(result,6)))
        completed.append(result['args']['tag'])
    qualifiers=[p for p in pilots if p['gate']['passed']]
    selected=min(qualifiers,key=lambda p:p['final']['nll']) if qualifiers else None
    confirmation=None
    if selected:
        result=job(selected['pool'],'confirmation',7,256,192,4)
        confirmation=dict(tag=result['args']['tag'],final=result['final'],gate=gate(result,7));completed.append(result['args']['tag'])
    capacity_gain=pilots[0]['final']['nll']-pilots[1]['final']['nll']
    summary=dict(status='completed',tag=PREFIX,completed=completed,pilots=pilots,selected_pool=selected['pool'] if selected else None,confirmation=confirmation,
        full_fit_nomination_pass=bool(selected and confirmation['gate']['passed']),tied8_over_tied2_nll_gain=capacity_gain,added_state_nomination_pass=capacity_gain>=.05,
        scope='Fixed exploratory integrated coarse/shared-map fits; reused DEV epoch/variant selection, independent fitting seed only. No full fit admitted automatically.',wall_s=time.monotonic()-started)
    assert not SUMMARY.exists();SUMMARY.write_text(json.dumps(summary,indent=2)+'\n');publish([SUMMARY], 'Record gated AWS coarse tied-family pilot and confirmation summary')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
