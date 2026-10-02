"""One-job deep replay stages and automatic publication of completed evidence."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PREFIX='aws_deep_replay_20261002T234200Z'
CONTRACT=ROOT/'experiments/results/diagnostics/aws_deep_replay_contracts_20261002T234100Z.json'
SUMMARY=ROOT/'experiments/results/diagnostics'/f'{PREFIX}_summary.json'
MODES=('teacher','factorized','replay')


def command(args):subprocess.run(args,cwd=ROOT,check=True)


def check_sources(result):
    import hashlib
    for name,digest in result['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def finite(value):
        if isinstance(value,float):assert math.isfinite(value)
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    finite(result)


def publish(paths,message):
    command(['git','add','-f',*[str(p.relative_to(ROOT)) for p in paths]])
    command(['git','commit','-m',message]);command(['git','pull','--rebase']);command(['git','push','origin','main'])


def job(mode,stage,seed,fit,dev,epochs):
    check_sources(json.loads(CONTRACT.read_text()))
    tag=f'{PREFIX}_d4_{mode}_{stage}_s{seed}';queue=ROOT/'experiments/queue'/f'{tag}.txt';assert not queue.exists()
    args=['experiments/aws_deep_replay.py','--tag',tag,'--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json',
        '--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--contracts',str(CONTRACT.relative_to(ROOT)),
        '--fit',str(fit),'--dev',str(dev),'--epochs',str(epochs),'--update-targets','16','--lr','.003','--payload','16','--depth','4',
        '--heads','2','--pool','2','--bins','4','--clock-step','.25','--weight-decay','0','--input-noise','0','--seed',str(seed),'--credit-mode',mode]
    queue.write_text(tag+' '+' '.join(args)+'\n')
    env=dict(os.environ,MEM_CAP_KB='6000000',MEM_CAP_RSS_KB='2000000',MIN_AVAIL_MB='8192',JOB_TIMEOUT_S='1200',WAIT='1',WAIT_TIMEOUT_S='600')
    subprocess.run(['bash','experiments/queue/run_safe.sh',str(queue.relative_to(ROOT))],cwd=ROOT,env=env,check=True)
    out=ROOT/'experiments/results/dvs_native'/f'{tag}.json';result=json.loads(out.read_text());assert result['status']=='completed';check_sources(result)
    for sample in result['work_samples']:assert all(s['formula_coverage_complete'] for s in sample['stages'].values())
    publish([queue,out,out.with_suffix('.progress.pt')],f'Record AWS depth4 {mode} {stage} seed{seed}')
    return result


def compare(arms):
    replay=arms['replay'];comparisons={}
    for mode in ('teacher','factorized'):
        control=arms[mode];gain=control['final']['nll']-replay['final']['nll'];acc=replay['final']['accuracy']-control['final']['accuracy']
        comparisons[mode]=dict(nll_gain=gain,accuracy_gain=acc,fit_work_ratio=replay['work']['whole_fit_unit_special_flops_estimate']/control['work']['whole_fit_unit_special_flops_estimate'],passed=gain>=.03 and acc>=-.01)
    return dict(comparisons=comparisons,passed=all(v['passed'] for v in comparisons.values()))


def main():
    started=time.monotonic();contract=json.loads(CONTRACT.read_text());assert contract['contracts_passed'];check_sources(contract);completed=[]
    for mode in MODES:
        result=job(mode,'smoke',7,24,8,2);assert result['small_fit_learning_passed'] and result['max_rss_kb']<1000000;completed.append(result['args']['tag'])
    pilots={}
    for mode in MODES:
        result=job(mode,'pilot',7,256,192,4);pilots[mode]=result;completed.append(result['args']['tag'])
    gate=compare(pilots);confirmation=None
    if gate['passed']:
        confirmations={}
        for mode in MODES:
            result=job(mode,'confirmation',8,256,192,4);confirmations[mode]=result;completed.append(result['args']['tag'])
        confirmation=dict(results={m:r['args']['tag'] for m,r in confirmations.items()},gate=compare(confirmations))
    summary=dict(status='completed',tag=PREFIX,depth=4,completed=completed,pilots={m:r['args']['tag'] for m,r in pilots.items()},pilot_gate=gate,confirmation=confirmation,
        full_fit_nomination_pass=bool(gate['passed'] and confirmation['gate']['passed']),wall_s=time.monotonic()-started,
        scope='Matched deep corrected full replay versus factorized control and original teacher; all costs paid; reused DEV epoch selection, no official test or supremacy claim')
    assert not SUMMARY.exists();SUMMARY.write_text(json.dumps(summary,indent=2)+'\n');publish([SUMMARY],'Record gated deep full-replay comparison and confirmation summary');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
