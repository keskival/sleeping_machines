"""Freeze a public configuration and epoch count by completed DEV-only screens."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.public_benchmarks.data import sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--dataset',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    plan=json.loads((ROOT/a.manifest).read_text());rows=[]
    for jobs in plan['slots'].values():
        for job in jobs:
            if job['kind']!='public' or '_screen_' not in job['tag'] or a.dataset not in job['tag']:continue
            path=ROOT/job['result'];r=json.loads(path.read_text())
            assert r['status']=='completed' and r['args']['stage']=='screen' and r['test'] is None
            assert r['args']['dataset']==a.dataset and r['development']['nll']==r['development']['nll']
            rows.append((r['development']['nll'],r['work']['whole_fit_flops_estimate'],r['parameters'],job['result'],r))
    assert len(rows)==3
    chosen=min(rows,key=lambda r:r[:4]);selected=chosen[4]
    result=dict(status='completed',args=vars(a),source_sha256={'experiments/public_benchmarks/select.py':sha(Path(__file__))},
                selected_screen=chosen[3],selected_screen_sha256=sha(ROOT/chosen[3]),
                rule='Minimum fixed TRAIN-only DEV NLL, then estimated fitting work, then parameters, then path; no TEST access.',
                candidates=[dict(path=r[3],sha256=sha(ROOT/r[3]),dev_nll=r[0],work=r[1],parameters=r[2]) for r in rows],
                final_epochs=selected['best_epoch'],final_seeds=[6,7,8],test=None)
    out=ROOT/'experiments/results/public_benchmarks'/(a.tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
