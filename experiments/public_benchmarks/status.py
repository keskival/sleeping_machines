"""Read-only campaign evidence inventory; no pending scores or automatic win claims."""
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def inventory():
    rows=[];final={}
    for path in sorted((ROOT/'experiments/results/public_benchmarks').glob('*.json')):
        record=json.loads(path.read_text())
        if record.get('status')!='completed' or 'dataset' not in record.get('args',{}):continue
        args=record['args'];dev=record.get('development');test=record.get('test')
        row=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 dataset=args['dataset'],stage=args.get('stage','baseline'),epochs=args.get('epochs'),
                 dev_accuracy=dev['accuracy'] if dev else None,
                 test_accuracy=test['accuracy'] if test else None,
                 test_targets=test['targets'] if test else None,parameters=record.get('parameters'),
                 fitting_presentations=record.get('fitting_presentations'),work=record.get('work'),wall_s=record.get('wall_s'))
        rows.append(row)
        if test:
            assert args['stage']=='final'
            final.setdefault(args['dataset'],[]).append(row)
    return dict(status='inventory',rows=rows,final_datasets={name:dict(completed_seeds=len(group),
                mean_test_accuracy=statistics.mean(r['test_accuracy'] for r in group),
                seed_stddev=statistics.stdev(r['test_accuracy'] for r in group) if len(group)>1 else None,
                scope='Seed dispersion on the same TEST examples, not independent test sets or a generalization confidence interval.')
                for name,group in final.items()},
                claims=[],scope='Inventory only. DEV/pilot values are not public TEST scores. No new benchmark win certified; published variant, baseline and resource parity must be verified.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args();result=inventory()
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        out=Path(args.output);assert not out.exists();out.write_text(text)
    else:print(text)
