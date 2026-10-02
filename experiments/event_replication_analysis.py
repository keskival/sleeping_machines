"""Complete three-seed development replication inventory, never best-seed selection."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OLD='experiments/gym/plans/aws_split_event_20261001T230029Z/manifest.json'
NEW='experiments/gym/plans/aws_event_replication_20261002T005408Z/manifest.json'


def cell(args):
    return (args['task'],args['sources'],args['shared_maps'],args['protected_pairs'],args['time_input'])


def analyze():
    desired={('paired_timing',4,False,0,'observed'),('paired_timing',4,False,0,'rank'),
             ('order',16,True,0,'observed'),('order',16,True,2,'observed')}
    groups={key:{} for key in desired};hashes={}
    for manifest in (OLD,NEW):
        plan=json.loads((ROOT/manifest).read_text())
        for job in plan['jobs']:
            if job['stage']!='pilot':continue
            path=ROOT/job['result']
            if not path.exists():
                if manifest==NEW:raise ValueError('All reserved pilots required: '+job['tag'])
                continue
            row=json.loads(path.read_text());key=cell(row['args'])
            if key not in desired:continue
            if row['status']!='completed' or row['args']['tag']!=job['tag']:
                raise ValueError('Completed exact result identity required')
            args=row['args'];seed=args['seed'];work=row['work']
            if seed in groups[key]:raise ValueError('Duplicate seed/cell')
            if (args['fit_targets'],args['dev_targets'],args['epochs'],args['payload'],args['heads'],args['depth'],args['pool'],args['update_targets'],args['lr'])!=(128,256,4,8,2,8,2,64,.003):
                raise ValueError('Matched frozen configuration required')
            if work['fitting_query_targets']!=512:raise ValueError('Fitting denominator changed')
            dev=row['final']['dev']
            groups[key][seed]=dict(seed=seed,result=job['result'],accuracy=dev['accuracy'],nll=dev['nll'],
                cleared_state_accuracy=row['final']['cleared_state']['accuracy'],
                long_gap_accuracy={k:v['accuracy'] for k,v in row['final'].get('long_gaps',{}).items()},
                whole_fit_gflops=work['total_training_arithmetic_flops']/1e9,
                fit_mflops_per_query=work['total_training_arithmetic_flops']/512/1e6,
                inference_mflops_per_query=work['inference_arithmetic_flops_per_query']/1e6,
                fitting_special_function_evaluations=work['training_special_function_evaluations'],
                available_receivers=work['available_receivers'],selected_updates_per_event=work['selected_updates_per_event'],
                key_scores_per_event=work['key_scores_per_event'],wall_s=row['wall_s'],
                data_sha256=row['data_sha256'])
            hashes[job['result']]=hashlib.sha256(path.read_bytes()).hexdigest()
    summaries=[]
    for key,rows in sorted(groups.items()):
        if set(rows)!={6,7,8}:raise ValueError('All three declared seeds required')
        ordered=[rows[i] for i in (6,7,8)]
        if any(r['data_sha256']!=ordered[0]['data_sha256'] for r in ordered):
            raise ValueError('Fixed data populations differ')
        accuracies=[r['accuracy'] for r in ordered]
        summaries.append(dict(configuration=dict(zip(('task','sources','shared_maps','protected_pairs','time_input'),key)),
            seeds=ordered,mean_accuracy=statistics.mean(accuracies),
            sample_std_accuracy=statistics.stdev(accuracies),min_accuracy=min(accuracies),max_accuracy=max(accuracies)))
    return dict(status='completed',cells=summaries,result_sha256=hashes,
        arithmetic_convention='2 FLOPs/MAC; special functions separate. Whole-fit and per-query denominators512 presentations for every seed.',
        scope='Three training seeds on unchanged fit/dev populations, cells selected after seed6. Development-selected checkpoints; no independent confirmation, seed selection or supremacy claim. Shared maps also replace source embeddings.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise ValueError('Never overwrite prior evidence')
    result=analyze();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cells=len(result['cells']))))


if __name__=='__main__':main()
