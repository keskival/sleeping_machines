"""Representative whole optimizer-step arithmetic, all stages charged.

Same four fitting queries for each architecture; one warm Adam step followed by
one measured step. This is not reconstructed historical whole-run work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e120_shared_bench import BUILDERS,inputs,loss_for
from e123_synthetic_baselines import EventTransformer,CountedTransformer,batch,predict
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.operation_audit import OperationAudit


def step(net,rows,kind,optimizer,measured):
    net.train();optimizer.zero_grad(set_to_none=True)
    def forward():
        if kind=='common':return net(**inputs(rows))[0]
        b,t,mask,_=batch(rows);return predict(net,b,t,mask,rows)
    if not measured:
        loss=loss_for(forward(),rows);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        return None
    stages={}
    with OperationAudit() as counter:loss=loss_for(forward(),rows)
    stages['forward_and_loss']=counter.result()
    with OperationAudit() as counter:loss.backward()
    stages['backward']=counter.result()
    with OperationAudit() as counter:torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
    stages['gradient_clipping']=counter.result()
    with OperationAudit() as counter:optimizer.step()
    stages['optimizer']=counter.result()
    return dict(stages=stages,total_arithmetic_flops=sum(v['arithmetic_flops'] for v in stages.values()),
        total_special_function_evaluations=sum(v['special_function_evaluations'] for v in stages.values()),
        formula_coverage_complete=all(v['formula_coverage_complete'] for v in stages.values()),
        loss=float(loss.detach()),queries=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--tasks',default='language,market,temporal,mnist,dvs');a=p.parse_args()
    out=Path('experiments/results/e172')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);started=time.perf_counter();rows=[]
    for name in a.tasks.split(','):
        common_path=Path('experiments/results/e120')/f'{name}_d8_20260929.json'
        record=json.loads(common_path.read_text());task=BUILDERS[name](record['args']['fit'],record['args']['dev'],6)
        samples=task.fit[:4]
        common=torch.load(common_path.with_suffix('.pt'),weights_only=False,map_location='cpu')
        net=SharedEventModel(**common['config']);net.load_state_dict(common['state_dict'])
        optimizer=torch.optim.Adam(net.parameters(),lr=.003)
        step(net,samples,'common',optimizer,False);c=step(net,samples,'common',optimizer,True)
        reference_path=Path('experiments/results/e123')/(f'{name}_tf_counts_d32_l2_s6.json' if name=='dvs' else f'{name}_tf_d32_l2_s6.json')
        reference=torch.load(reference_path.with_suffix('.pt'),weights_only=False,map_location='cpu')
        net=(CountedTransformer(task.config['bands'],task.config['classes'],32,2) if name=='dvs' else
             EventTransformer(task.config['bands'],task.config['classes'],32,2,1.2))
        net.load_state_dict(reference['state_dict']);optimizer=torch.optim.Adam(net.parameters(),lr=.003)
        step(net,samples,'transformer',optimizer,False);r=step(net,samples,'transformer',optimizer,True)
        row=dict(task=name,common=c,transformer=r,query_ids=[q.identity for q in samples],
                 query_events=[len(q.prefix.channels) for q in samples],batch_size=4,
                 ratio_common_to_transformer=c['total_arithmetic_flops']/r['total_arithmetic_flops'])
        rows.append(row);print(json.dumps({'task':name,'ratio':row['ratio_common_to_transformer'],
            'coverage':(c['formula_coverage_complete'],r['formula_coverage_complete'])}),flush=True)
    result=dict(status='completed',rows=rows,hardware=dict(platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),
        scope='Representative complete optimizer steps at saved parameters, warm fresh Adam moments. Four identical fitting queries/model; actual batch padding charged. Forward/loss, backward, gradient clipping and optimizer all included. Not historic total training or evidence fitting/preprocessing/inherited work. Fused scalar kernels use declared shape formulas; arithmetic FLOPs, transcendental evaluations and non-arithmetic control separated. Unsupported floating ops are listed. No physical memory traffic or energy measured.',
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in
             (Path(__file__),Path('sleeping_machines/operation_audit.py'),Path('sleeping_machines/shared_event.py'),Path('experiments/e123_synthetic_baselines.py'))},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
