"""Batched native gradient/state and actual local/pair driver recovery contracts."""
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_batched_benchmark as B
import dvs_native_contracts as C


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--data',required=True);p.add_argument('--controls',required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    common=['--data',a.data,'--controls',a.controls,'--fit','4','--dev','2','--epochs','2','--update-targets','2']
    config=B.parser().parse_args(['--tag','contract']+common)
    fitting,_,metadata=B.N.load(config);rows=fitting[:4];seed=1237
    reference=B.N.make_model(config).double();batch=copy.deepcopy(reference)
    independent=[];states=[]
    for row in rows:
        z,state=B.N.predict(reference,row,seed,True);independent.append(z);states.append(state)
    torch.stack([F.cross_entropy(z[None],torch.tensor([row['target']])) for z,row in zip(independent,rows)]).sum().backward()
    logits,state,_=B.forward(batch,rows,seed)
    F.cross_entropy(logits,torch.tensor([row['target'] for row in rows]),reduction='sum').backward()
    torch.testing.assert_close(logits,torch.stack(independent),rtol=1e-9,atol=1e-10)
    for (name,first),(other_name,second) in zip(reference.named_parameters(),batch.named_parameters()):
        assert name==other_name
        first_grad=torch.zeros_like(first) if first.grad is None else first.grad
        second_grad=torch.zeros_like(second) if second.grad is None else second.grad
        torch.testing.assert_close(first_grad,second_grad,rtol=1e-8,atol=1e-9)
    for index,original in enumerate(states):
        for (depth,head,source,unit),value in original.memories.items():
            torch.testing.assert_close(state['memories'][depth][index,head*config.pool+unit],value,rtol=1e-9,atol=1e-10)
            torch.testing.assert_close(state['arrivals'][depth][index,head*config.pool+unit],original.arrivals[(depth,head,source,unit)],rtol=1e-9,atol=1e-10)
        context,times=original.contexts[0]
        torch.testing.assert_close(state['context'][index].reshape(-1),context,rtol=1e-9,atol=1e-10)
        torch.testing.assert_close(state['context_times'][index],times,rtol=1e-9,atol=1e-10)
    # Independent enumeration of the same finite terminal conditional risk.
    batch.zero_grad();logits,_,pairs=B.forward(batch,rows,seed,True)
    values,weights=pairs;targets=torch.tensor([row['target'] for row in rows])
    vectorized=B.loss(logits,pairs,targets)
    explicit=sum(weights[i,j]*F.cross_entropy(values[i,j][None],targets[i:i+1])
        for i in range(len(rows)) for j in range(config.pool**2))
    torch.testing.assert_close(vectorized,explicit,rtol=1e-12,atol=1e-12)
    params=list(batch.parameters());one=torch.autograd.grad(vectorized,params,retain_graph=True,allow_unused=True)
    two=torch.autograd.grad(explicit,params,allow_unused=True)
    for first,second in zip(one,two):
        if first is None:assert second is None
        else:torch.testing.assert_close(first,second,rtol=1e-9,atol=1e-10)
    recoveries=[]
    with B.activate(),tempfile.TemporaryDirectory(prefix='dvs-batch-contract-') as temp:
        for mode in ('local','pairs'):
            def args(tag):return B.parser().parse_args(['--tag',tag]+common+['--terminal-risk',mode])
            continuous=args(mode+'_continuous');first=B.N.run(continuous,temp)
            for sample in first['work_samples']:
                for stage in sample['stages'].values():
                    assert stage['formula_coverage_complete'],stage['unsupported_floating_operators']
            recovered=args(mode+'_recovered');recovered.stop_after_updates=1;B.N.run(recovered,temp)
            recovered.stop_after_updates=None;recovered.resume=True;second=B.N.run(recovered,temp)
            x=torch.load(Path(temp)/(continuous.tag+'.progress.pt'),weights_only=False)
            y=torch.load(Path(temp)/(recovered.tag+'.progress.pt'),weights_only=False)
            for name in ('online_model','optimizer','best_state','best','cursor','torch_rng'):C.equal(x[name],y[name])
            for name in ('final','activity','work','work_samples','selected_epoch'):C.equal(first[name],second[name])
            recoveries.append(mode)
    result=dict(status='completed',args=vars(a),contracts_passed=4,
        full_shape=dict(payload=16,depth=2,heads=2,pool=2),data=metadata,
        every_parameter_gradient_and_persistent_state_matches_independent_clips=True,
        terminal_pair_risk_and_every_derivative_match_explicit_enumeration=True,
        actual_driver_recovery_modes=recoveries,formula_coverage_complete=True,
        source_sha256={**B.sources(),'experiments/dvs_native_contracts.py':B.N.sha(ROOT/'experiments/dvs_native_contracts.py')},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Native independent-clip vectorization with unchanged local teacher, and optional exact terminal content-risk derivatives; prior routes remain surrogate. Actual local/pair model/Adam/cursor/RNG recovery and complete-window accounting. No fit-quality or total-resource advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
