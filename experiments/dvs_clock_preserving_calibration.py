"""Preregistered frozen native routing calibration on disjoint unseen FIT inputs."""
import argparse
import copy
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from clock_preserving_temperature_contracts import make
from dvs_key_score_decomposition import replay
from race_language_screen import RaceAudit


class CalibrationAudit(RaceAudit):
    def formula(self,func,args,kwargs,out):
        if str(func).split('.')[1].rstrip('_')=='cumsum' and args[0].is_floating_point():
            x=args[0];dimension=args[1];groups=x.numel()//x.shape[dimension]
            return x.numel()-groups,0,0,'inclusive cumulative sum additions'
        return super().formula(func,args,kwargs,out)


@torch.no_grad()
def evaluate(model,rows,seeds):
    cases=[];trace=None
    for draw,seed in enumerate(seeds):
        for index,row in enumerate(rows):
            if draw==0 and index==0:
                with CalibrationAudit() as audit:z,state,_=replay(model,row,seed)
                trace=audit.result();assert trace['formula_coverage_complete'],trace['unsupported_floating_operators']
            else:z,state,_=replay(model,row,seed)
            assert (state.events,state.candidate_scores,state.selected_updates,state.counterfactual_values)==(21,168,84,0)
            cases.append(dict(index=row['index'],identity=row['identity'],target=row['target'],draw=draw,noise_seed=seed,
                nll=float(F.cross_entropy(z[None],torch.tensor([row['target']]))),predicted=int(z.argmax()),
                probabilities=z.softmax(-1).tolist(),key_scores=state.candidate_scores,selected_updates=state.selected_updates,
                final_ready_time=float(state.contexts[0][1].max()),state_tensor_bytes=state.storage()['persistent_tensor_bytes']))
    return dict(nll=float(np.mean([c['nll'] for c in cases])),accuracy=float(np.mean([c['predicted']==c['target'] for c in cases])),
        cases=cases,representative_first_prefix_work=trace,
        representative_first_prefix_unit_special_flops_estimate=trace['arithmetic_flops']+trace['special_function_evaluations'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--contracts',required=True);p.add_argument('--native',action='append',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    contract=json.loads((ROOT/a.contracts).read_text());assert contract['status']=='completed'
    for f,digest in contract['source_sha256'].items():assert N.sha(ROOT/f)==digest
    torch.set_num_threads(1);begin=time.perf_counter();outcomes=[];parents=[];gates=[];sources=dict(contract['source_sha256'])
    old=set(np.linspace(256,983,16,dtype=int));available=[i for i in range(256,984) if i not in old]
    indices=[available[i] for i in np.linspace(0,len(available)-1,16,dtype=int)]
    seeds=[411173+1009*k for k in range(4)];settings=[(1.,'all'),(2.,'all'),(4.,'all'),(2.,'layer0')]
    assert not set(indices)&old
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        config=argparse.Namespace(**parent['args'])
        assert (config.fit,config.dev,config.epochs,config.pool,config.depth,config.heads)==(256,192,4,2,2,2)
        for f,digest in parent['source_sha256'].items():assert N.sha(ROOT/f)==digest
        sources.update(parent['source_sha256']);expanded=copy.copy(config);expanded.fit=984
        fit,_,data=N.load(expanded);assert data==parent['data'];selected=[fit[i] for i in indices]
        cp=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(cp,weights_only=False)
        assert saved['cursor']['epoch']==5 and saved['source_sha256']==parent['source_sha256']
        initial=N.make_model(config,fast=False).state_dict()
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp),seed=config.seed,
            original_trained_core_fit_gflops=parent['work']['whole_fit_unit_special_flops_estimate']/1e9))
        for encoder,weights in [('initial',initial),('fixed_pass4',saved['online_model'])]:
            model=make(config,weights);before={n:v.detach().clone() for n,v in model.state_dict().items()};local=[]
            for temperature,scope in settings:
                model.route_temperature=temperature;model.temperature_scope=scope;quality=evaluate(model,selected,seeds)
                record=dict(seed=config.seed,encoder=encoder,temperature=temperature,scope=scope,**quality)
                local.append(record);outcomes.append(record)
                print(json.dumps({k:record[k] for k in ('seed','encoder','temperature','scope','nll','accuracy')}),flush=True)
            assert all(torch.equal(before[n],v.detach()) for n,v in model.state_dict().items())
            if encoder=='fixed_pass4':
                baseline,candidate=local[:2];gain=baseline['nll']-candidate['nll'];accuracy_gain=candidate['accuracy']-baseline['accuracy']
                gates.append(dict(seed=config.seed,nll_gain=gain,accuracy_gain=accuracy_gain,
                    smoke_nomination_passed=gain>=.02 and accuracy_gain>=0.))
    own=['experiments/dvs_clock_preserving_calibration.py','sleeping_machines/operation_audit.py']
    sources.update({f:N.sha(ROOT/f) for f in own})
    result=dict(status='completed',args=vars(a),contracts_sha256=N.sha(ROOT/a.contracts),parents=parents,rows=outcomes,
        gates=gates,integrated_smoke_nomination_passed=all(g['smoke_nomination_passed'] for g in gates),
        indices=indices,noise_seeds=seeds,disjoint_from_prior_cohort=True,distinct_unused_fit_inputs=16,
        prefix_evaluations=1024,optimizer_steps=0,development_evaluations=0,all_weights_bitwise_preserved=True,
        source_sha256=sources,whole_audit_flops=None,wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hardware=dict(host=platform.node(),device='CPU',threads=1),
        scope='Frozen actual native calibrated race choices; current-state common-rate first-time law retained, '
            'later state/timing allowed to change. Two fixed-four-pass producers plus initial reservoirs, '
            '16 disjoint producer-unseen FIT inputs/four noisy histories/all16 model configurations. '
            'Mean per-history loss/accuracy, not ensemble loss. Winner-only values and real sparse writes remain. '
            'Prior training and candidate discovery paid; representative traced work includes calibration transforms '
            'and explicit cumulative-sum formula, not whole-audit FLOPs, traffic, energy or hardware advantage. '
            'No fitting, DEV/test, installed positive-temperature training estimator or benchmark superiority.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
