"""Finite paired-route expectations and native two/eight-candidate integration."""
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
import dvs_paired_choice_benchmark as P
import dvs_native_contracts as C


def finite_contracts():
    trials=[];torch.manual_seed(127)
    for pool in (2,3,8,64):
        scores=torch.randn(pool,dtype=torch.float64,requires_grad=True)
        losses=torch.rand(pool,dtype=torch.float64)*3
        probabilities=scores.softmax(-1)
        exact=torch.autograd.grad((probabilities*losses).sum(),scores)[0]
        pi=probabilities.detach();w=torch.arange(pool).repeat_interleave(pool)
        i=torch.arange(pool).repeat(pool);eligible=w!=i;w=w[eligible];i=i[eligible]
        expanded=pi[None].expand(len(w),-1)
        for exploration in (0.,.1,.5):
            all_q=P.alternative_probabilities(pi[None].expand(pool,-1),torch.arange(pool),exploration)
            torch.testing.assert_close(all_q.sum(-1),torch.ones(pool,dtype=torch.float64),rtol=1e-12,atol=1e-12)
            assert torch.equal(all_q.diagonal(),torch.zeros(pool,dtype=torch.float64))
            q=all_q[w];mass=pi[w]*q.gather(1,i[:,None]).squeeze(-1)
            estimate=P.paired_choice_credit(expanded,w,i,q,losses[w],losses[i])
            mean=(mass[:,None]*estimate).sum(0)
            torch.testing.assert_close(mean,exact,rtol=1e-12,atol=1e-12)
            shifted=P.paired_choice_credit(expanded,w,i,q,losses[w]+7,losses[i]+7)
            torch.testing.assert_close(shifted,estimate,rtol=1e-12,atol=1e-12)
            constant=torch.ones(len(w),dtype=torch.float64)*1.23
            zero=P.paired_choice_credit(expanded,w,i,q,constant,constant)
            assert torch.equal(zero,torch.zeros_like(zero))
            importance=pi[i]/q.gather(1,i[:,None]).squeeze(-1)
            assert float(importance.max())<=1/(1-exploration)+1e-12
            if pool==2:torch.testing.assert_close(estimate,exact[None].expand_as(estimate),rtol=1e-12,atol=1e-12)
            variance=float((mass*(estimate-exact[None]).square().sum(-1)).sum())
            trials.append(dict(pool=pool,exploration=exploration,maximum_importance=float(importance.max()),
                finite_score_gradient_variance=variance,gradient_mean_matches_autograd=True))
    return trials


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--data',required=True);p.add_argument('--controls',required=True)
    a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    trials=finite_contracts()
    common=['--data',a.data,'--controls',a.controls,'--fit','4','--dev','2','--epochs','2','--update-targets','3']
    config=P.parser().parse_args(['--tag','contract']+common)
    fitting,_,metadata=P.N.load(config);rows=fitting[:4]
    reference=P.N.make_model(config).double();paired=copy.deepcopy(reference)
    first,old,_=P.B.corrected_loss(reference,rows,1237,1)
    second,new,detail=P.corrected_loss(paired,rows,1237,1)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    for key in ('memories','arrivals','seen','context','context_times','winners'):C.equal(old[key],new[key])
    first.backward();second.backward()
    for (name,one),(other,two) in zip(reference.named_parameters(),paired.named_parameters()):
        assert name==other
        zero=torch.zeros_like(one)
        torch.testing.assert_close(zero if one.grad is None else one.grad,zero if two.grad is None else two.grad,rtol=1e-8,atol=1e-9)
    large=P.parser().parse_args(['--tag','large']+common+['--pool','8'])
    model=P.N.make_model(large).double()
    with torch.no_grad():base,factual,_=P.K.forward(model,rows,1237)
    loss,state,detail=P.corrected_loss(model,rows,1237,1,audit=True)
    torch.testing.assert_close(base,detail['logits'],rtol=0,atol=0)
    for key in ('memories','arrivals','seen','context','context_times','winners'):C.equal(factual[key],state[key])
    actual=detail['record'];shadow=detail['shadow']
    for key in ('scores','values','first'):torch.testing.assert_close(actual[key],shadow[key],rtol=0,atol=0)
    assert bool((detail['alternative']!=actual['winner']).all())
    assert torch.equal(detail['shadow_state']['winners'][detail['site']][:,detail['head']],detail['alternative'])
    from dvs_counterfactual_route_audit import force_and_trace
    targets=torch.tensor([r['target'] for r in rows])
    for clip in range(len(rows)):
        with torch.no_grad(),force_and_trace(detail['site'],clip,detail['head'],int(detail['alternative'][clip])):
            z,_,_=P.K.forward(model,rows,1237)
        expected=F.cross_entropy(z[clip:clip+1],targets[clip:clip+1])
        torch.testing.assert_close(expected,detail['alternative_loss'][clip],rtol=1e-12,atol=1e-12)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    recoveries=[]
    with tempfile.TemporaryDirectory(prefix='dvs-paired-credit-') as temp:
        for pool in (2,8):
            def args(tag):return P.parser().parse_args(['--tag',tag]+common+['--pool',str(pool)])
            continuous=args('continuous_'+str(pool));first=P.run(continuous,temp)
            for sample in first['work_samples']:
                for stage in sample['stages'].values():
                    assert stage['formula_coverage_complete'],stage['unsupported_floating_operators']
            recovered=args('recovered_'+str(pool));recovered.stop_after_updates=1;P.run(recovered,temp)
            recovered.stop_after_updates=None;recovered.resume=True;second=P.run(recovered,temp)
            x=torch.load(Path(temp)/(continuous.tag+'.progress.pt'),weights_only=False)
            y=torch.load(Path(temp)/(recovered.tag+'.progress.pt'),weights_only=False)
            for key in ('online_model','optimizer','best_state','best','cursor','torch_rng'):C.equal(x[key],y[key])
            for key in ('final','activity','work','work_samples','selected_epoch','credit_protocol'):C.equal(first[key],second[key])
            recoveries.append(dict(pool=pool,full_partial_windows=first['window_size_counts'],
                whole_fit_gflops=first['work']['whole_fit_unit_special_flops_estimate']/1e9,
                fit_mflops_per_target=first['work']['fit_unit_special_flops_per_target_estimate']/1e6,
                inference_mflops_per_prefix=first['work']['inference_unit_special_flops_per_target_estimate']/1e6,
                actual_model_adam_cursor_rng_recovery=True,formula_coverage_complete=True))
    result=dict(status='completed',args=vars(a),contracts_passed=5,finite_trials=trials,data=metadata,
        exact_two_route_native_state_and_every_parameter_gradient_nesting=True,
        eight_route_actual_alternative_commits_and_losses_match_individual_replays=True,
        native_timing_credit_retained=True,full_partial_driver_recovery=recoveries,
        source_sha256={**P.sources(),'experiments/dvs_native_contracts.py':P.N.sha(ROOT/'experiments/dvs_native_contracts.py')},
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Finite2/3/8/64-candidate estimator contracts; two-route exact native nesting; eight-route actual legal write/suffix and actual2/8-pool tiny-driver recovery with U3/partialU1 accounting. These temporary4-fit/2-dev two-pass updates are numerical readiness, not a benchmark gain, matched larger-pool screen or correction of full-history timing jumps.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
