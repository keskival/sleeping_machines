"""Separate categorical/time derivatives, deep VJP and native driver recovery."""
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
import dvs_state_choice_credit_benchmark as B
import dvs_native_contracts as C


def isolated_contract():
    torch.manual_seed(19)
    scores=torch.randn(3,2,2,dtype=torch.float64,requires_grad=True)
    values=torch.randn(3,2,2,3,dtype=torch.float64,requires_grad=True)
    noise=torch.tensor([[.4,1.7],[1.2,.5]],dtype=torch.float64)
    reference_scores=scores.detach().clone().requires_grad_()
    reference_values=values.detach().clone().requires_grad_()
    selected,delay,winner=B.REFERENCE_ROUTE.apply(reference_scores,reference_values,noise)
    target=torch.tensor([0,1,2]);coefficients=delay.new_tensor([[1.3,2.1],[.7,1.1],[1.9,.3]])
    reference_loss=F.cross_entropy(selected.sum(1),target,reduction='sum')+(coefficients*delay.square()).sum()
    reference_loss.backward()
    # Independent finite conditional risk, differentiating only the choice probabilities.
    conditional_scores=scores.detach()[:,0].clone().requires_grad_()
    outcomes=[]
    for option in range(2):
        candidate=selected.detach().clone();candidate[:,0]=values.detach()[:,0,option]
        outcomes.append(F.cross_entropy(candidate.sum(1),target,reduction='none'))
    risks=torch.stack(outcomes,-1)
    exact=torch.autograd.grad((conditional_scores.softmax(-1)*risks).sum(),conditional_scores)[0]
    # Independent ordinary autograd on the raw exponential clocks, away from ties.
    smooth_scores=scores.detach().clone().requires_grad_()
    raw=(noise[None]/smooth_scores.exp()).min(-1).values
    physical=.001+.010*raw/(1+raw)
    native_time=torch.autograd.grad((coefficients*physical.square()).sum(),smooth_scores)[0]
    assert native_time[:,0].norm()>1e-8
    box=dict(head=0,credit=exact,audit=True)
    chosen,timing,indices=B.ChoiceWithTimingRoute.apply(scores,values,noise,box)
    loss=F.cross_entropy(chosen.sum(1),target,reduction='sum')+(coefficients*timing.square()).sum()
    loss.backward()
    torch.testing.assert_close(chosen,selected,rtol=0,atol=0)
    torch.testing.assert_close(timing,delay,rtol=0,atol=0)
    torch.testing.assert_close(indices,winner,rtol=0,atol=0)
    torch.testing.assert_close(scores.grad[:,0],exact+native_time[:,0],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(scores.grad[:,1],reference_scores.grad[:,1],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(values.grad,reference_values.grad,rtol=0,atol=0)
    assert (scores.grad[:,0]-reference_scores.grad[:,0]).norm()>1e-6


def integrated_contract(model,rows):
    reference=copy.deepcopy(model);seed=1237
    targets=torch.tensor([row['target'] for row in rows]);site=9*model.depth
    with B.S.trace_site(site,0) as old:
        original,state,_=B.K.forward(reference,rows,seed)
    old[0]['scores'].retain_grad()
    F.cross_entropy(original,targets,reduction='sum').backward(retain_graph=True)
    loss,new_state,detail=B.corrected_loss(model,rows,seed,1,audit=True)
    torch.testing.assert_close(original,detail['logits'],rtol=0,atol=0)
    for key in ('memories','arrivals','seen','context','context_times','winners'):C.equal(state[key],new_state[key])
    actual,shadow=detail['record'],detail['shadow']
    for key in ('scores','values','first'):torch.testing.assert_close(actual[key],shadow[key],rtol=0,atol=0)
    changed=detail['shadow_state']['winners'][site]
    assert torch.equal(changed[:,0],1-actual['winner'])
    assert torch.equal(changed[:,1],new_state['winners'][site][:,1])
    # Individual legal alternative replays match the simultaneous independent-clip shadow.
    from dvs_counterfactual_route_audit import force_and_trace
    for clip in range(len(rows)):
        with torch.no_grad(),force_and_trace(site,clip,0,int(1-actual['winner'][clip])):
            independent,_,_=B.K.forward(model,rows,seed)
        alternative=F.cross_entropy(independent[clip:clip+1],targets[clip:clip+1])
        torch.testing.assert_close(alternative,detail['outcome_losses'][clip,1-actual['winner'][clip]],rtol=1e-12,atol=1e-12)
    conditional=actual['scores'].detach()[:,0].clone().requires_grad_()
    exact=torch.autograd.grad((conditional.softmax(-1)*detail['outcome_losses']).sum(),conditional)[0]
    torch.testing.assert_close(actual['credit'],exact,rtol=1e-12,atol=1e-12)
    actual['scores'].retain_grad();loss.backward()
    # Only selected-head times are needed for this independent native timing term.
    expected_time=torch.zeros_like(exact)
    derivative=-actual['error_delay'][:,0]*.010*actual['first']/(1+actual['first']).square()
    expected_time.scatter_add_(1,actual['winner'][:,None],derivative[:,None])
    assert expected_time.norm()>1e-12
    torch.testing.assert_close(actual['scores'].grad[:,0],exact+expected_time,rtol=1e-11,atol=1e-12)
    residual=actual['scores'].grad-old[0]['scores'].grad
    # The change reaches every earlier parameter through the real producer graph.
    correction=torch.autograd.grad(old[0]['scores'],tuple(reference.parameters()),
        grad_outputs=residual,allow_unused=True)
    affected=[]
    for (name,first),(other,second),delta in zip(reference.named_parameters(),model.named_parameters(),correction):
        assert name==other
        zero=torch.zeros_like(first)
        expected=(zero if first.grad is None else first.grad)+(zero if delta is None else delta)
        received=zero if second.grad is None else second.grad
        torch.testing.assert_close(received,expected,rtol=1e-8,atol=1e-9)
        if delta is not None and delta.norm()>1e-12:affected.append(name)
    assert any('content' in name for name in affected)
    assert any('key' in name or 'query' in name for name in affected)
    return affected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--data',required=True);p.add_argument('--controls',required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused plain tag required')
    isolated_contract()
    common=['--data',a.data,'--controls',a.controls,'--fit','4','--dev','2','--epochs','2','--update-targets','2']
    config=B.N.parser().parse_args(['--tag','contract']+common)
    fitting,_,metadata=B.N.load(config)
    affected=integrated_contract(B.N.make_model(config).double(),fitting[:4])
    with tempfile.TemporaryDirectory(prefix='dvs-state-choice-') as temp:
        continuous=B.N.parser().parse_args(['--tag','continuous']+common);first=B.run(continuous,temp)
        for sample in first['work_samples']:
            for stage in sample['stages'].values():
                assert stage['formula_coverage_complete'],stage['unsupported_floating_operators']
        recovered=B.N.parser().parse_args(['--tag','recovered']+common)
        recovered.stop_after_updates=1;B.run(recovered,temp)
        recovered.stop_after_updates=None;recovered.resume=True;second=B.run(recovered,temp)
        x=torch.load(Path(temp)/'continuous.progress.pt',weights_only=False)
        y=torch.load(Path(temp)/'recovered.progress.pt',weights_only=False)
        for name in ('online_model','optimizer','best_state','best','cursor','torch_rng'):C.equal(x[name],y[name])
        for name in ('final','activity','work','work_samples','selected_epoch','credit_protocol'):C.equal(first[name],second[name])
    result=dict(status='completed',args=vars(a),contracts_passed=4,data=metadata,
        exact_conditional_choice_matches_independent_autograd=True,native_timing_gradient_retained_and_nonzero=True,
        winning_payload_and_uncorrected_head_gradients_preserved=True,factual_forward_and_state_unchanged=True,
        actual_alternative_commit_and_loss_match_independent_replays=True,
        every_parameter_gradient_change_matches_upstream_score_residual_vjp=True,affected_parameter_names=affected,
        actual_model_adam_cursor_rng_recovery=True,formula_coverage_complete=True,
        source_sha256={**B.sources(),'experiments/dvs_native_contracts.py':B.N.sha(ROOT/'experiments/dvs_native_contracts.py')},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='One node exact conditional write-choice utility with original native pathwise timing; other route teachers and downstream timing jumps remain approximate. No fit-quality, global unbiasedness or superiority claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
