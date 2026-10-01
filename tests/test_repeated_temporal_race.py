"""Read-only numerical checks: no optimizer or fitting job is launched."""
import copy
import math

import torch

from sleeping_machines.sparse_race_language import TemporalRoute
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel
from sleeping_machines.repeated_arrival_race_language import RepeatedArrivalRaceLanguageModel
from sleeping_machines.repeated_temporal_race import poisson_clocks, temporal_arrivals


def test_one_arrival_preserves_reference_value_delay_gradients_and_rng():
    scores=torch.tensor([-.5,.2,1.],requires_grad=True)
    values=torch.tensor([[1.,2.],[-1.,.2],[.1,-2.]],requires_grad=True)
    with torch.random.fork_rng():
        torch.manual_seed(47)
        v,t,w=TemporalRoute.apply(scores,values)
        gradients=torch.autograd.grad(v.sum()+.3*t,(scores,values))
        rng=torch.get_rng_state()
        torch.manual_seed(47)
        vv,tt,ww=temporal_arrivals(scores,values,1)
        repeated=torch.autograd.grad(vv.sum()+.3*tt.sum(),(scores,values))
        assert torch.equal(rng,torch.get_rng_state())
    for actual,expected in [(vv[0],v),(tt[0],t),(ww[0],w),*zip(repeated,gradients)]:
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_aggregated_teacher_matches_explicit_per_arrival_rule_and_is_conserved():
    scores=torch.tensor([-.5,.2,1.],dtype=torch.float64,requires_grad=True)
    values=torch.tensor([[1.,2.],[-1.,.2],[.1,-2.]],dtype=torch.float64,requires_grad=True)
    errors=torch.tensor([[1.,-.3],[.4,.8],[-.1,2.],[.7,-.6]],dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(67)
        rates,times,winners=poisson_clocks(scores.detach(),4)
        torch.manual_seed(67)
        messages,_,_=temporal_arrivals(scores,values,4)
    credit,vcredit=torch.autograd.grad((messages*errors).sum(),(scores,values))
    expected=torch.zeros_like(scores);ev=torch.zeros_like(values)
    centered=values.detach()-values.detach().mean(0)
    previous=0.
    for time,winner,error in zip(times,winners,errors):
        contribution=rates*(time-previous)*(centered@error)
        contribution[winner]-=contribution.sum()
        expected+=contribution;ev[winner]+=error;previous=time
    torch.testing.assert_close(credit,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(vcredit,ev,rtol=0,atol=0)
    assert abs(float(credit.sum()))<1e-12


def test_realized_delay_derivative_matches_finite_difference_at_fixed_marks():
    scores=torch.tensor([-.4,.2,1.],dtype=torch.float64,requires_grad=True)
    values=torch.zeros(3,2,dtype=torch.float64)
    coefficients=torch.tensor([.2,-.3,.5,.8],dtype=torch.float64)
    def forward(s):
        with torch.random.fork_rng():
            torch.manual_seed(71)
            _,delays,marks=temporal_arrivals(s,values,4)
            return (delays*coefficients).sum(),marks
    loss,marks=forward(scores)
    gradient,=torch.autograd.grad(loss,scores)
    epsilon=1e-5
    for j in range(len(scores)):
        plus=scores.detach().clone();minus=plus.clone()
        plus[j]+=epsilon;minus[j]-=epsilon
        a,ma=forward(plus);b,mb=forward(minus)
        assert torch.equal(ma,marks) and torch.equal(mb,marks)
        assert math.isclose(float(gradient[j]),float((a-b)/(2*epsilon)),rel_tol=1e-7,abs_tol=1e-10)


def test_winner_local_renewal_has_softmax_marks_and_exponential_gaps():
    with torch.random.fork_rng(),torch.no_grad():
        torch.manual_seed(83)
        scores=torch.tensor([.25,.5,1.25],dtype=torch.float64).log()
        all_marks=[];all_gaps=[]
        for _ in range(1024):
            _,times,marks=poisson_clocks(scores,16)
            all_marks.append(marks);all_gaps.append(torch.diff(times,prepend=times.new_zeros(1)))
        marks=torch.cat(all_marks);gaps=torch.cat(all_gaps)
        probabilities=scores.exp()/scores.exp().sum()
        frequency=torch.bincount(marks,minlength=3).to(probabilities)/len(marks)
        torch.testing.assert_close(frequency,probabilities,rtol=0,atol=.013)
        joint=torch.bincount(marks[:-1]*3+marks[1:],minlength=9).reshape(3,3).to(probabilities)/(len(marks)-1)
        torch.testing.assert_close(joint,probabilities[:,None]*probabilities[None,:],rtol=0,atol=.015)
        assert abs(float(gaps.mean())-.5)<.02
        assert abs(float(gaps.var())-.25)<.03


def test_integrated_single_arrival_nests_original_forward_and_credit():
    with torch.random.fork_rng():
        torch.manual_seed(89)
        reference=ParallelHeadRaceLanguageModel(8,2,2,heads=2)
        model=RepeatedArrivalRaceLanguageModel(8,2,2,heads=2,arrivals_per_query=1)
        model.load_state_dict(copy.deepcopy(reference.state_dict()))
        tokens=torch.tensor([1,2,1,3,4,1,2,5])
        torch.manual_seed(97);a,state=reference.forward_chunk(tokens)
        a.square().sum().backward();rng=torch.get_rng_state()
        torch.manual_seed(97);b,other=model.forward_chunk(tokens)
        b.square().sum().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert torch.equal(rng,torch.get_rng_state())
        assert state.selected_updates==other.selected_updates
        for p,q in zip(reference.parameters(),model.parameters()):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_integrated_repeated_teacher_matches_inference_and_keeps_key_budget():
    with torch.random.fork_rng(),torch.no_grad():
        torch.manual_seed(101)
        model=RepeatedArrivalRaceLanguageModel(8,2,2,heads=2,arrivals_per_query=4)
        tokens=torch.tensor([1,2,1,3,4,1,2,5])
        model.eval();torch.manual_seed(103);a,state=model.forward_chunk(tokens)
        model.train();torch.manual_seed(103);b,teacher=model.forward_chunk(tokens)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert state.kv_values==4*state.kv_queries
        assert state.kv_emitter_renewals==3*state.kv_queries
        assert teacher.kv_teacher_reads==teacher.kv_scores
        assert state.kv_scores<=12*state.kv_queries
        assert float(state.context_arrivals.max())<len(tokens)-1+.5


def test_repeated_operator_audit_covers_forward_credit_and_clock_projection():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
    from race_language_screen import capture
    from repeated_arrival_language_helpers import activity,architecture_forward
    with torch.random.fork_rng():
        torch.manual_seed(107)
        model=RepeatedArrivalRaceLanguageModel(8,2,2,heads=2,arrivals_per_query=4)
        state=model.new_state();before=activity(state,model);box={}
        def forward():
            box['output'],box['state']=model.forward_chunk(torch.tensor([1,2,1,3]),state)
        trace=capture(forward)
        backward=capture(lambda:box['output'].square().sum().backward())
        delta={k:v-before[k] for k,v in activity(box['state'],model).items()}
        projected=architecture_forward(trace,delta)
        assert trace['formula_coverage_complete'] and backward['formula_coverage_complete']
        assert projected['arithmetic_flops']>=0 and projected['special_function_evaluations']>=0
        assert trace['exponential_random_draws']==delta['scored_clock_rates']+delta['kv_emitter_renewals']
        assert projected['physical_arrivals']==delta['receiver_updates']+delta['kv_delivered_values']
