"""Finite witnesses for conditional time, replay choice bias and critic sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_local_expectation_benchmark as LE
from sleeping_machines.addressed_event_heads import AddressedEventHeads


class SingleRace:
    """Actual replay wrapper, with one time-recording event and constant readout."""
    training=False
    credit='pathwise'

    def __init__(self):
        self.scores=torch.tensor([1.,3.],dtype=torch.float64).log()
        self.values=torch.tensor([[0.],[1.]],dtype=torch.float64)

    def new_state(self):
        return SimpleNamespace(delay=None,winner=None)

    def consume_event(self,source,timestamp,content,state):
        _,state.delay,state.winner=self.race(self.scores,self.values)
        return torch.zeros(2,dtype=torch.float64),None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    a=p.parse_args();out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    started=time.perf_counter();torch.set_num_threads(1)
    scores=torch.tensor([1.,3.],dtype=torch.float64).log().requires_grad_()
    rates=scores.exp();total=rates.sum();pi=rates/total
    true=torch.autograd.grad(1/total,scores,retain_graph=True)[0]
    # Conditional W does not alter the first-time Exp(total) distribution.
    forced_means=(1/rates).detach()
    wrong_choice=torch.autograd.grad((pi*forced_means).sum(),scores,retain_graph=True)[0]
    interior_mean=-pi.detach()/total.detach()
    combined=interior_mean+wrong_choice
    torch.testing.assert_close(true,torch.tensor([-.0625,-.1875],dtype=torch.float64),rtol=0,atol=1e-12)
    torch.testing.assert_close(wrong_choice,torch.tensor([.125,-.125],dtype=torch.float64),rtol=0,atol=1e-12)
    torch.testing.assert_close(combined,torch.tensor([.0625,-.3125],dtype=torch.float64),rtol=0,atol=1e-12)
    # Integrate the true joint score against E[T] and E[T^2] exactly.
    joint_mean=pi.detach()/total.detach()-rates.detach()*2/total.detach().square()
    torch.testing.assert_close(joint_mean,true,rtol=0,atol=1e-12)
    # Audit the actual implementation's factual/forced time with identical RNG.
    original=LE.PATHWISE;LE.PATHWISE=AddressedEventHeads.race
    records=[]
    try:
        for seed in (7,11,23,91):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                individual=torch.empty(2,dtype=torch.float64).exponential_()/rates.detach()
            first,winner=individual.min(0);model=SingleRace()
            row=dict(events=[(0.,None)],target=0,index=0)
            _,_,factual=LE.run(model,row,seed)
            expected_delay=.001+.010*first/(1+first)
            torch.testing.assert_close(factual.delay,expected_delay,rtol=0,atol=0)
            forced=[]
            for i in range(2):
                _,_,state=LE.run(model,row,seed,force=(0,i))
                expected=.001+.010*individual[i]/(1+individual[i])
                torch.testing.assert_close(state.delay,expected,rtol=0,atol=0)
                assert int(state.winner)==i
                forced.append(float(state.delay))
            assert forced[int(winner)]==float(factual.delay)
            assert forced[1-int(winner)]>float(factual.delay)
            records.append(dict(seed=seed,actual_winner=int(winner),factual_delay=float(factual.delay),forced_delays=forced))
    finally:
        LE.PATHWISE=original
    # Horvitz-Thompson critic residual: fixed critic is unbiased under subset sampling.
    fixed=torch.tensor([2.,-3.],dtype=torch.float64);truth=torch.tensor([.4,.7],dtype=torch.float64)
    estimates=[fixed.sum()+2*(truth[i]-fixed[i]) for i in range(2)]
    torch.testing.assert_close(torch.stack(estimates).mean(),truth.sum(),atol=1e-12,rtol=0)
    # A critic fitted to indicate the same sampled site defeats that cancellation.
    adaptive=[]
    for i in range(2):
        critic=torch.zeros(2,dtype=torch.float64);critic[i]=1
        adaptive.append(float(critic.sum()-2*critic[i]))
    assert adaptive==[-1.,-1.]
    own=['experiments/race_replay_law_contracts.py','experiments/theory/92_conditional_replay_clocks_and_critic_sampling.md']
    observed=['experiments/dvs_local_expectation_benchmark.py','sleeping_machines/addressed_event_heads.py','tests/test_dvs_local_expectation.py']
    result=dict(status='completed',args=vars(a),contracts_passed=4,
        time_only_loss=dict(rates=[1.,3.],true_expected_gradient=true.tolist(),
            forced_individual_time_choice_gradient=wrong_choice.tolist(),
            retained_interior_expected_gradient=interior_mean.tolist(),incorrect_combined_gradient=combined.tolist(),
            correct_joint_score_expected_gradient=joint_mean.tolist(),first_score_direction_reversed=True),
        actual_replay_time_records=records,
        critic_sampling=dict(fixed_critic_expectation=float(torch.stack(estimates).mean()),true_sum=float(truth.sum()),
            sample_adaptive_critic_zero_target_expectation=-1.,fixed_before_sampling_required=True),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in own},
        audited_source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in observed},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Analytic finite expectation and actual one-race wrapper time identity. Audited external sources are version observations, not frozen campaign dependencies. No integrated model fit or whole-gradient exactness claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
