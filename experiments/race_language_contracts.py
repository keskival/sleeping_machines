"""Guarded contracts for indexed winner retrieval and its counterfactual teacher."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sleeping_machines.race_language import RaceLanguageModel, WinnerRace

SOURCES = ['sleeping_machines/race_language.py','sleeping_machines/selective_stream_language.py',
    'sleeping_machines/parallel_stream_language.py','sleeping_machines/stream_language.py',
    'sleeping_machines/event_state.py','sleeping_machines/event_memory.py',
    'sleeping_machines/operation_audit.py','sleeping_machines/language_memory.py',
    'experiments/parallel_event_language.py','experiments/e120_shared_tasks.py',
    'experiments/race_language_contracts.py','experiments/race_language_screen.py']


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args = parser.parse_args()
    out = ROOT/'experiments/results/parallel_language'/f'{args.tag}.json'
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique plain tag required')
    torch.set_num_threads(1);torch.manual_seed(6);start=time.perf_counter()
    # Test both mechanisms before quality fitting. Nonzero controls ensure the
    # equality is not merely an identity-initialization accident.
    checks = []
    tokens = torch.tensor([1,2,1,3,1,2,1,4,2,1,3,1,2,4,1,2,1,3,1,2])
    for arm in ('race','softmax'):
        model = RaceLanguageModel(width=32,modes=16,attention=arm,capacity=3,races=4)
        for layer in model.core.layers:
            with torch.no_grad():layer.memory_control.weight.normal_(std=.01)
        model.eval()
        def run(parts,origin=0,sequence=tokens):
            torch.manual_seed(901)
            state=model.new_state();state.core.position=origin;outputs=[]
            for part in parts:
                z,state=model.forward_chunk(sequence[part[0]:part[1]],state);outputs.append(z)
            return torch.cat(outputs),state
        whole,state=run([(0,len(tokens))])
        split,other=run([(0,7),(7,12),(12,len(tokens))])
        torch.testing.assert_close(whole,split,atol=4e-5,rtol=4e-4)
        shifted,_=run([(0,len(tokens))],10_000_000)
        torch.testing.assert_close(whole,shifted,atol=4e-5,rtol=4e-4)
        perturbed=tokens.clone();perturbed[12:]=(perturbed[12:]+4)%27
        future,_=run([(0,len(tokens))],sequence=perturbed)
        torch.testing.assert_close(whole[:12],future[:12],atol=1e-6,rtol=1e-5)
        assert max(map(len,state.buckets.values()))<=3
        assert state.scored_keys<=state.retrieval_queries*3
        if arm=='race':assert state.delivered_values==state.retrieval_queries*4
        # Values always belong to observed successors, never the target token
        # after the current query; the bank stores preceding positions only.
        assert all(t<state.core.position for rows in state.buckets.values() for _,_,t in rows)
        model.train();model.zero_grad(set_to_none=True);torch.manual_seed(3)
        prediction,train_state=model.forward_chunk(tokens)
        F.cross_entropy(prediction[:-1],tokens[1:]).backward()
        required=['query.weight','key.weight','value.weight','readout.weight']
        norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if n in required}
        assert all(math.isfinite(v) and v>0 for v in norms.values()),norms
        assert train_state.teaching_value_visits==train_state.scored_keys
        checks.append(dict(arm=arm,partition_error=float((whole-split).detach().abs().max()),
            origin_error=float((whole-shifted).detach().abs().max()),gradient_norms=norms,
            causality='passed',bounded_index='passed',winner_delivery='passed'))
    # Closed-form expectation claims tested independently of quality or loss.
    # Do not claim the nonlinear sampled-loss gradient equals softmax training.
    torch.manual_seed(46)
    scores=torch.tensor([-.7,.3,1.1,.1],dtype=torch.float64)
    values=torch.tensor([[1.,2.],[3.,-1.],[-2.,.5],[.2,.7]],dtype=torch.float64)
    rates=scores.exp();probability=rates/rates.sum();mean=probability@values
    noise=torch.empty((100000,4),dtype=torch.float64).exponential_()
    times,winners=(noise/rates).min(dim=1)
    frequencies=torch.bincount(winners,minlength=4)/len(winners)
    centered=values-values.mean(0)
    teacher=rates[None,:,None]*times[:,None,None]*centered[None]
    race_sum=times[:,None]*(rates@centered)[None]
    teacher[torch.arange(len(times)),winners]-=race_sum
    expected=probability[:,None]*(values-mean)
    assert float((frequencies-probability).abs().max())<.006
    assert float((teacher.mean(0)-expected).abs().max())<.012
    assert float(teacher.sum(1).abs().max())<1e-10
    # Shared clock covariance counterexample to the historical sum-of-squares
    # bound: equal values still have relative variance one, however many keys.
    equal_values=torch.ones(4,dtype=torch.float64)
    clock_output=times*(rates@equal_values)
    assert abs(float(clock_output.var())-1)<.025
    result=dict(status='completed',args=vars(args),checks=checks,
        winner_probability_error=float((frequencies-probability).abs().max()),
        teacher_expected_jacobian_error=float((teacher.mean(0)-expected).abs().max()),
        teacher_credit_conservation_error=float(teacher.sum(1).abs().max()),
        historical_shared_clock_variance=float(clock_output.var()),
        source_sha256=hashes(),wall_s=time.perf_counter()-start,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(device='cpu',threads=1,platform=platform.platform()),
        scope='Numerical contracts only. Fixed-cotangent expected output Jacobian, not unbiased sampled-loss gradient or quality evidence.')
    out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
