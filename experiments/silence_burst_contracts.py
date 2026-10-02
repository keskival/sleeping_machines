"""Causality, two-layer gradients and finite timeout boundary-credit contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sleeping_machines.silence_burst import silence_bursts,finite_timeout_risk


def scalar(x):return torch.tensor(x,dtype=torch.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(1)
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    t=scalar([0.,.1,.4,.5,1.]);v=scalar([[1.],[2.],[4.],[8.],[16.]])
    ts,ys,state=silence_bursts(t,v,scalar(.2),scalar(.8),scalar(0.))
    torch.testing.assert_close(ts,scalar([.3,.7]));torch.testing.assert_close(ys,scalar([[3.],[12.]]))
    assert not state['active'] and state['accepted_events']==4
    # Exact deadline tie extends, later input cannot revise a past prediction.
    ts,ys,state=silence_bursts(scalar([0.,.2]),scalar([[1.],[2.]]),scalar(.2),scalar(.4),scalar(0.))
    torch.testing.assert_close(ts,scalar([.4]));torch.testing.assert_close(ys,scalar([[3.]]))
    assert not state['active']
    early=silence_bursts(t,v,scalar(.2),scalar(.25),scalar(.3))
    changed=v.clone();changed[2:]*=1000
    future=silence_bursts(t,changed,scalar(.2),scalar(.25),scalar(.3))
    assert len(early[0])==0 and early[2]['active']
    torch.testing.assert_close(early[2]['pending'],future[2]['pending'],rtol=0,atol=0)
    ended=silence_bursts(t[:2],v[:2],scalar(.2),scalar(.25),scalar(.3))
    torch.testing.assert_close(early[2]['pending'],ended[2]['pending'],rtol=0,atol=0)
    # Exact all-input finite differences through completed two-layer deadlines.
    times=scalar([0.,.08,.48,.55,1.1]).requires_grad_()
    values=scalar([[.2,.7],[.8,-.1],[-.4,.3],[.1,.9],[.6,-.5]]).requires_grad_()
    h1=scalar(.16).requires_grad_();h2=scalar(.20).requires_grad_()
    decay=scalar(.4).requires_grad_()
    def compose(times,values,h1,h2,decay):
        first_t,first_y,_=silence_bursts(times,values,h1,scalar(2.),decay)
        second_t,second_y,_=silence_bursts(first_t,first_y,h2,scalar(2.5),decay)
        return torch.cat((second_t,second_y.flatten()))
    assert torch.autograd.gradcheck(compose,(times,values,h1,h2,decay),eps=1e-6,atol=2e-6,rtol=2e-5)
    # Hard option outcomes retain times and values through a real two-layer
    # stream. Compare nested exact risk to explicit joint enumeration.
    options=scalar([.06,.22,.70]);s1=scalar([.2,-.4,.5]).requires_grad_()
    s2=scalar([-.1,.3,.4]).requires_grad_();weights=scalar([.8,-.3]).requires_grad_()
    def outcome(h1,h2):
        ft,fy,_=silence_bursts(times,values,h1,scalar(2.),decay)
        st,sy,_=silence_bursts(ft,fy,h2,scalar(2.5),decay)
        feature=(sy.sum(0)*weights).sum()+.1*st.sum()
        return torch.nn.functional.softplus(-feature)
    def first_loss(h1):return finite_timeout_risk(s2,options,lambda h2:outcome(h1,h2))[0]
    nested,first_losses=finite_timeout_risk(s1,options,first_loss)
    explicit=sum(s1.softmax(-1)[i]*s2.softmax(-1)[j]*outcome(h1,h2)
        for i,h1 in enumerate(options) for j,h2 in enumerate(options))
    torch.testing.assert_close(nested,explicit,rtol=1e-12,atol=1e-12)
    parameters=(s1,s2,values,times,decay,weights)
    ng=torch.autograd.grad(nested,parameters,retain_graph=True)
    eg=torch.autograd.grad(explicit,parameters,retain_graph=True)
    for x,y in zip(ng,eg):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(ng[0],s1.softmax(-1)*(first_losses-nested),rtol=1e-12,atol=1e-12)
    files=['sleeping_machines/silence_burst.py','experiments/silence_burst_contracts.py',
        'experiments/theory/79_learned_windows_and_silence_bursts.md']
    result=dict(status='completed',args=vars(a),contracts_passed=4,
        scheduled_causality_ties_and_no_eof_flush_passed=True,
        two_layer_all_parameter_and_input_time_finite_differences_passed=True,
        exact_two_layer_timeout_option_risk_and_all_derivatives_passed=True,
        timeout_options=options.tolist(),actual_option_outcome_losses=first_losses.detach().tolist(),
        exact_first_policy_gradient=ng[0].tolist(),joint_outcomes_evaluated=9,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Event-only additive/decaying primitive and two-layer composition; exact history-cell gradients and finite-option unrealized boundary credit. No fitted integrated native model, natural-stream advantage, or free continuous merge/split gradient claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
