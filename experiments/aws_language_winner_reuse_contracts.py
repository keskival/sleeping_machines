"""Winner reuse versus frozen all-lane replay: gradients, factual state/RNG and measured work."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import causal_language_replay_contracts as C
import causal_language_replay_helpers_rng as R
import aws_language_winner_reuse as W
from race_language_screen import capture


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    begin=time.perf_counter();out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists();cases=[]
    for family in ('private','depth'):
        for length in (1,3):
            m=C.make(family);st=C.entering(m);n=copy.deepcopy(m);tokens=torch.tensor([1,2,3,4][:length]);targets=(tokens+3)%27;seed=47290+length
            q,z,s,act=R.batched_objective(m,tokens,targets,st,seed);q.backward()
            v,y,t,other=W.objective(n,tokens,targets,st,seed);v.backward()
            C.close(z,y,True);C.state_close(s,t,True);C.close(act['factual_end_rng'],other['factual_end_rng'],True)
            maximum=0.
            for (name,par),(nn,np) in zip(m.named_parameters(),n.named_parameters()):
                assert name==nn and (par.grad is None)==(np.grad is None)
                if par.grad is not None:C.close(par.grad,np.grad);maximum=max(maximum,float((par.grad-np.grad).abs().max()))
            assert other['shadow_lanes']*2==act['shadow_lanes'] and other['shadow_events']*2==act['shadow_events']
            cases.append(dict(family=family,targets=length,every_parameter_gradient_equivalent=True,max_abs_gradient_error=maximum,factual_logits_state_rng_bitwise_equal=True,reference_lanes=act['shadow_lanes'],new_lanes=other['shadow_lanes'],reference_events=act['shadow_events'],new_events=other['shadow_events']))
    audits={}
    for label,fn in [('full',R.batched_objective),('reuse',W.objective)]:
        m=C.make('private');st=C.entering(m);tokens=torch.tensor([1,2,3]);box={}
        def forward():box['output']=fn(m,tokens,(tokens+3)%27,st,52791)
        forward_trace=capture(forward);backward_trace=capture(lambda:box['output'][0].backward());assert forward_trace['formula_coverage_complete'] and backward_trace['formula_coverage_complete']
        audits[label]=dict(forward_and_shadows=forward_trace,backward=backward_trace,whole_arithmetic_and_unit_special_flops=sum(t['arithmetic_flops']+t['special_function_evaluations'] for t in (forward_trace,backward_trace)))
    files=['experiments/aws_language_winner_reuse.py','experiments/aws_language_winner_reuse_contracts.py','experiments/causal_language_replay_helpers_rng.py','sleeping_machines/causal_language_shadow_rng.py']
    result=dict(status='completed',args=vars(a),contracts_passed=True,cases=cases,work_audits=audits,measured_work_ratio=audits['reuse']['whole_arithmetic_and_unit_special_flops']/audits['full']['whole_arithmetic_and_unit_special_flops'],source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},wall_s=time.perf_counter()-begin,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exact losing-route-only implementation diagnostic; shared/private double all-param equivalence, factual state/RNG unchanged. Work includes both forward and backward; no optimizer or quality/energy advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
