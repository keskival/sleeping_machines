"""Exercise actual depth8 language drivers: traced/untraced update, pause/resume and replay ledger."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_depth8_language_credit as B


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif hasattr(a,'__dict__'):equal(vars(a),vars(b))
    else:assert a==b,(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);start=time.perf_counter();out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists()
    directory=ROOT/'experiments/results/aws_depth8_language_credit';cases=[]
    for family in ('private','depth'):
        for mode in ('teacher','factorized','replay'):
            full=f'{a.tag}_{family}_{mode}_full';partial=f'{a.tag}_{family}_{mode}_recover'
            common=['--receiver-sharing',family,'--credit-mode',mode,'--fit','49','--dev','33','--epochs','1','--chunk','16','--update-targets','16','--warmup-targets','32','--payload','4','--depth','8','--heads','2','--pool','2','--initial-dev','33','--seed','7']
            def run(tag,extra=()):
                sys.argv=['aws_depth8_language_credit.py','--tag',tag,*common,*extra];B.main()
                return torch.load(directory/f'{tag}.progress.pt',weights_only=False)
            baseline=run(full);stopped=run(partial,('--stop-after-updates','1'));assert stopped['cursor']['next_target']==16
            resumed=run(partial,('--resume',))
            for key in ('model','optimizer','cursor','total_targets','updates','learner_counters','torch_rng','stream_state'):equal(baseline[key],resumed[key])
            for key in ('work','final','curve'):equal(baseline['result'][key],resumed['result'][key])
            counters=baseline['learner_counters'];assert counters['total_targets']==48 and counters['credit_chunks']==3
            assert counters['shadow_lanes']==(1536 if mode=='replay' else 0)
            assert counters['shadow_events']==(24576 if mode=='replay' else 0)
            for trace in baseline['result']['work']['traces'].values():
                for name in ('forward_and_loss','backward','gradient_normalization','gradient_clipping','optimizer'):assert trace[name]['formula_coverage_complete']
            cases.append(dict(family=family,mode=mode,exact_actual_driver_recovery=True,traced_and_untraced_same_objective=True,counters=counters,work_complete=True))
    result=dict(status='completed',args=vars(a),contracts_passed=True,cases=cases,source_sha256={**B.sources(),'experiments/aws_language_credit_driver_contracts_retry.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Actual production driver, three16-target chunks including untraced third chunk; six modes exact Adam/cursor/RNG/work recovery. Small-data numerical admission only, not language advantage.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
