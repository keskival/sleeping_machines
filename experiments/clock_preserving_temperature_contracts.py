"""Bounded calibrated-race distribution, nesting and real-native contracts."""
import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
from dvs_native_window_intervention import equal_state
from dvs_key_score_decomposition import replay
from sleeping_machines.clock_preserving_temperature import calibrated_from_raw,ClockPreservingTemperatureHeads


def make(config,weights):
    torch.manual_seed(config.seed)
    model=ClockPreservingTemperatureHeads(sources=1,content_dim=33,classes=11,
        payload=config.payload,depth=config.depth,heads=config.heads,pool=config.pool)
    model.load_state_dict(weights);model.eval();return model


def laws():
    torch.manual_seed(810177);count=131072;s=torch.tensor([-1.2,.4,.9],dtype=torch.float64)
    rates=s.exp();raw=torch.empty(count,3,dtype=torch.float64).exponential_()/rates
    scores=s.expand_as(raw);expected_time=float(1/rates.sum());rows=[]
    for temperature in (1.,2.,4.):
        first,winner,u=calibrated_from_raw(scores,raw,torch.tensor(temperature,dtype=torch.float64))
        assert torch.equal(first,raw.min(-1).values)
        if temperature==1.:assert torch.equal(winner,raw.min(-1).indices)
        probability=(s/temperature).softmax(-1);frequency=torch.bincount(winner,minlength=3)/count
        assert torch.all((frequency-probability).abs()<6*(probability*(1-probability)/count).sqrt())
        conditional=[]
        for j in range(3):
            sample=first[winner==j];mean=float(sample.mean())
            assert abs(mean-expected_time)<6*expected_time/len(sample)**.5
            conditional.append(dict(winner=j,count=len(sample),mean_first_time=mean))
        bins=torch.bincount((u*20).long().clamp_max(19),minlength=20)/count
        assert torch.all((bins-.05).abs()<6*(.05*.95/count)**.5)
        for shift in (-2.,1.5):
            t2,w2,u2=calibrated_from_raw(scores+shift,raw*torch.exp(torch.tensor(-shift,dtype=torch.float64)),
                torch.tensor(temperature,dtype=torch.float64))
            torch.testing.assert_close(u,u2,rtol=1e-12,atol=1e-12);assert torch.equal(winner,w2)
            torch.testing.assert_close(t2,first*torch.exp(torch.tensor(-shift,dtype=torch.float64)),rtol=0,atol=0)
        rows.append(dict(temperature=temperature,probability=probability.tolist(),frequency=frequency.tolist(),
            first_time_mean=float(first.mean()),expected_first_time=expected_time,conditional_first_time=conditional,
            uniform_bins=bins.tolist(),common_shift_winner_preserved=True))
    return rows


def native(config,weights,row):
    reference=N.make_model(config,fast=False);reference.load_state_dict(weights);candidate=make(config,weights)
    z,state,rng=replay(reference,row,411173);zc,sc,rngc=replay(candidate,row,411173)
    assert torch.equal(z,zc) and torch.equal(rng,rngc);equal_state(state,sc)
    for model in (reference,candidate):model.zero_grad(set_to_none=True)
    z,_=N.predict(reference,row,411173,True);zc,_=N.predict(candidate,row,411173,True)
    F.cross_entropy(z[None],torch.tensor([row['target']])).backward()
    F.cross_entropy(zc[None],torch.tensor([row['target']])).backward();assert torch.equal(z,zc)
    checked=0
    for (name,x),(other,y) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name==other and (x.grad is None)==(y.grad is None)
        if x.grad is not None:assert torch.equal(x.grad,y.grad),name;checked+=1
    for temperature,scope in ((2.,'all'),(4.,'all'),(2.,'layer0')):
        candidate.route_temperature=temperature;candidate.temperature_scope=scope
        zc,sc,rngc=replay(candidate,row,411173);assert torch.equal(rng,rngc)
        assert sc.events==21 and sc.candidate_scores==168 and sc.selected_updates==84
        assert sc.counterfactual_values==0 and float(sc.contexts[0][1].max())>=1.
        swapped=dict(row,target=(row['target']+1)%11)
        zs,ss,rngs=replay(candidate,swapped,411173);assert torch.equal(zc,zs) and torch.equal(rngc,rngs);equal_state(sc,ss)
        try:N.predict(candidate,row,411173,True)
        except ValueError as e:assert 'training not installed' in str(e)
        else:raise AssertionError('Positive-temperature training must be refused')
    return dict(seed=config.seed,tau1_exact_logit_state_rng=True,exact_parameter_gradients=checked,
        three_positive_settings_exact_rng_target_invariance_sparse_activity_training_refusal=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    torch.set_num_threads(1);begin=time.perf_counter();distribution=laws();checks=[];parents=[]
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        config=argparse.Namespace(**parent['args']);expanded=argparse.Namespace(**vars(config));expanded.fit=984
        fit,_,data=N.load(expanded);assert data==parent['data']
        cp=(ROOT/name).with_suffix('.progress.pt');saved=torch.load(cp,weights_only=False)
        assert saved['cursor']['epoch']==5 and saved['source_sha256']==parent['source_sha256']
        for f,digest in parent['source_sha256'].items():assert N.sha(ROOT/f)==digest
        checks.append(native(config,saved['online_model'],fit[257]))
        parents.append(dict(result=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(cp)))
    own=['experiments/clock_preserving_temperature_contracts.py','sleeping_machines/clock_preserving_temperature.py',
        'experiments/theory/101_clock_preserving_route_calibration.md','experiments/dvs_key_score_decomposition.py',
        'experiments/dvs_native_window_intervention.py']
    result=dict(status='completed',args=vars(a),distribution_contracts=distribution,native_contracts=checks,parents=parents,
        source_sha256={f:N.sha(ROOT/f) for f in own},wall_s=time.perf_counter()-begin,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hardware=dict(host=platform.node(),threads=1),
        optimizer_steps=0,development_evaluations=0,whole_audit_flops=None,
        scope='Finite-draw six-standard-error distribution screens supplement exact exponential conditioning proof; '
              'actual native tau1 forward/state/all-gradient/RNG nesting and positive-temperature sparse writes, '
              'target invariance and training refusal. Two frozen producers. No quality or learning advantage.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
