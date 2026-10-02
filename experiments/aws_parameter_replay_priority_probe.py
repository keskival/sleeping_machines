"""Fresh FIT confirmation of three distinct learned-priority replay sites."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import dvs_critic_le_benchmark as CR
import dvs_local_expectation_benchmark as LE
import dvs_native_benchmark as N
from sleeping_machines.shadow_lanes import shadow_losses
from sleeping_machines.replay_priority_sampling import inclusion,orders,parameter_variance


def contract():
    p=np.array([.1,.3,.6]);v=np.array([[1.,2.],[-3.,4.],[5.,-6.]])
    for k in (1,2,3):
        marginal,pair=inclusion(p,k);draws=orders(3,k);x=p[draws];used=np.cumsum(x,1)-x;weights=np.prod(x/(1-used),1)
        estimates=(v[draws]/marginal[draws,None]).sum(1);target=v.sum(0)
        np.testing.assert_allclose((weights[:,None]*estimates).sum(0),target,rtol=0,atol=1e-13)
        mse=float((weights*((estimates-target)**2).sum(1)).sum());assert abs(mse-parameter_variance(v,marginal,pair))<1e-11
    m,pair=inclusion(np.full(20,.05),3)
    np.testing.assert_allclose(m,.15,rtol=0,atol=1e-13)
    np.testing.assert_allclose(pair-np.diag(m), (np.ones((20,20))-np.eye(20))*6/380,rtol=0,atol=1e-13)
    return 'exact inclusion, uniform nesting, shared-vector unbiased mean and variance pass'


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);started=time.perf_counter();checks=contract()
    oldpath=ROOT/'experiments/results/diagnostics/aws_learned_replay_priority_20261002T232400Z.json';old=json.loads(oldpath.read_text())
    for n,h in old['source_sha256'].items():assert N.sha(ROOT/n)==h,n
    modelpath=oldpath.with_suffix('.models.joblib');assert N.sha(modelpath)==old['artifacts'][str(modelpath.relative_to(ROOT))]['sha256'];models=joblib.load(modelpath)
    args=CR.parser().parse_args(['--tag',a.tag,'--data','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_data.json',
        '--controls','experiments/results/dvs_calibration/local_dvs_calibration_20261002T141400Z_controls.json','--fit','128','--dev','1'])
    args.bins=4;args.clock_step=.25;rows,_,data=A.load(args);results=[];parents={}
    for seed in (7,8):
        args.seed=seed;model=LE.make_model(args,fast=True);checkpoint=ROOT/f'experiments/results/dvs_native/aws_full_coarse_20261002T213100Z_coarse_matchedclock_s{seed}.progress.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=False,map_location='cpu')['best_state']);model.train();parents[str(seed)]=N.sha(checkpoint)
        bankpath=ROOT/'experiments/results/diagnostics/aws_signed_replay_variance_20261002T230400Z.pt'
        bankinfo=json.loads(bankpath.with_suffix('.json').read_text());assert N.sha(bankpath)==bankinfo['artifact_sha256']
        bank=torch.load(bankpath,weights_only=False)[str(seed)]['examples']
        parameter_targets=[];train_features=[];parameters=tuple(model.parameters())
        for index in range(32):
            _,R,_,scores,_,_=CR.run_with_values(model,rows[index],100000+seed,False)
            feats,pi,utility=bank[index]
            torch.testing.assert_close(torch.stack([x.detach().double().softmax(0) for x in scores]),pi,rtol=0,atol=0)
            g=pi*(utility-(pi*utility).sum(-1,keepdim=True));norms=[]
            for r in range(R):
                grads=torch.autograd.grad(scores[r],parameters,grad_outputs=g[r].to(scores[r].dtype),retain_graph=True,allow_unused=True)
                norms.append(float(torch.sqrt(sum(x.detach().double().square().sum() for x in grads if x is not None))))
            parameter_targets.extend(norms);train_features.extend(feats.reshape(R,-1).numpy())
        parameter_targets=np.array(parameter_targets);floor=max(float(np.median(parameter_targets))*.001,1e-12)
        predictor=ExtraTreesRegressor(n_estimators=64,max_depth=6,min_samples_leaf=8,random_state=681,n_jobs=1)
        predictor.fit(np.array(train_features),np.log(parameter_targets+floor));models[str(seed)]=dict(predictor=predictor,floor=floor)
        print(f'seed{seed}:32 parameter-target training prefixes complete',flush=True)
        baseline=weighted=uniform=0.;cases=[];parameters=tuple(model.parameters());parameter_cases=[]
        for index in range(96,128):
            row=rows[index];_,races,_,scores,values,_=CR.run_with_values(model,row,100000+seed,False)
            with torch.no_grad():
                label=torch.nn.functional.one_hot(torch.tensor(row['target']),11).float().expand(model.pool,-1)
                feats=torch.stack([torch.cat([CR.features(scores[r],values[r],r,races,(r//model.heads)%model.depth,model.depth),values[r].detach(),values[r].detach()-values[r].detach().mean(0),label],-1) for r in range(races)])
                losses=shadow_losses(model,row,100000+seed,[(r,i) for r in range(races) for i in range(model.pool)]).reshape(races,model.pool)
                pi=torch.stack([s.detach().double().softmax(0) for s in scores]);g=pi*(losses-(pi*losses).sum(-1,keepdim=True));e=g.square().sum(-1).numpy()
                saved=models[str(seed)];prediction=np.maximum(np.exp(saved['predictor'].predict(feats.reshape(races,-1).numpy()))-saved['floor'],saved['floor']);proposal=.9*prediction/prediction.sum()+.1/races
                before=time.perf_counter();marginal,pair=inclusion(proposal,3);inclusion_wall=time.perf_counter()-before
                w=float(((1/marginal-1)*e).sum());b=float((races/4-1)*e.sum());u=float((races/3-1)*e.sum());weighted+=w;baseline+=b;uniform+=u
                cases.append(dict(fit_index=index,weighted_k3_mse=w,uniform_k4_mse=b,uniform_k3_mse=u,proposal=proposal.tolist(),inclusion=marginal.tolist(),inclusion_wall_s=inclusion_wall))
            if index in (96,97):
                vectors=[]
                for r in range(races):
                    grads=torch.autograd.grad(scores[r],parameters,grad_outputs=g[r].to(scores[r].dtype),retain_graph=True,allow_unused=True)
                    vectors.append(torch.cat([(torch.zeros_like(p) if x is None else x).detach().double().flatten() for p,x in zip(parameters,grads)]).numpy())
                vectors=np.array(vectors);R=len(vectors);centered=vectors-vectors.mean(0);base=float(R*(R-4)/(4*(R-1))*(centered**2).sum());var=parameter_variance(vectors,marginal,pair)
                parameter_cases.append(dict(fit_index=index,weighted_k3_parameter_variance=var,uniform_k4_parameter_variance=base,ratio=var/base))
        ratio=weighted/baseline;results.append(dict(seed=seed,heldout_score_variance_ratio=ratio,uniform_k3_score_variance_ratio=uniform/baseline,
            score_gate_passed=ratio<=1,parameter_cases=parameter_cases,parameter_gate_passed=all(c['ratio']<=1 for c in parameter_cases),cases=cases,
            proposed_replay_lanes_per_episode=6,baseline_replay_lanes_per_episode=8,diagnostic_actual_replay_lanes=32*20*2))
    out=ROOT/'experiments/results/diagnostics'/f'{a.tag}.json';assert not out.exists()
    files=['experiments/aws_parameter_replay_priority_probe.py','sleeping_machines/replay_priority_sampling.py']
    result=dict(status='completed',tag=a.tag,contract=checks,results=results,gate_passed=all(r['score_gate_passed'] and r['parameter_gate_passed'] for r in results),
        data=data,predictor_result_sha256=N.sha(oldpath),checkpoint_sha256=parents,source_sha256={**old['source_sha256'],**{f:N.sha(ROOT/f) for f in files}},
        scope='New critic-heldout FIT96..127, trained producer labels reused; exact conditional sampling variance only, no quality. Diagnostic enumerates all replay utilities; six-lane proposed cost is not diagnostic cost.',
        fitting_flops=None,wall_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    modelout=out.with_suffix('.models.joblib');joblib.dump(models,modelout);result['priority_artifact']=dict(path=str(modelout.relative_to(ROOT)),sha256=N.sha(modelout));result['additional_training_vjps']=1280
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='cases'} for r in results],indent=2))


if __name__=='__main__':main()
