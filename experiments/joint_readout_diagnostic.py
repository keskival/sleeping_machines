"""Frozen temporal encoders: retention versus affine/quadratic query decoding."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import balanced_joint_benchmark as B
from balanced_joint_protocol import joint_examples,data_hash
from sleeping_machines.polynomial_query_readout import PolynomialQueryReadout,query_features,integrated_query_logit


class ReadoutAudit(B.JointAudit):
    def formula(self,func,args,kwargs,out):
        if str(func)=='aten.std.correction':
            return 4*args[0].numel(),out.numel(),0,'Population std conventional mean/center/square/reduce/sqrt formula'
        return super().formula(func,args,kwargs,out)


def capture(action):
    with ReadoutAudit() as audit:action()
    result=audit.result()
    if not result['formula_coverage_complete']:raise ValueError(result['unsupported_floating_operators'])
    return result


def sources():
    names=['experiments/joint_readout_diagnostic.py','sleeping_machines/polynomial_query_readout.py',
        'experiments/theory/67_retention_and_interaction_readout.md']
    return {**B.sources(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def fingerprint(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def extract(encoder,rows,seeds):
    features=[];base=[];labels=[];bits=[];traces=[];events=keys=updates=0;max_bytes=0;started=time.perf_counter()
    for view,seed in enumerate(seeds):
        for index,row in enumerate(rows):
            box={}
            def action():box['value']=query_features(encoder,row,seed)
            if view==0 and index<4:traces.append(capture(action))
            else:action()
            h,z,state=box['value'];features.append(h.to(torch.float64));base.append(z.to(torch.float64));labels.append(row['target']);bits.append(row['bits'])
            events+=state.events;keys+=state.candidate_scores;updates+=state.selected_updates
            max_bytes=max(max_bytes,state.storage()['persistent_tensor_bytes'])
    flops=sum(x['arithmetic_flops']+x['special_function_evaluations'] for x in traces)/len(traces)
    return dict(features=torch.stack(features),base=torch.stack(base),labels=torch.tensor(labels,dtype=torch.float64),
        bits=torch.tensor(bits,dtype=torch.float64),targets=len(labels),input_events=events,key_scores=keys,selected_updates=updates,
        max_state_tensor_bytes=max_bytes,prefix_unit_special_flops_per_target_estimate=flops,
        traces=traces,wall_s=time.perf_counter()-started)


def fit_readout(features,base,labels,degree,regularization=1e-5):
    started=time.perf_counter();head=PolynomialQueryReadout(features.shape[1],degree);box={}
    def prepare():
        head.fit_statistics(features);box['design']=head.design(features).detach()
    preparation=capture(prepare);design=box['design'];calls=0
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=100,tolerance_grad=1e-8,
        tolerance_change=1e-10,line_search_fn='strong_wolfe')
    sign=2*labels-1
    def closure():
        nonlocal calls
        calls+=1;optimizer.zero_grad(set_to_none=True)
        logit=base+design@head.coefficients
        loss=F.softplus(-sign*logit).mean()+.5*regularization*head.coefficients[1:].square().sum()
        loss.backward();return loss
    def optimize():optimizer.step(closure);closure()
    optimization=capture(optimize)
    with torch.no_grad():
        logits=head(features,base);loss=float(F.softplus(-sign*logits).mean())/math.log(2)
    state=optimizer.state[head.coefficients]
    return head,dict(degree=degree,regularization=regularization,closure_evaluations=calls,
        optimizer_iterations=state['n_iter'],fitting_bits=loss,regularized_gradient_norm=float(head.coefficients.grad.norm()),
        fitting_targets=len(labels),optimizer_target_evaluations=calls*len(labels),
        coefficient_norm=float(head.coefficients.detach().norm()),parameters=head.coefficients.numel(),
        preparation_trace=preparation,optimization_trace=optimization,wall_s=time.perf_counter()-started,
        unit_special_flops=sum(t['arithmetic_flops']+t['special_function_evaluations'] for t in [preparation,optimization]),
        scope='Fixed-feature convex residual readout; every closure/optimizer operation counted. Fitting derivative reported, not assumed converged. No encoder update.')


@torch.no_grad()
def head_score(head,data,base=None,labels=None):
    z=head(data['features'],data['base'] if base is None else base)
    y=data['labels'] if labels is None else labels
    loss=F.softplus(-(2*y-1)*z)/math.log(2)
    return dict(query_bits=float(loss.mean()),accuracy=float(((z>0)==y.bool()).double().mean()),
        per_target_bits=loss.tolist(),class1_probabilities=z.sigmoid().tolist())


def walsh(features):
    x=features.reshape(-1,4,features.shape[-1]);out={}
    for name,signs in [('first_bit',[1,1,-1,-1]),('second_bit',[1,-1,1,-1]),('interaction',[1,-1,-1,1])]:
        w=(x*torch.tensor(signs,dtype=x.dtype)[None,:,None]).mean(1)
        out[name]=dict(mean_coefficient_norm=float(w.norm(dim=-1).mean()),
            mean_direction_norm=float(w.mean(0).norm()),max_coefficient_norm=float(w.norm(dim=-1).max()))
    return out


def numerical_contracts():
    torch.manual_seed(19);x=torch.randn(4,3,dtype=torch.float64,requires_grad=True);base=torch.randn(4,dtype=torch.float64)
    q=PolynomialQueryReadout(3,2);q.fit_statistics(x.detach())
    torch.testing.assert_close(q(x,base),base,rtol=0,atol=0)
    with torch.no_grad():q.coefficients.copy_(torch.randn_like(q.coefficients)*.1)
    analytical=torch.autograd.grad(q(x,base).sum(),x)[0];numeric=torch.zeros_like(x);eps=1e-6
    for i in range(4):
        for j in range(3):
            hi=x.detach().clone();lo=x.detach().clone();hi[i,j]+=eps;lo[i,j]-=eps
            numeric[i,j]=(q(hi,base).sum()-q(lo,base).sum())/(2*eps)
    torch.testing.assert_close(analytical,numeric,rtol=1e-6,atol=1e-8)
    labels=torch.tensor([0.,1.,1.,0.],dtype=torch.float64);z=q(x,base)
    loss=F.softplus(-(2*labels-1)*z).mean()+.5e-5*q.coefficients[1:].square().sum()
    gradient=torch.autograd.grad(loss,q.coefficients)[0]
    expected=q.design(x).detach().T@(z.detach().sigmoid()-labels)/len(labels)
    expected[1:]+=1e-5*q.coefficients.detach()[1:]
    torch.testing.assert_close(gradient,expected,rtol=1e-12,atol=1e-12)
    a=B.parser().parse_args(['--tag','readout-contract','--payload','2','--depth','2'])
    encoder=B.make_model(a);before=fingerprint(encoder);rows=joint_examples(groups=1)
    data=extract(encoder,rows,[314159]);head,fit=fit_readout(data['features'],data['base'],data['labels'],2)
    original=head_score(head,data)['class1_probabilities'];live=[]
    for row in rows:
        z,_=integrated_query_logit(encoder,head,row,314159);live.append(float(z.sigmoid()))
    np.testing.assert_allclose(original,live,rtol=1e-12,atol=1e-12)
    recovered=PolynomialQueryReadout(encoder.total_payload,2);recovered.load_state_dict(copy.deepcopy(head.state_dict()))
    torch.testing.assert_close(recovered(data['features'],data['base']),head(data['features'],data['base']),rtol=0,atol=0)
    assert fingerprint(encoder)==before
    changed=copy.deepcopy(rows[0]);changed['target']=1-changed['target'];changed['bits']=[8,9]
    z,_=integrated_query_logit(encoder,head,changed,314159)
    assert float(z.sigmoid())==live[0]
    return dict(zero_residual_nests_original_logit_bitwise=True,polynomial_adjoint_matches_finite_difference=True,
        fitted_readout_integrated_query_matches_cached_prediction=True,restored_head_output_bitwise=True,
        frozen_encoder_fingerprint_preserved=True,target_and_bit_metadata_cannot_change_prediction=True,
        exact_fixed_feature_regularized_logistic_gradient=True,optimization_operator_coverage_complete=True,fit=fit)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan')
    p.add_argument('--contracts-only',action='store_true');p.add_argument('--contracts');a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Fresh plain tag required')
    torch.set_num_threads(1);started=time.perf_counter();contracts=numerical_contracts();rows=[];retention=[];parents=[]
    if not a.contracts_only:
        prerequisite=json.loads((ROOT/a.contracts).read_text())
        if prerequisite['status']!='completed' or prerequisite['source_sha256']!=sources():raise ValueError('Completed current-source readout prerequisites required')
        path=ROOT/a.plan;plan=json.loads(path.read_text())
        if json.loads(path.with_suffix('.status.json').read_text())['status']!='completed':raise ValueError('Completed parent cycle required')
        for job in plan['jobs']:
            if job['stage']!='pilot':continue
            r=json.loads((ROOT/job['result']).read_text());ckpath=(ROOT/job['result']).with_suffix('.progress.pt')
            if r['status']!='completed':raise ValueError('Completed selected encoder required')
            for n,sha in r['source_sha256'].items():
                if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=sha:raise ValueError('Changed encoder source')
            ck=torch.load(ckpath,weights_only=False);variants=[('selected',ck['best_state'])]
            if job['arm']=='native_full':variants.append(('initial',ck['initial']))
            for variant,weights in variants:
                encoder=B.make_model(SimpleNamespace(**r['args']));encoder.load_state_dict(weights);before=fingerprint(encoder)
                args=r['args'];fit_rows=joint_examples(args['gap'],args['fit_groups'],args['fit_data_seed'])
                dev_rows=joint_examples(args['gap'],args['dev_groups'],args['dev_data_seed'])
                fresh_rows=joint_examples(args['gap'],32,73001)
                if {tuple(x['inputs'][6:-1]) for x in fresh_rows}&{tuple(x['inputs'][6:-1]) for x in fit_rows+dev_rows}:
                    raise ValueError('Fresh suffix overlap')
                datasets={name:extract(encoder,examples,seeds) for name,examples,seeds in
                    [('fit',fit_rows,[314159,271829,271830,271831]),('dev',dev_rows,[314159]),('fresh',fresh_rows,[314159])]}
                fit=datasets['fit'];parity_heads={};ledger=[]
                for degree in (1,2):
                    head,learning=fit_readout(fit['features'],fit['base'],fit['labels'],degree)
                    scores={name:head_score(head,data) for name,data in datasets.items()};parity_heads[degree]=(head,scores)
                    infer_box={}
                    def infer():infer_box['value']=head(datasets['fresh']['features'][:4],datasets['fresh']['base'][:4])
                    infer_trace=capture(infer)
                    prefix_work=fit['prefix_unit_special_flops_per_target_estimate']*fit['targets']
                    parent_work=0. if variant=='initial' else r['work']['whole_fit_unit_special_flops_estimate']
                    fresh_prefix=datasets['fresh']['prefix_unit_special_flops_per_target_estimate']
                    head_infer=(infer_trace['arithmetic_flops']+infer_trace['special_function_evaluations'])/4
                    row=dict(arm=job['arm']+'_'+variant,degree=degree,parent_result=job['result'],
                        parent_result_sha256=hashlib.sha256((ROOT/job['result']).read_bytes()).hexdigest(),
                        checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),encoder_fingerprint=before,
                        fitting=learning,scores=scores,head_state={n:v.tolist() for n,v in head.state_dict().items()},
                        total_fit_gflops_estimate=(parent_work+prefix_work+learning['unit_special_flops'])/1e9,
                        parent_fit_gflops_estimate=parent_work/1e9,feature_replay_gflops_estimate=prefix_work/1e9,
                        head_fit_gflops=learning['unit_special_flops']/1e9,
                        head_fit_mflops_per_target_evaluation=learning['unit_special_flops']/learning['optimizer_target_evaluations']/1e6,
                        inference_mflops_per_query_estimate=(fresh_prefix+head_infer)/1e6,
                        inference_head_trace=infer_trace,encoder_updates=0,
                        scope='Frozen encoder plus learned residual readout. Total fit includes original selected-encoder fit, four-view prefix replay and exact head optimization; prefix work estimate, head actual. Initial encoder pays zero original fit. Fresh suffix set never selects head hyperparameters/weights.')
                    rows.append(row);ledger.append(row)
                first,second=parity_heads[1][1],parity_heads[2][1]
                ledger[-1]['fresh_gain_over_affine_bits']=first['fresh']['query_bits']-second['fresh']['query_bits']
                ledger[-1]['fresh_dependency_gate_passed']=second['fresh']['accuracy']>=.75 and second['fresh']['query_bits']<=.8
                ledger[-1]['fresh_interaction_gate_passed']=ledger[-1]['fresh_dependency_gate_passed'] and ledger[-1]['fresh_gain_over_affine_bits']>=.05
                # Additional bit labels only fit separate diagnostic probes; they
                # never enter the parity feature vector or parity predictor.
                bit_rows=[]
                for bit in (0,1):
                    probe,learning=fit_readout(fit['features'],torch.zeros_like(fit['base']),fit['bits'][:,bit],1)
                    scores={name:head_score(probe,data,base=torch.zeros_like(data['base']),labels=data['bits'][:,bit]) for name,data in datasets.items()}
                    bit_rows.append(dict(bit=bit,learning=learning,scores=scores,scope='Extra supervised retention diagnostic, not a parity predictor input'))
                retention.append(dict(arm=job['arm']+'_'+variant,bit_probes=bit_rows,
                    walsh={name:walsh(data['features']) for name,data in datasets.items()},
                    prefix_replays={name:{k:v for k,v in data.items() if k not in ('features','base','labels','bits')} for name,data in datasets.items()},
                    fitting_data_sha256=data_hash(fit_rows),development_data_sha256=data_hash(dev_rows),fresh_data_sha256=data_hash(fresh_rows)))
                assert before==fingerprint(encoder)
                parents.append(dict(result=job['result'],variant=variant,encoder_preserved=True))
                print(json.dumps(dict(completed_encoder=job['arm']+'_'+variant,affine_fresh=first['fresh']['query_bits'],
                    quadratic_fresh=second['fresh']['query_bits'],quadratic_accuracy=second['fresh']['accuracy'])),flush=True)
    result=dict(status='completed',args=vars(a),contracts=contracts,common_unit_ledger=rows,retention=retention,parents=parents,
        source_sha256=sources(),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        encoder_updates=0,fresh_data_seed=73001,fresh_groups=32,
        scope='Standard polynomial query readout over unchanged frozen integrated temporal encoders; initial-reservoir control included. Fresh predictions fixed before labels are evaluated, four declared encoder variants/eight readouts, one fitted encoder seed and synthetic distribution. No end-to-end feature-learning, dense comparison, full count-table or supremacy claim. Bit probes use extra labels/work separately.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
