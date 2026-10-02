"""Fitting-only covariance from common race noise, without optimizer updates."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K
from dvs_route_content_audit import group


def moments(vectors):
    # [draw, clip, parameter]. The same draws preserve each clip's marginal
    # sample covariance; covariance under independent coupling is its sum/B^2.
    D,B,_=vectors.shape
    centered=vectors-vectors.mean(0,keepdim=True)
    marginal=centered.square().sum((0,2))/(D-1)
    common=float(centered.mean(1).square().sum()/(D-1))
    independent=float(marginal.sum()/B**2)
    cross=float((centered.sum(1).square().sum()-centered.square().sum())/((D-1)*B**2))
    if abs(common-independent-cross)>1e-10*max(1.,common):
        raise AssertionError('Covariance decomposition')
    return dict(common_batch_trace_covariance=common,
        independent_batch_trace_covariance_estimate=independent,
        off_diagonal_covariance_contribution=cross,
        common_over_independent_ratio=common/independent if independent>1e-24 else None,
        marginal_clip_trace_covariances=marginal.tolist(),
        mean_batch_gradient_norm=float(vectors.mean((0,1)).norm()),sample_count=D)


def numerical_contracts():
    v=torch.tensor([[[1.,0.],[2.,1.]],[[2.,2.],[4.,0.]],[[0.,-1.],[1.,3.]]],dtype=torch.float64)
    row=moments(v)
    centered=v-v.mean(0)
    covariance=torch.einsum('dbp,dcp->bc',centered,centered)/2
    torch.testing.assert_close(torch.tensor(row['common_batch_trace_covariance']),covariance.sum().float()/4)
    repeated=v[:,:1].expand(-1,4,-1)
    repeat=moments(repeated)
    assert abs(repeat['common_over_independent_ratio']-4)<1e-12
    return dict(contracts_passed=2,covariance_direct_matrix_matches=True,
        identical_four_clip_common_over_independent_ratio=repeat['common_over_independent_ratio'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--native',action='append',required=True);p.add_argument('--draws',type=int,default=32)
    p.add_argument('--prefixes',type=int,default=4);a=p.parse_args()
    if not 2<=a.draws<=64 or not 2<=a.prefixes<=8:raise ValueError('Bounded audit required')
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    started=time.perf_counter();torch.set_num_threads(1);contracts=numerical_contracts()
    results=[];work=[];source={};generator_before=torch.get_rng_state().clone()
    for name in a.native:
        parent=json.loads((ROOT/name).read_text());assert parent['status']=='completed'
        assert parent['args'].get('terminal_risk')=='local'
        for path,h in parent['source_sha256'].items():assert N.sha(ROOT/path)==h,('Changed source',path)
        source.update(parent['source_sha256']);config=argparse.Namespace(**parent['args'])
        rows,_,metadata=N.load(config);assert metadata==parent['data'];rows=rows[:a.prefixes]
        model=N.make_model(config);checkpoint=(ROOT/name).with_suffix('.progress.pt')
        saved=torch.load(checkpoint,weights_only=False);assert saved['source_sha256']==parent['source_sha256']
        model.load_state_dict(saved['best_state']);before={n:v.detach().clone() for n,v in model.named_parameters()}
        parameter_groups={};offset=0
        for n,v in model.named_parameters():
            parameter_groups.setdefault(group(n),[]).extend(range(offset,offset+v.numel()));offset+=v.numel()
        vectors=[];losses=[];choices=[]
        for draw in range(a.draws):
            clip_vectors=[];clip_losses=[];clip_choices=[]
            for row in rows:
                box={}
                def forward():
                    z,state,_=K.forward(model,[row],730001+1009*draw)
                    box.update(loss=F.cross_entropy(z,torch.tensor([row['target']])),state=state)
                def backward():
                    gradients=torch.autograd.grad(box['loss'],tuple(model.parameters()),allow_unused=True)
                    box['gradient']=torch.cat([torch.zeros_like(v).flatten() if g is None else g.detach().flatten()
                        for v,g in zip(model.parameters(),gradients)]).double()
                if draw==0:
                    for stage,fn in [('forward',forward),('backward',backward)]:
                        tr=N.capture(fn)
                        if not tr['formula_coverage_complete']:raise AssertionError('Incomplete work')
                        work.append(dict(native=name,clip_index=row['index'],stage=stage,
                            arithmetic_flops=tr['arithmetic_flops'],special_function_evaluations=tr['special_function_evaluations'],
                            exponential_random_draws=tr['exponential_random_draws']))
                else:forward();backward()
                clip_vectors.append(box['gradient']);clip_losses.append(float(box['loss'].detach()))
                clip_choices.append(torch.stack(box['state']['winners']).flatten())
            vectors.append(torch.stack(clip_vectors));losses.append(clip_losses);choices.append(torch.stack(clip_choices))
        vectors=torch.stack(vectors);choices=torch.stack(choices)
        grouped={'all_parameters':moments(vectors)}
        grouped.update({g:moments(vectors[:,:,indices]) for g,indices in parameter_groups.items()})
        pair_agreement=[float((choices[:,i]==choices[:,j]).double().mean())
            for i in range(len(rows)) for j in range(i+1,len(rows))]
        assert all(torch.equal(before[n],v.detach()) for n,v in model.named_parameters())
        results.append(dict(native=name,result_sha256=N.sha(ROOT/name),checkpoint_sha256=N.sha(checkpoint),
            seed=config.seed,fitting_prefixes=len(rows),fit_indices=[r['index'] for r in rows],groups=grouped,
            mean_fitting_loss=float(torch.tensor(losses).mean()),mean_common_noise_winner_agreement=sum(pair_agreement)/len(pair_agreement),
            optimizer_updates=0,weights_preserved=True,
            admission_supports_independent_rows=grouped['route_maps']['common_over_independent_ratio'] is not None
                and grouped['route_maps']['common_over_independent_ratio']>=1.20))
        print(json.dumps(dict(seed=config.seed,route_ratio=grouped['route_maps']['common_over_independent_ratio'],
            full_ratio=grouped['all_parameters']['common_over_independent_ratio'])),flush=True)
    # Model initialization changes RNG, while K.forward preserves it. The audit
    # never uses the global post-initialization state as an unreported noise source.
    total=sum(w['arithmetic_flops']+w['special_function_evaluations'] for w in work)*a.draws
    own=['experiments/dvs_noise_covariance_audit.py','experiments/dvs_route_content_audit.py',
        'experiments/theory/93_race_noise_coupling_and_effective_batch_size.md']
    source.update({path:N.sha(ROOT/path) for path in own})
    result=dict(status='completed',args=vars(a),contracts=contracts,models=results,source_sha256=source,
        stage_work=work,whole_audit_unit_special_flops_estimate=total,formula_coverage_complete=True,
        independent_row_candidate_admitted=all(r['admission_supports_independent_rows'] for r in results),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Fixed fitting prefixes, frozen dev-selected native weights, independent whole-history audit draws. '
            'Common batch covariance measured; independent covariance estimated from the same marginal samples. '
            'Finite sample, not conditional node covariance, optimizer trajectory or causal seed-failure proof. '
            'Work uses complete first-draw per-clip forward/backward traces times draws; reporting reductions separate.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
