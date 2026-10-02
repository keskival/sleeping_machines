"""Attribute frozen state-access signals to payloads, clocks and layer state."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from dvs_persistent_state_probe import probe
from dvs_native_benchmark import sha


def partitions():
    q=list(range(32))
    return dict(payloads=q+list(range(32,96))+list(range(104,168)),
        clocks=q+list(range(96,104))+list(range(168,176)),
        layer0=q+list(range(32,104)),layer1=q+list(range(104,176)))


def fixed_probe(xf,xd,labels,dev_labels,config):
    center=xf.mean(0);scale=np.maximum(xf.std(0),.5)
    xt=(xf-center)/scale;xv=(xd-center)/scale
    if config['kind']=='linear':decoder=LogisticRegression(C=config['C'],max_iter=1000,random_state=681)
    else:decoder=SVC(C=config['C'],gamma=config['gamma']/(xt.shape[1]*max(float(xt.var()),1e-12)),probability=True,random_state=681)
    decoder.fit(xt,labels);probabilities=decoder.predict_proba(xv)
    from dvs_frozen_feature_diagnostic import nll
    quality=dict(accuracy=float((probabilities.argmax(1)==dev_labels).mean()),nll=nll(probabilities,dev_labels),
        probabilities=probabilities.tolist())
    return quality,dict(decoder=decoder,feature_center=center,feature_scale=scale)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--state-probe',required=True);a=p.parse_args();started=time.perf_counter()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    parent=json.loads((ROOT/a.state_probe).read_text());assert parent['status']=='completed'
    query_parent=json.loads((ROOT/parent['args']['reference_probe']).read_text())
    assert sha(ROOT/parent['args']['reference_probe'])==parent['reference_probe_sha256']
    selected=next(x for x in parent['rows'] if x['encoder']=='selected')
    if not selected['state_access_signal_passed']:raise ValueError('Completed selected-state access signal required')
    for name,h in parent['source_sha256'].items():assert sha(ROOT/name)==h,('Changed source',name)
    artifact=ROOT/parent['feature_artifact'];assert sha(artifact)==parent['feature_artifact_sha256']
    z=np.load(artifact,allow_pickle=False);labels=z['fit_labels'];dev_labels=z['dev_labels']
    assert len(labels)==984 and len(dev_labels)==192
    parts=partitions();assert [len(x) for x in parts.values()]==[160,48,104,104]
    assert sorted(set(parts['payloads'])|set(parts['clocks']))==list(range(176))
    assert set(parts['payloads'])&set(parts['clocks'])==set(range(32))
    assert sorted(set(parts['layer0'])|set(parts['layer1']))==list(range(176))
    rows=[];heads={}
    for arm in ('initial','selected'):
        xf=z[arm+'_fit'];xd=z[arm+'_dev'];assert xf.shape==(984,176) and xd.shape==(192,176)
        reference=next(r for r in parent['rows'] if r['encoder']==arm)
        for name,indices in parts.items():
            begin=time.perf_counter()
            quality,chosen,candidates,head,warnings=probe(xf[:,indices],xd[:,indices],labels,dev_labels)
            rows.append(dict(encoder=arm,partition=name,feature_dimension=len(indices),development=quality,
                fit_only_selected_decoder=chosen,all_fit_cv_candidates=candidates,
                query_nll_improvement=reference['query_probe_development']['nll']-quality['nll'],
                full_state_nll_difference=quality['nll']-reference['augmented_probe_development']['nll'],
                decoder_grid_wall_s=time.perf_counter()-begin,warnings=warnings,
                encoder_optimizer_updates=0,additional_core_replay_gflops_estimate=0.,decoder_solver_fit_flops=None))
            heads[arm+'_'+name]=head
            print(json.dumps(dict(encoder=arm,partition=name,nll=quality['nll'],accuracy=quality['accuracy'])),flush=True)
        query_reference=next(r for r in query_parent['rows'] if r['encoder']==arm)
        controls=[('query_with_state_setting',xf[:,:32],xd[:,:32],reference['fit_only_selected_decoder']['configuration']),
            ('state_with_query_setting',xf,xd,query_reference['fit_only_selected_decoder']['configuration'])]
        for name,train,evaluation,config in controls:
            begin=time.perf_counter();quality,head=fixed_probe(train,evaluation,labels,dev_labels,config)
            rows.append(dict(encoder=arm,partition=name,feature_dimension=train.shape[1],development=quality,
                fixed_configuration=config,selection='Configuration already selected by parent FIT-only CV; no grid or new selection',
                query_nll_improvement=reference['query_probe_development']['nll']-quality['nll'],
                full_state_nll_difference=quality['nll']-reference['augmented_probe_development']['nll'],
                decoder_grid_wall_s=time.perf_counter()-begin,warnings=[],encoder_optimizer_updates=0,
                additional_core_replay_gflops_estimate=0.,decoder_solver_fit_flops=None))
            heads[arm+'_'+name]=head
            print(json.dumps(dict(encoder=arm,partition=name,nll=quality['nll'],accuracy=quality['accuracy'])),flush=True)
    head_path=out.with_suffix('.heads.joblib');joblib.dump(heads,head_path)
    own=['experiments/dvs_state_partition_probe.py','experiments/theory/95_state_payload_clock_and_layer_partitions.md']
    result=dict(status='completed',args=vars(a),rows=rows,state_probe_sha256=sha(ROOT/a.state_probe),
        feature_artifact_sha256=sha(artifact),source_sha256={**parent['source_sha256'],**{name:sha(ROOT/name) for name in own}},
        partition_indices=parts,contracts_passed=1,decoder_artifact=str(head_path.relative_to(ROOT)),decoder_artifact_sha256=sha(head_path),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Four prespecified frozen diagnostic partitions, same9-cell/3-fold fitting-only decoder protocol; '
            'two additional per-encoder fixed configurations cross-check regularization without new selection. '
            'Cached causal features reused; no new encoder fit or replay. Prior core fitting/replay costs still belong to the pipeline. '
            'Solver arithmetic unknown. Differences do not isolate causal depth or certify population conditional information; '
            'dev already used for encoder selection. No core substitution, sparse inference, official test or superiority claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
