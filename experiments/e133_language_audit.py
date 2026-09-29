"""Matched generic depth screen: context probes and explicitly partial work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e120_shared_tasks import Example, prefix, text_slice
from e120_shared_bench import inputs, loss_for, evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e133')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid or existing output')
    torch.set_num_threads(1)
    start=time.perf_counter()
    paths={d:Path(f'experiments/results/e133/generic_language_d{d}_s6_20260929.json') for d in (1,8)}
    records={d:json.loads(p.read_text()) for d,p in paths.items()}
    assert all(r['status']=='completed' for r in records.values())
    for name in ('fit','dev','context','epochs','bs','lr','seed','dim'):
        assert records[1]['args'][name]==records[8]['args'][name],name
    assert records[1]['data_sha256']==records[8]['data_sha256']
    assert records[1]['protocol']==records[8]['protocol']
    for r in records.values():
        assert r['config']['evidence_count']==0
        for name,hash_value in r['source_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==hash_value,name
    args=records[8]['args'];C=args['context'];N=args['dev']
    raw=text_slice(90000000,N+C)
    rows=[Example(prefix(raw[i-C:i]),int(raw[i]),str(90000000+i)) for i in range(C,N+C)]
    rng=np.random.default_rng(133)
    orders=[rng.permutation(C-1) for _ in rows]
    donors=rng.permutation(N)
    shuffled=[];unrelated=[]
    for j,r in enumerate(rows):
        obs=r.prefix.channels
        shuffled.append(Example(prefix(np.r_[obs[:-1][orders[j]],obs[-1]]),r.label,r.identity))
        unrelated.append(Example(prefix(np.r_[rows[donors[j]].prefix.channels[:-1],obs[-1]]),r.label,r.identity))
    result_rows={}
    for depth,r in records.items():
        saved=torch.load(paths[depth].with_suffix('.pt'),weights_only=False,map_location='cpu')
        model=SharedEventModel(**saved['config']);model.load_state_dict(saved['state_dict']);model.eval()
        original=evaluate(model,rows,args['bs'])
        assert abs(original['nll']-r['final']['dev']['nll'])<1e-7
        probes={name:evaluate(model,data,args['bs']) for name,data in
                (('prefix_order_shuffled',shuffled),('unrelated_prefix_same_last_character',unrelated))}
        for value in probes.values():value.pop('predictions',None);value['bpc']=value['nll']/np.log(2)
        d,K,F,classes=args['dim'],3,2*args['dim']+1,27
        def contractions(E,V,S,n):
            return int(2*(depth*E*K*F+V*d*F+K*(d+1)*S+n*((d+1)**2+(d+1)*classes)))
        training_flops=sum(contractions(e['training_prefix_packets'],e['training_value_evaluations'],
                                       e['training_scan_compositions'],args['fit']) for e in r['curve'])
        inference_flops=contractions(original['prefix_packets_processed'],original['selected_value_evaluations'],
                                    original['scan_compositions'],N)/N
        changed_layers=[i for i in range(depth) if all(
            r['final']['parameter_l2_changes'][f'layers.{i}.{name}']>0 for name in ('value','route','log_tau'))]
        assert len(changed_layers)==depth
        # Instrumented operator estimates, not hardware floating-point counters.
        model.train();model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as pf:
            scores,_,_,_=model(**inputs(rows[:args['bs']]))
            loss=loss_for(scores,rows[:args['bs']])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as pb:
            loss.backward()
        profiled={'training_forward_contraction_flops_batch':sum(e.flops for e in pf.key_averages()),
                  'backward_contraction_flops_batch':sum(e.flops for e in pb.key_averages()),
                  'batch_queries':args['bs'],'coverage':'PyTorch instrumented operator FLOP estimates; unsupported operations omitted; no optimizer, calibration or dataset preprocessing.'}
        result_rows[str(depth)]={'parameters':r['parameters'],'initial_dev_bpc':r['initial']['dev']['bpc'],
            'final_dev_bpc':r['final']['dev']['bpc'],'final_fit_bpc':r['final']['fit']['bpc'],
            'layer_value_route_tau_updates_verified':changed_layers,
            'last_character_only_bpc':r['last_character_only']['bpc'],
            'context_probes':probes,'training_forward_map_scan_flops':training_flops,
            'inference_forward_map_scan_flops_per_query':inference_flops,
            'profiled_contractions':profiled,'wall_s':r['wall_s'],'peak_rss_kib':r['peak_rss_kib']}
    result={'status':'completed','rows':result_rows,'protocol':records[8]['protocol'],
            'all_matching_source_ids_and_data_verified':True,
            'scope':'One seed, equal width/data/presentations, different parameter counts. Development screen; no tuned Transformer/RNN/state-space comparison or scaling claim.',
            'context_probe_scope':'Frozen input perturbations, not retrained controls. Shuffle/swap preserve count, timestamps and last character. Last-character-only changes count/time support and cannot establish a context advantage.',
            'work_scope':'Map/scan/score/head forward contractions estimated from recorded executed counts, 2 FLOPs/MAC. Excludes nonlinearities, sorting, normalization arithmetic, encoding, calibration, preprocessing, backward and optimizer. Profiled backward is a separate partial operator estimate. No hardware memory/energy measurement.',
            'physical_memory_reads':None,'physical_memory_writes':None,'energy_joules':None,
            'wall_s':time.perf_counter()-start,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths.values()]}}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result_rows),flush=True)


if __name__=='__main__':main()
