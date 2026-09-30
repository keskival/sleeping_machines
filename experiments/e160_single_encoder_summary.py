"""Summarize completed consolidation/depth evidence and verify its provenance.

Private development, a reused same-speaker audit and one CPU timing observation
are reported separately. Partial operation ledgers are not complete FLOPs or
energy. Historical scheduler-overwritten runs are excluded from calibrated
depth comparisons, with their actual-rate discrepancy explicitly retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    out=Path('experiments/results/e160')/(args.tag+'.json')
    out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique output required')
    paths={
        'combined':'e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json',
        'clean_single':'e150/single_state_n6144_s6_e3_20260930.json',
        'paired_single':'e152/nuisance_state_n6144_s6_e2_20260930.json',
        'absorption_contract':'e149/readout_absorption_contract_v2_20260930.json',
        'identity_contract':'e151/identity_depth_contract_20260930.json',
        'nuisance_contract':'e153/nuisance_geometry_contract_20260930.json',
        'adam_replay':'e156/adam_step_audit_20260930.json',
        'original_depth_replay':'e157/original_d12_audit_20260930.json',
        'conditioned_depth_replay':'e157/conditioned_d12_audit_20260930.json',
        'uncalibrated_d6':'e155/matched_d6_n6144_s6_e1_20260930.json',
        'uncalibrated_d12':'e155/identity_d12_n6144_s6_e1_20260930.json',
        'calibrated_d6':'e159/calibrated_d6_n6144_s6_e1_20260930.json',
        'calibrated_d12':'e159/calibrated_d12_n6144_s6_e1_20260930.json',
        'reused_audit':'e154/single_encoder_audit_20260930.json'}
    paths={k:str(Path('experiments/results')/p) for k,p in paths.items()}
    records={k:json.loads(Path(p).read_text()) for k,p in paths.items()}
    sources={}
    for key,record in records.items():
        if record['status']!='completed':raise ValueError('Incomplete: '+key)
        for path,expected in record['source_sha256'].items():
            if digest(path)!=expected:raise ValueError('Changed executed source: '+path)
            sources[path]=expected
    for path,expected in records['reused_audit']['file_sha256'].items():
        if digest(path)!=expected:raise ValueError('Changed audited result/checkpoint: '+path)
    d6=records['calibrated_d6'];d12=records['calibrated_d12']
    if d6['fit_absolute_ids']!=d12['fit_absolute_ids'] or d6['dev_absolute_ids']!=d12['dev_absolute_ids']:
        raise ValueError('Depth arms use different examples')
    if d6['checkpoint_sha256']!=d12['checkpoint_sha256']:raise ValueError('Different starting checkpoint')
    if d6['verified_initial_optimizer_rates'][0]!=d12['verified_initial_optimizer_rates'][0]:
        raise ValueError('Unmatched old group rates')
    for run in (d6,d12):
        if run['final']['lr']!=run['verified_initial_optimizer_rates'][0]:
            raise ValueError('Actual epoch rate differs from verified rate')
    audit=records['reused_audit']
    if set(audit['absolute_ids'])&set(d6['fit_absolute_ids']+d6['dev_absolute_ids']):
        raise ValueError('Audit overlaps current fitting/development')
    paired=records['paired_single'];combined=records['combined']
    selected=min(paired['curve'],key=lambda row:(-row['dev']['correct'],row['dev']['nll']))
    combined_best=min(combined['curve'],key=lambda row:(-row['dev']['correct'],row['dev']['nll']))
    dev6=d6['final']['dev'];dev12=d12['final']['dev']
    if dev6['labels']!=dev12['labels']:raise ValueError('Different development order')
    depth_pairs=dict(
        d12_only_correct=sum(p==y and q!=y for p,q,y in zip(dev12['predictions'],dev6['predictions'],dev6['labels'])),
        d6_only_correct=sum(p!=y and q==y for p,q,y in zip(dev12['predictions'],dev6['predictions'],dev6['labels'])))
    rows={}
    for name,run in (('calibrated_d6',d6),('calibrated_d12',d12)):
        final=run['final'];work=final['training']
        rows[name]=dict(parameters=run['deployed_parameters'],fit_correct=final['fit']['correct'],
            dev_correct=final['dev']['correct'],dev_nll=final['dev']['nll'],
            online_nll=work['online_nll'],training_wall_s=work['training_wall_s'],
            full_recorded_wall_s=run['wall_s'],max_rss_kib=run['max_rss_kb'],
            actual_old_rate=final['lr'],initial_added_clock_s=run['initial_added_clock_s'],
            partial_training_forward_work={k:work[k] for k in (
                'source_events','new_packets','source_table_projection_macs','source_transported_scalars',
                'source_payload_sum_scalars','event_projection_macs','gate_macs','state_compositions','clock_candidates')},
            layer_gradient_norm_sums=work['layer_gradient_norm'])
    guards={}
    for job in ('e159_calibrated_d6_n6144_20260930','e159_calibrated_d12_n6144_20260930',
                'e154_single_encoder_audit_20260930'):
        path=Path('experiments/queue')/('runner_'+job+'.out');log=path.read_text()
        if f'done {job} (exit 0)' not in log:raise ValueError('Missing successful safe-run marker')
        samples=[int(x) for x in re.findall(r'MemAvailable=(\d+)MB',log)]
        guards[job]=dict(log=str(path),memory_floor_mib=8192,
            address_space_cap_kib=4200000 if job.startswith('e159') else 3600000,
            process_group_rss_cap_kib=2600000 if job.startswith('e159') else 2000000,
            timeout_s=1800 if job.startswith('e159') else 600,
            minimum_sampled_available_mib=min(samples) if samples else None)
    result=dict(status='completed',result_files=paths,
        result_sha256={key:digest(path) for key,path in paths.items()},
        source_sha256={**sources,str(Path(__file__)):digest(__file__)},
        strongest_family_private=dict(correct=combined_best['dev']['correct'],n=512),
        selected_single=dict(epoch=selected['epoch'],correct=selected['dev']['correct'],n=512,
            nll=selected['dev']['nll'],parameters=paired['deployed_parameters']),
        depth_rows=rows,paired_depth_development=depth_pairs,
        reused_audit_rows=audit['rows'],reused_audit_paired=audit['paired'],
        scheduler_discrepancy={key:dict(intended_old_rate=records[key]['starting_old_learning_rate'],
            actual_epoch_rate=records[key]['final']['lr']) for key in ('uncalibrated_d6','uncalibrated_d12')},
        resource_guards=guards,
        protocol=__doc__.strip(),energy_joules=None,
        inference_timing_scope='One warmed sequential CPU evaluation per checkpoint on the same reused audit; packing/query included, loading excluded; timing order and CPU variation are not statistically controlled',
        training_cost_scope='Head view generation/teacher/head fitting and inherited E143 training must be charged; partial ledgers do not include complete backward/optimizer/sorting/traffic',
        official_test_evaluated=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('strongest_family_private','selected_single','depth_rows','paired_depth_development')}),flush=True)


if __name__=='__main__':main()
