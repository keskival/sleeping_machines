"""Reader-first MD/PDF report, assembled from the same editorial blocks.

Superseded reports with leaked comparisons are quarantined in experiments/archive/invalid_protocol/. Raw
experiment histories remain in FINDINGS.md and the numbered theory notes.
"""
from datetime import date
import hashlib
import html
import json
import math
from pathlib import Path
import re
import runpy


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"experiments/results"
FIG = ROOT/"report/figures"
full_bank_comparison=runpy.run_path(str(ROOT/'report/language_scaling.py'))['full_bank_comparison']
_historical_result_view=runpy.run_path(str(ROOT/'experiments/legacy_batched_driver_binding.py'))['historical_view']


def read(path):
    relative=Path(path)
    if (relative.parts and relative.parts[0] in ('e63','e79')) or 'aws_e79wk_' in str(relative):
        raise ValueError(f'Quarantined target-leakage result is ineligible for report: {path}; use causal E173')
    row = json.loads((RES/path).read_text())
    if row.get("status", "completed") != "completed":
        raise ValueError(f"Report requires completed result: {path}")
    return _historical_result_view(row)


def results():
    tasks = {task: read(f"e120/{task}_d8_20260929.json")
             for task in ("language", "market", "temporal", "mnist", "modular", "dvs")}
    tasks["recall"] = read("e120/recall_d8_supported_20260929.json")
    tasks["phase_fixed"] = read("e121/shared_d2_phase_s6_200.json")
    tasks["phase"] = read("e121/shared_d2_guard_s6_200.json")
    tasks["plain"] = read("e121/shared_d2_plain_s6_200.json")
    tasks["shd_control"] = read("e122/d8_n2048_control_s6.json")
    tasks["shd_invariance"] = read("e122/d8_n2048_invariance_s6.json")
    tasks["phase_only"] = read("e124/modular_phase_only_s6_200.json")
    tasks["work_audit"] = read("e124/consolidated_work_20260929.json")
    tasks["shd_scaled"] = read("e122/d8_n4096_invariance_continue_s6_e2.json")
    tasks["shd_pool_mean"] = read("e122/d8_n4096_pool_control_s6.json")
    tasks["shd_pool_weighted"] = read("e122/d8_n4096_pool_weighted_s6.json")
    tasks["temporal_shallow"] = read("e120/temporal_d2_20260929.json")
    tasks["recall_shallow"] = read("e120/recall_d2_20260929.json")
    tasks["breadth_work"] = read("e124/breadth_work_counted_20260929.json")
    tasks["shd_bridge"] = read("e122/d8_n4096_bridge_s6.json")
    tasks["shd_bridge_frozen"] = read("e122/d8_n4096_bridge_frozen_s6.json")
    tasks["shd_key_value"] = read("e131/key_value_comparison_20260929.json")
    tasks["generic_language"] = {depth: read(f"e133/generic_language_d{depth}_s6_20260929.json") for depth in (1,8)}
    tasks["generic_language_audit"] = read("e133/generic_language_audit_20260929.json")
    tasks["complete_work"] = read("e172/complete_work_v2_20260930.json")
    tasks["training_work"] = read("training_work/local_total_training_work_v4_20260930T123846Z.json")
    tasks["event_language_work"] = read("event_language_work/local_event_language_work_20260930T124646Z.json")
    tasks["online_language"] = read("online_language/local_online_language_20260930T121741Z.json")
    tasks["stream_contract"] = read("e175/stream_language_contract_20260930.json")
    tasks["stream_training"] = read("e176/stream_language_d8_20260930.json")
    tasks["token_training"] = read("e178/prefix_language_d8_20260930.json")
    tasks["language_scaling"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_scale_capacity_*_20260930T153653Z.json"))]
    tasks["language_memory"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_memory_w128_D131072_*_20260930T155000Z.json"))]
    tasks["language_selective"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_selective_w128_D131072_*_20260930T161050Z.json"))]
    tasks["language_scaleup"] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/"parallel_language").glob("local_staged_language_*Z.json"))]
    tasks['language_race'] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/'parallel_language').glob('local_indexed_language_*Z.json'))]
    tasks['language_full_sparse'] = [read(str(path.relative_to(RES)))
        for path in sorted([*(RES/'parallel_language').glob('local_full_sparse_language_*Z.json'),
                            *(RES/'parallel_language').glob('aws_full_sparse_language_*Z.json')])]
    tasks['integrated_online_language'] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/'online_language').glob('local_integrated_online_backbone_*Z.json'))]
    tasks['native_language'] = [r for path in sorted((RES/'native_language').glob('local_native_language_*Z.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed' and 'final' in r
        and r['args']['fit']>=2048 and r['args']['dev']==8192]
    tasks['count_carrying_language'] = [r for path in sorted((RES/'count_carrying_language').glob('*Z.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed' and 'final' in r
        and r['args']['fit']>=2048 and r['args']['dev']==8192]
    tasks['count_credit_pair']=[]
    for path in sorted((RES/'diagnostics').glob('local_count_credit64_analysis_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')=='completed':
            for entry in r['common_unit_ledger']:
                result=ROOT/entry['result']
                if hashlib.sha256(result.read_bytes()).hexdigest()!=entry['result_sha256']:
                    raise ValueError('Count-credit ledger result changed')
            tasks['count_credit_pair'].append(r)
    tasks['deep_core_attribution']=[]
    for path in sorted((RES/'diagnostics').glob('local_deep_core_attribution_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')=='completed' and r.get('weights_preserved') and r.get('optimizer_steps')==0:
            for entry in r['models']:
                if hashlib.sha256((ROOT/entry['result']).read_bytes()).hexdigest()!=entry['result_sha256']:
                    raise ValueError('Frozen deep-core source result changed')
            tasks['deep_core_attribution'].append(r)
    tasks['deep_memory_preflight']=[]
    for path in sorted((RES/'diagnostics').glob('local_deep_feature_preflight_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')=='completed':
            for entry in r['common_unit_ledger']:
                if hashlib.sha256((ROOT/entry['result']).read_bytes()).hexdigest()!=entry['result_sha256']:
                    raise ValueError('Integrated memory accounting result changed')
            tasks['deep_memory_preflight'].append(r)
    tasks['value_credit_pair']=[]
    for path in sorted((RES/'diagnostics').glob('local_value_credit_analysis_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')=='completed':
            for entry in r['common_unit_ledger']:
                if hashlib.sha256((ROOT/entry['result']).read_bytes()).hexdigest()!=entry['result_sha256']:
                    raise ValueError('Value-credit result changed')
            tasks['value_credit_pair'].append(r)
    tasks['value_credit_frozen']=[]
    for path in sorted((RES/'diagnostics').glob('local_value_credit_frozen_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')=='completed':
            if hashlib.sha256((ROOT/r['analysis_result']).read_bytes()).hexdigest()!=r['analysis_sha256']:
                raise ValueError('Frozen value-credit analysis changed')
            tasks['value_credit_frozen'].append(r)
    for key,pattern in [('balanced_joint','local_balanced_joint_*_analysis.json'),
                        ('joint_readout','local_joint_readout_*_frozen.json'),
                        ('joint_outcome','local_joint_outcome_*_analysis.json'),
                        ('joint_replication','local_joint_confirmation_*_analysis.json')]:
        tasks[key]=[]
        for path in sorted((RES/'diagnostics').glob(pattern)):
            r=read(str(path.relative_to(RES)))
            if r.get('status')!='completed':continue
            for entry in r['common_unit_ledger']:
                name=entry.get('result',entry.get('parent_result'))
                sha=entry.get('result_sha256',entry.get('parent_result_sha256'))
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                    raise ValueError('Joint learning source result changed')
            tasks[key].append(r)
    tasks['uniform_address']=[r for path in sorted((RES/'diagnostics').glob('local_joint_publication_*_contracts.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed'
        and r.get('actual_bank_uniform_pair_law_alias_verified')]
    tasks['joint_stateful_table']=[r for path in sorted((RES/'diagnostics').glob('local_joint_stateful_table_*Z.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed']
    tasks['dvs_practical_controls']=[r for path in sorted((RES/'dvs_calibration').glob('local_dvs_calibration_*_controls.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed']
    tasks['dvs_practical_native']=[]
    for path in sorted((RES/'diagnostics').glob('local_dvs_native_analysis_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')!='completed':continue
        for name,sha in r['inputs'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Changed real-packet comparison evidence')
        tasks['dvs_practical_native'].append(r)
    tasks['dvs_practical_inference']=[r for path in sorted((RES/'diagnostics').glob('local_dvs_practical_inference_*Z.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed']
    appendix=runpy.run_path(str(ROOT/'report/current_dvs_appendix.py'))
    tasks['current_dvs_appendix']=appendix['load'](read)
    tasks['native_language_batched']=runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))['load'](read)
    tasks['current_language_status']=runpy.run_path(str(ROOT/'report/current_language_status.py'))['load'](tasks['native_language_batched'])
    tasks['reception_evidence']=runpy.run_path(str(ROOT/'report/reception_evidence.py'))['load'](read)
    historical=runpy.run_path(str(ROOT/'report/depth_sampling_evidence.py'))['history_read']
    historical_read=lambda path:historical(read,path)
    tasks['route_calibration_evidence']=runpy.run_path(str(ROOT/'report/route_calibration_state_scope.py'))['load'](historical_read)
    tasks['bridge_training_evidence']=runpy.run_path(str(ROOT/'report/trained_route_coverage_evidence.py'))['load'](read)
    tasks['score_bound_evidence']=runpy.run_path(str(ROOT/'report/language_replay_driver_evidence.py'))['load_score_evidence'](historical_read)
    tasks['dvs_noise_covariance']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_noise_covariance_audit_*Z.json'))]
    tasks['dvs_persistent_state_probes']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_persistent_state_probe_*Z.json'))]
    tasks['dvs_state_partition_probes']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_state_partition_probe_*Z.json'))]
    for r in tasks['dvs_noise_covariance']:
        for model in r['models']:
            if hashlib.sha256((ROOT/model['native']).read_bytes()).hexdigest()!=model['result_sha256']:
                raise ValueError('Changed noise audit parent')
    for r in tasks['dvs_persistent_state_probes']:
        for name,digest in [(r['native_result'],r['native_result_sha256']),
                (r['args']['reference_probe'],r['reference_probe_sha256'])]:
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed persistent-state parent')
    for r in tasks['dvs_state_partition_probes']:
        if hashlib.sha256((ROOT/r['args']['state_probe']).read_bytes()).hexdigest()!=r['state_probe_sha256']:
            raise ValueError('Changed state partition parent')
    tasks['dvs_frozen_features']=[]
    for path in sorted((RES/'diagnostics').glob('local_dvs_frozen_features_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')!='completed':continue
        if hashlib.sha256((ROOT/r['args']['native']).read_bytes()).hexdigest()!=r['native_result_sha256']:
            raise ValueError('Changed frozen-feature parent')
        tasks['dvs_frozen_features'].append(r)
    tasks['dvs_compact_controls']=[]
    for path in sorted((RES/'diagnostics').glob('local_dvs_compact_controls_*Z.json')):
        r=read(str(path.relative_to(RES)))
        if r.get('status')!='completed':continue
        for key,digest in [('controls',r['controls_sha256']),('audit',r['inference_audit_sha256'])]:
            if hashlib.sha256((ROOT/r['args'][key]).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed compact-kernel comparator')
        tasks['dvs_compact_controls'].append(r)
    tasks['dvs_clock_completed']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'dvs_native').glob('local_dvs_clock_full_*Z.json'))]
    tasks['dvs_external_capacity_completed']=[read(str(path.relative_to(RES)))
        for name in ('curie_dvs_clock_p16d2pool8_s6_20261002T172500Z.json',
                     'curie_dvs_clock_p16d4pool2_s6_20261002T172500Z.json')
        if (path:=RES/'dvs_native'/name).exists()]
    tasks['dvs_state_choice_smokes']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'dvs_native').glob('local_dvs_state_choice_smoke_*Z.json'))]
    tasks['dvs_paired_choice_smokes']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'dvs_native').glob('local_dvs_paired_choice_smoke_pool8_*Z.json'))]
    tasks['dvs_branch_exposure_audits']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_branch_exposure_audit_*Z.json'))]
    tasks['race_teacher_expectation_contracts']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_race_teacher_expectation_contracts_*Z.json'))]
    tasks['race_replay_law_contracts']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_race_replay_law_contracts_*Z.json'))]
    tasks['local_credit_geometry_contracts']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_credit_geometry_contracts_*Z.json'))]
    tasks['dvs_evolution_offset_smokes']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'dvs_native').glob('local_dvs_evolution_offset_*_smoke_*Z.json'))]
    tasks['dvs_evolution_offset_comparisons']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_evolution_offset_comparison_*Z.json'))]
    for audit in tasks['dvs_branch_exposure_audits']:
        for model in audit['models']:
            if hashlib.sha256((ROOT/model['native']).read_bytes()).hexdigest()!=model['result_sha256']:
                raise ValueError('Changed branch-exposure parent')
    for audit in tasks['dvs_evolution_offset_comparisons']:
        for row in audit['common_unit_ledger']:
            if hashlib.sha256((ROOT/row['result']).read_bytes()).hexdigest()!=row['result_sha256']:
                raise ValueError('Changed offset comparison parent')
    tasks['dvs_paired_choice_contracts']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_paired_choice_contracts_*Z.json'))]
    tasks['dvs_calibration_contracts']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_credit_calibration_contracts_*Z.json'))]
    tasks['dvs_prefix_replay_savings']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('aws_prefix_replay_smokes_*_analysis.json'))]
    for audit in tasks['dvs_prefix_replay_savings']:
        for row in audit['rows']:
            if hashlib.sha256((ROOT/row['result']).read_bytes()).hexdigest()!=row['result_sha256']:
                raise ValueError('Changed prefix-replay saving parent')
    tasks['dvs_credit_balance_audits']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_credit_gradient_balance_*Z.json'))]
    tasks['dvs_credit_comparisons']=[]
    for path in sorted((RES/'diagnostics').glob('local_dvs_credit_comparison_*Z.json')):
        r=read(str(path.relative_to(RES)))
        for entry in r['common_unit_ledger']:
            if hashlib.sha256((ROOT/entry['result']).read_bytes()).hexdigest()!=entry['result_sha256']:
                raise ValueError('Changed DVS credit evidence')
        tasks['dvs_credit_comparisons'].append(r)
    tasks['dvs_suffix_credit_audits']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_counterfactual_route_audit_*Z.json'))]
    tasks['dvs_clock_variance_audits']=[read(str(path.relative_to(RES)))
        for path in sorted((RES/'diagnostics').glob('local_dvs_joint_credit_variance_*Z.json'))]
    for key in ('dvs_suffix_credit_audits','dvs_clock_variance_audits'):
        for audit in tasks[key]:
            for model in audit['models']:
                expected=model.get('native_result_sha256',model.get('result_sha256'))
                if hashlib.sha256((ROOT/model['native']).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Changed frozen credit audit parent')
    for audit in tasks['dvs_credit_balance_audits']:
        for model in audit['models']:
            if hashlib.sha256((ROOT/model['native']).read_bytes()).hexdigest()!=model['result_sha256']:
                raise ValueError('Changed gradient-balance parent')
    tasks['dvs_batched_smoke_admission']=[]
    for path in sorted((RES/'diagnostics').glob('local_dvs_batched_smoke_admission_*Z.json')):
        r=read(str(path.relative_to(RES)))
        for entry in r['rows']:
            if hashlib.sha256((ROOT/entry['result']).read_bytes()).hexdigest()!=entry['result_sha256']:
                raise ValueError('Changed batched smoke evidence')
        tasks['dvs_batched_smoke_admission'].append(r)
    tasks['delay_language'] = [r for path in sorted((RES/'clock_feature_language').glob('local_delay_feature_*Z.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed' and 'final' in r
        and r['args']['fit']>=2048 and r['args']['dev']==8192]
    tasks['native_event'] = [r for path in sorted((RES/'native_event').glob('local_native_event_*Z.json'))
        if (r:=read(str(path.relative_to(RES)))).get('status')=='completed' and 'final' in r
        and r['args']['fit_targets']>=512 and r['args']['epochs']>=8]
    tasks['episodic_language'] = [read(str(path.relative_to(RES)))
        for path in sorted((RES/'episodic_language').glob('local_episodic_pair_*Z.json'))]
    tasks['historical_write_contracts'] = [read(str(path.relative_to(RES))) for path in sorted((RES/'episodic_language').glob('local_write_credit_contracts_*Z.json'))]
    tasks['head_diagnosis'] = [read(str(path.relative_to(RES))) for path in sorted((RES/'diagnostics').glob('local_parallel_head_diagnosis_*Z.json'))]
    tasks['parallel_head_contracts'] = [read(str(path.relative_to(RES)))
        for path in sorted([*(RES/'episodic_language').glob('local_parallel_head_contracts_*Z.json'),
                            *(RES/'episodic_language').glob('local_parallel_head_accum_contracts_*Z.json')])]
    audit_path="parallel_language/local_language_representation_20260930T162337Z.json"
    tasks['language_representation'] = read(audit_path) if (RES/audit_path).exists() else None
    tasks["parallel_contract"] = read("parallel_language/local_parallel_language_contract_v3_20260930T153300Z.json")
    tasks["mechanisms"] = {
        "timing": read("e35/free.json"),
        "motifs": read("e34/d2_K15_ph1.5_sum_t0.6_a1_b0.5_latest_T0_W4.2_iv3.json"),
        "deep_order": read("e54/D4_L2_S5R4_T0.3.json"),
        "order_curve": read("e53/d3_S5R4_t0.6_curve_latest_T0_b0.5_m0.9_g0.9.json"),
        "order_references": {n: read(f"e36/transformer_e53_e53_n{n//1000}k_wd0.json")
                             for n in (2000, 5000, 10000, 20000, 40000)},
        "deep_reference": read("e36/transformer_e54_e54_n40k_wd0.json"),
        "timing_reference": read("e36/transformer_e27_rel.json"),
        "motif_reference": read("e36/transformer_e28_rel.json"),
    }
    tasks["shd_full_values"] = read("e134/full_value_comparison_20260929.json")
    content_path = "e135/content_comparison_20260929.json"
    tasks["shd_content"] = read(content_path) if (RES/content_path).exists() else None
    exchange_paths = {name: f"e136/scattering_{name}_d12_n1024_s6_e3_20260929.json"
                      for name in ("state", "packets")}
    exchange_paths["ablation"] = "e136/scattering_angle_ablation_20260929.json"
    tasks["shd_exchange"] = ({name: read(path) for name,path in exchange_paths.items()}
                              if all((RES/path).exists() for path in exchange_paths.values()) else None)
    compact_path = "e137/compact_comparison_20260930.json"
    tasks["shd_compact"] = read(compact_path) if (RES/compact_path).exists() else None
    for key,path in (
        ("shd_warm", "e122/d8_n6144_best_warm_s6_e1_20260930.json"),
        ("shd_fine", "e139/d8_fine_source_n6144_warm_s6_e1_20260930.json"),
        ("shd_phase", "e140/d8_phase_source_n6144_warm_s6_e1_20260930.json"),
        ("shd_local", "e141/d8_new_only_n6144_warm_s6_e1_20260930.json"),
        ("shd_state_residual", "e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json"),
        ("shd_state_ablation", "e145/state_residual_ablation_20260930.json"),
        ("shd_state_summary", "e146/event_state_summary_20260930.json"),
        ("shd_single_clean", "e150/single_state_n6144_s6_e3_20260930.json"),
        ("shd_single_paired", "e152/nuisance_state_n6144_s6_e2_20260930.json"),
        ("shd_calibrated_d6", "e159/calibrated_d6_n6144_s6_e1_20260930.json"),
        ("shd_calibrated_d12", "e159/calibrated_d12_n6144_s6_e1_20260930.json"),
        ("shd_single_audit", "e154/single_encoder_audit_20260930.json"),
        ("shd_observer_depth", "e163/observer_depth_n6144_s6_e1_20260930.json"),
        ("shd_observer_audit", "e164/observer_depth_audit_20260930.json"),
        ("shd_selected_prefix", "e165/selected_prefix_20260930.json")):
        tasks[key] = (read(path) if (RES/path).exists() and
            json.loads((RES/path).read_text()).get("status") == "completed" else None)
    tasks['gym_screen']=[]
    seen_gym=set()
    for gym_plan in sorted((ROOT/'experiments/gym/plans').glob('aws_fast_matrix*/manifest.json')):
        for job in json.loads(gym_plan.read_text())['jobs']:
            if job['stage']!='pilot' or job['domain']=='tabular' or job['tag'] in seen_gym:continue
            seen_gym.add(job['tag']);path=ROOT/job['result']
            if path.exists():
                r=json.loads(path.read_text())
                if r.get('status')=='completed' and 'final' in r:tasks['gym_screen'].append(dict(job=job,result=r))
    tasks['state_credit']=[]
    for path in sorted((RES/'state_credit').glob('*_pilot.json')):
        r=json.loads(path.read_text())
        if r.get('status')=='completed' and 'final' in r:
            for name,sha in r['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('State-credit source changed: '+name)
            tasks['state_credit'].append(r)
    tasks['language_learning_audit']=None
    for path in sorted((RES/'diagnostics').glob('local_language_learning_audit_*.json')):
        r=json.loads(path.read_text())
        if r.get('status')=='completed':
            for name,sha in r['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                    raise ValueError('Language-learning audit source changed: '+name)
            tasks['language_learning_audit']=r
    tasks['language_credit_horizon']=None
    for path in sorted((RES/'diagnostics').glob('local_language_credit_horizon_*.json')):
        r=json.loads(path.read_text())
        if r.get('status')=='completed':
            for name,sha in r['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                    raise ValueError('Credit-horizon audit source changed: '+name)
            tasks['language_credit_horizon']=r
    tasks['split_screen']=[]
    for path in sorted((RES/'event_variants').glob('*_pilot.json')):
        r=json.loads(path.read_text())
        if (r.get('status')=='completed' and 'final' in r
                and r['args']['fit_targets']>=128 and r['args']['epochs']>=4):
            tasks['split_screen'].append(r)
    tasks['event_confirmation']=[]
    for path in sorted((ROOT/'experiments/gym/plans').glob('aws_native_confirmation_*/manifest.json')):
        plan=json.loads(path.read_text())
        if plan['status']=='superseded_unlaunched':continue
        for job in plan['jobs']:
            result=ROOT/job['result']
            if job['stage']!='pilot' or not result.exists():continue
            r=json.loads(result.read_text())
            if r.get('status')=='completed' and 'confirmation' in r.get('final',{}):
                tasks['event_confirmation'].append(dict(job=job,result=r))
    tasks['native_tabular']=[]
    for path in sorted((RES/'native_tabular').glob('*.json')):
        if path.name.endswith('.running.json'):continue
        r=json.loads(path.read_text())
        if r.get('status')=='completed' and 'final' in r and r['args']['fit']>=128 and r['args']['dev']>=128:
            tasks['native_tabular'].append(r)
    tasks['tabular_confirmation']=[]
    for path in sorted((RES/'tabular_confirmation').glob('*_pilot.json')):
        r=json.loads(path.read_text())
        if (r.get('status')=='completed' and r.get('protocol',{}).get('test_labels_scored')
                and r['protocol'].get('weights_frozen_before_test') and 'test' in r.get('final',{})):
            tasks['tabular_confirmation'].append(r)
    tasks['aws_hierarchy'] = []
    for meta_path in sorted((RES/'aws_20260929').glob('aws_e19_*/provenance.json')):
        meta=json.loads(meta_path.read_text())
        if meta.get('status')!='completed' or meta.get('script')!='experiments/e19_rhm.py':continue
        for path in sorted(meta_path.parent.glob('*.json')):
            if path.name=='provenance.json':continue
            row=json.loads(path.read_text())
            if 'test_acc' in row:tasks['aws_hierarchy'].append(dict(result=row,provenance=meta,path=str(path.relative_to(ROOT))))
    return tasks


def language_work_points(tasks, ev):
    """Completed quality/work pairs; preserve architecture and scoring split."""
    rows=[]
    for r in tasks['language_full_sparse']:
        a=r['args'];w=r['work']
        official=r['protocol']['official_test_read']
        location='AWS ' if a['tag'].startswith('aws_') else ''
        rows.append(dict(model=f"Ours: integrated d{a['payload']}/p{a['pool']}",family='integrated',
            label=location+f"I{a['payload']}/p{a['pool']}/{a['fit']//1024}K/s{a['seed']}",parameters=r['parameters'],
            fit=a['fit'],passes=a['epochs'],split='test' if official else 'dev',
            bpc=r['final']['official_test' if official else 'dev']['bpc'],
            total=w['total_training_unit_special_flops'],targets=w['fitting_targets'],
            inference=w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'],
            inference_method='Winner-only inference trace'))
    for r in tasks['episodic_language']:
        a=r['args'];w=r['work'];kv=a['memory']=='kv';semantic=a.get('candidate_index')=='semantic'
        parallel=a.get('heads',1)>1
        name=f"parallel race KV H{a['heads']}×d{a['payload']}" if parallel else ('episodic race KV' if kv else 'receiver memory')+f" d{a['payload']}"
        label=(f"IHR{a['heads']}x{a['payload']}" if parallel else f"{'IKVS' if semantic and kv else 'IKV' if kv else 'I'}{a['payload']}")
        if a.get('chunk',16)!=16:label+=f"/b{a['chunk']}"
        if 'write_credit' in a:
            name+=f" / delayed write credit {a['write_credit']:g}";label+=f"/wc{a['write_credit']:g}"
        if a.get('arrivals',1)>1:
            name+=f" / m{a['arrivals']} arrivals";label+=f"/m{a['arrivals']}"
        if 'update_targets' in a:label+=f"/u{a['update_targets']}@{a['lr']:g}"
        rows.append(dict(model='Ours: '+name,
            family='integrated',label=label+f"D{a['depth']}/{a['fit']//1024}K/s{a['seed']}",
            parameters=r['parameters'],fit=a['fit'],passes=a['epochs'],split='dev',
            bpc=r['final']['dev']['bpc'],total=w['cpu_emulator']['total_training_unit_special_flops'],
            targets=w['fitting_targets'],
            inference=w['cpu_emulator']['inference_arithmetic_flops_per_character']+w['cpu_emulator']['inference_special_functions_per_character'],
            projected_inference=w['projected_event_architecture']['inference_arithmetic_flops_per_character']+w['projected_event_architecture']['inference_special_functions_per_character'],
            inference_method='Selected-arrival inference trace'))
    for r in tasks.get('native_language',[]):
        a=r['args'];w=r['work'];cpu=w['cpu_emulator'];projected=w['projected_event_architecture']
        rows.append(dict(model=f"Ours: native event receivers H{a['heads']}×d{a['payload']}",family='integrated',
            label=f"Native H{a['heads']}d{a['payload']}/p{a['pool']}/{a['fit']//1024}K/s{a['seed']}",
            parameters=r['parameters'],fit=a['fit'],passes=a['epochs'],split='dev',bpc=r['final']['dev']['bpc'],
            total=cpu['total_training_unit_special_flops'],targets=w['fitting_targets'],
            inference=cpu['inference_arithmetic_flops_per_character']+cpu['inference_special_functions_per_character'],
            projected_inference=projected['inference_arithmetic_flops_per_character']+projected['inference_special_functions_per_character'],
            inference_method='Native receiver-state inference trace'))
    for r in tasks.get('delay_language',[]):
        a=r['args'];w=r['work'];cpu=w['cpu_emulator'];projected=w['projected_event_architecture']
        variant=f"R{a['clock_features']}/{a.get('clock_allocation','uniform')}/{'reception' if a['clock_readout'] else 'waiting'}"
        rows.append(dict(model='Ours: native temporal reception '+variant,family='integrated',
            label=variant+f"/{a['fit']//1024}K/s{a['seed']}",parameters=r['parameters'],fit=a['fit'],passes=a['epochs'],
            split='dev',bpc=r['final']['dev']['bpc'],total=cpu['total_training_unit_special_flops'],targets=w['fitting_targets'],
            inference=cpu['inference_arithmetic_flops_per_character']+cpu['inference_special_functions_per_character'],
            projected_inference=projected['inference_arithmetic_flops_per_character']+projected['inference_special_functions_per_character'],
            inference_method='Native content-gated clock-feature trace'))
    carrier=tasks['language_scaling']+tasks['language_selective'][:1]+tasks['language_scaleup']
    for r in carrier:
        a=r['args'];w=r['work']
        if a.get('official_test'):continue
        gate='g' if a.get('content_memory') else ''
        rows.append(dict(model=f"Ours: carrier w{a['width']}{gate}",family='carrier',
            label=f"C{a['width']}{gate}/{a['fit']//1024}K",parameters=r['parameters'],
            fit=a['fit'],passes=a['epochs'],split='dev',bpc=r['final']['dev']['bpc'],
            total=w['total_training_unit_special_flops'],targets=w['fitting_targets'],
            inference=w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'],
            inference_method='Saved forward operator trace'))
    estimate=runpy.run_path(str(ROOT/'experiments/lm_training_flops.py'))['estimate_training_flops']
    variants=[
        ('L256/1M', 'e64/lstm_D1000000_s256_p20_dr0.2_v.json'),
        ('L256/10M', 'e64/lstm_D10000000_s256_p1.json'),
        ('L512/10M', 'e64/lstm_D10000000_s512_p6_dr0.1_v.json'),
        ('T112x8/1M', 'e64/tf_D1000000_s112_L8_p5_b4_dr0_v.json'),
        ('T256x2/1M', 'e64/tf_D1000000_s256_p20_dr0.2_v.json'),
        ('T256x2/10M', 'e64/tf_D10000000_s256_p1.json'),
        ('T256x4/10M', 'e64/tf_D10000000_s256_L4_p4_dr0.1_v.json')]
    aws=ev['aws_references']['lstm']
    if aws:variants.append(('L512/90M',aws['path']))
    aws_tf=ev['aws_references']['tf']
    if aws_tf:variants.append(('T256x4/90M',aws_tf['path']))
    for label,path in variants:
        relative=str(Path(path).relative_to('experiments/results')) if path.startswith('experiments/results/') else path
        r=read(relative);a=r['args'];w=estimate(a,r['params'],r['steps'])
        family='lstm' if a['model']=='lstm' else 'tf'
        quality=ev['lstm10'] if label=='L512/10M' else ev['tf10'] if label=='T256x4/10M' else r['test_bpc']
        rows.append(dict(model=('LSTM' if family=='lstm' else 'Transformer')+f": {a['size']}"+
            (f"x{a.get('layers',2)}" if family=='tf' else ''),family=family,label=label,
            parameters=r['params'],fit=a['D'],passes=a['passes'],split='test',bpc=quality,
            total=w['total_training_flops'],targets=w['training_token_positions'],
            inference=w['forward_flops']/w['training_token_positions']*(2 if family=='tf' else 1),
            cached_inference=w['forward_flops']/w['training_token_positions'] if family=='tf' else None,
            inference_method='Overlapping-window shape estimate' if family=='tf' else 'Recurrent shape estimate'))
    # Keep later segment-batched native fits in Appendix B's common inventory.
    # Preserve older point IDs by appending; current native scores use T256
    # where completed, while the first v1 result retains its T128 score.
    native_paths=runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))['NATIVE']
    native_rows={r['label']:r for r in tasks['native_language_batched']['native']}
    for path,label in native_paths:
        if label not in native_rows:continue
        row=native_rows[label];r=read(path);a=r['args']
        segment=r.get('eval_segment') if row['test256'] is not None else a['segment']
        windows=len(range(0,a['test']-segment-1,segment//2))
        scored=segment+(windows-1)*(segment//2)
        recorded=r['test_targets_eval_segment'] if row['test256'] is not None else r['test_targets']
        if scored!=recorded:raise ValueError('Native window accounting changed: '+path)
        # Common inference graph charges evaluated warm positions per scored
        # target, using the emulator throughout. Winner-only traces stay in
        # the separate native appendix pending actual-trained parity/rescore.
        if row['infer'] is None:raise ValueError('Missing native emulator trace: '+path)
        rows.append(dict(model='Ours: batched native '+label,family='native_batched',
            label='NB '+label+f"/T{segment}",parameters=row['parameters'],fit=a['fit'],passes=a['passes'],
            split='test',bpc=row['test256'] if row['test256'] is not None else row['test'],
            total=row['whole'],targets=r['fitting_chars'],
            inference=row['infer']*windows*segment/scored,
            inference_method='Emulator trace × evaluated positions/scored targets',
            source=path,evaluation_segment=segment,evaluation_positions=windows*segment,scored_targets=scored))
    return rows


def episodic_pairs(tasks):
    grouped={}
    for r in tasks['episodic_language']:
        a=r['args']
        if a.get('heads',1)>1:continue  # Preserve the matched single-head intervention boundary.
        key=tuple(a[k] for k in ('fit','dev','epochs','chunk','payload','depth','pool','seed','lr'))
        group=grouped.setdefault(key,{'kv':{}})
        if a['memory']=='receiver':group['receiver']=r
        else:group['kv'][a.get('candidate_index','character')]=r
    pairs=[]
    for group in grouped.values():
        if 'receiver' in group:
            for right in group['kv'].values():
                left=group['receiver']
                for name in ('fitting_data_sha256','development_data_sha256'):
                    if left[name]!=right[name]:raise ValueError('Unmatched episodic data: '+name)
                pairs.append({'receiver':left,'kv':right})
    return sorted(pairs,key=lambda group:group['kv']['args']['fit'])


def count_reference_bpc(fit):
    """Best frozen Kneser-Ney reference on the shared 8,191-target development window (THEORY §376)."""
    rows=[row for path in sorted((RES/'count_reference').glob('*.json'))
          for row in json.loads(path.read_text()).get('rows',[])
          if row['fit']==fit and row.get('method')=='kn' and not row.get('adaptive')]
    return min((row['bpc'] for row in rows),default=None)


def count_scale_reference(fit, segment='e64_test_95M_1M'):
    """Best completed closed-form count model (any order/method) on a large-data segment (Theory §381)."""
    rows=[row for path in sorted((RES/'count_reference').glob('*scale*.json'))
          for row in json.loads(path.read_text()).get('rows',[])
          if row.get('fit')==fit and row.get('segment')==segment]
    return min(rows,key=lambda r:r['bpc'],default=None)


def aws_e64_reference(model, data_size):
    """Read completed AWS reference evidence, including its cost and source path."""
    for provenance_path in sorted((RES / "aws_20260929").glob("*/provenance.json")):
        try:
            meta = json.loads(provenance_path.read_text())
            args = meta.get("arguments", [])
            if (meta.get("status") != "completed" or
                    meta.get("script") != "experiments/e64_lm_baselines.py" or
                    "--model" not in args or args[args.index("--model") + 1] != model or
                    "--D" not in args or int(args[args.index("--D") + 1]) != data_size):
                continue
            for result_path in sorted(provenance_path.parent.glob("*.json")):
                if result_path.name == "provenance.json":
                    continue
                row = json.loads(result_path.read_text())
                value = row.get("test_bpc")
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value > 0):
                    return {"result": row, "path": str(result_path.relative_to(ROOT))}
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return None


def evidence(M):
    causal = read("e173/causal_language_10m_20260930.json")
    aws_references = {model: aws_e64_reference(model, 90_000_000) for model in ("lstm", "tf")}
    def aws_e64_bpc(model):
        reference = aws_references[model]
        return reference["result"]["test_bpc"] if reference else None
    return {"native10": causal["arms"]["without_word"]["test_bpc"],
            "native_word10": causal["arms"]["with_causal_word"]["test_bpc"],
            "lstm1": read("e64/lstm_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "lstm10": read("e174/aligned_lstm_10m_20260930.json")["test_bpc"],
            "lstm90": aws_e64_bpc("lstm"),
            "tf1": read("e64/tf_D1000000_s256_p20_dr0.2_v.json")["test_bpc"],
            "tf10": read("e174/aligned_tf_10m_20260930.json")["test_bpc"],
            "tf90": aws_e64_bpc("tf"),
            "aws_references": aws_references,
            "recall_tf": max(p["n32"] for f in (RES/"e61").glob("tf_K32_n8*.json")
                             for row in json.loads(f.read_text())["rows"] for p in row["curve"])}


def language_90m_reference_text(ev):
    """Publish each completed control without treating validation logs as test evidence."""
    scores = []
    pending = []
    for key, label in (("lstm90", "LSTM"), ("tf90", "four-layer Transformer")):
        if ev[key] is not None:
            scores.append(f"{ev[key]:.3f} for the {label}")
        else:
            pending.append(label)
    text = ("At 90M training characters, reference test scores are " + ", ".join(scores) + ". "
            if scores else "")
    costs = []
    for provenance in sorted((RES / 'aws_20260929').glob('aws_e64_*/provenance.json')):
        meta = json.loads(provenance.read_text())
        if meta.get('status') != 'completed':
            continue
        for path in provenance.parent.glob('*.json'):
            result = json.loads(path.read_text())
            if result.get('args', {}).get('D') != 90_000_000:
                continue
            estimate = result.get('training_flops_estimate', {})
            if estimate.get('total_training_flops'):
                costs.append(f"{result['args']['model'].upper()}: {estimate['total_training_flops'] / 1e15:.2f} PFLOP")
    if costs:
        text += ('Estimated training work (forward, backward, Adam and gradient clipping): '
                 + ', '.join(costs) + '. Shape-based estimates count multiply-add as two operations; '
                 'backward is approximated as twice forward. Validation/test inference is excluded. ')
    if pending:
        text += " and ".join(pending) + " reference results are pending. "
    text += "These are single-seed comparisons; capacities and fitting budgets are not matched. "
    scale = sorted((RES / 'count_reference').glob('*scale*.json'))
    if scale:
        rows = [row for path in scale for row in json.loads(path.read_text())['rows']
                if row['segment'] == 'e64_test_95M_1M' and row['method'] == 'mkn' and row['fit'] == 90_000_000
                and row['order'] == 7]
        if rows:
            best = rows[0]
            text += (f"Calibration (Theory §381): untuned modified Kneser–Ney counts of the same 90M characters score "
                     f"{best['bpc']:.3f} on the same test targets, so these controls sit near count level and are not "
                     "frontier bars. Target-leaked historical mixtures have been quarantined; "
                     "a corrected 90M mixture comparison remains open. ")
    return text


def accomplishments_figure(M, ev, tasks):
    import matplotlib.pyplot as plt
    import numpy as np
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    f, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    event = [100*next(point["test"] for point in row["curve"] if point["step"] == 2000)
             for row in tasks["mechanisms"]["order_curve"]["rows"]]
    reference = [100*row["acc"] for row in tasks["mechanisms"]["order_references"][2000]["rows"]]
    for i, (values, color) in enumerate(((event, blue), (reference, orange))):
        mean = np.mean(values)
        ax[0].bar(i, mean, width=.55, color=color)
        ax[0].errorbar(i, mean, yerr=[[mean-min(values)], [max(values)-mean]],
                       color="#172431", capsize=4, linewidth=1.2)
        ax[0].text(i, max(values)+3, f"{min(values):.2f}–{max(values):.2f}%",
                   ha="center", fontsize=9)
    ax[0].set(xticks=[0,1], xticklabels=["Ours: event chains\n5 runs; one pass",
                                      "Transformer\n2 runs; repeated fitting"],
              ylim=(0,116), ylabel="Held-out accuracy (%) ↑")
    ax[0].set_title("Learning an order rule\nSame 2,000 distinct fitting examples", fontsize=10)
    ax[1].bar(range(2), [100, 100*ev["recall_tf"]], color=[blue, orange], width=.55)
    ax[1].set_xticks([0, 1], ["Ours: race retrieval\n5 runs", "Best recorded\nTransformer result"])
    ax[1].set_ylim(0, 116)
    ax[1].set_ylabel("Accuracy at 4× context (%) ↑")
    for i, value in enumerate([100, 100*ev["recall_tf"]]):
        ax[1].text(i, value+2, f"{value:.1f}%", ha="center", fontsize=10)
    ax[1].set_title("Retrieval beyond training length\nFour times the training context", fontsize=10)
    for a in ax:
        a.grid(axis="x", visible=False)
        a.set_yticks([0,25,50,75,100])
        a.tick_params(axis="x", labelsize=7.3)
    f.tight_layout(w_pad=2.5)
    return f



def historical_write_groups(tasks):
    fields=('heads','payload','depth','pool','matching','recent','fit','dev','epochs',
            'chunk','update_targets','warmup_targets','seed','lr')
    groups={}
    for r in tasks['episodic_language']:
        if 'write_credit' in r['args'] and r['args']['dev']==8192 and r['args']['fit']>=2048:
            groups.setdefault(tuple(r['args'].get(k) for k in fields),[]).append(r)
    for key,trials in groups.items():
        matched=[r for r in tasks['episodic_language'] if 'write_credit' not in r['args']
                 and r['args'].get('arrivals',1)==1 and tuple(r['args'].get(k) for k in fields)==key
                 and r.get('fitting_data_sha256')==trials[0].get('fitting_data_sha256')
                 and r.get('development_data_sha256')==trials[0].get('development_data_sha256')]
        # Source-level coupled forward/gradient/Adam nesting is required by this driver.
        trials=sorted(trials,key=lambda r:r['args']['write_credit'])
        rows=matched[:1]+trials
        yield 'historical_write_quality_work_'+hashlib.sha256(repr(key).encode()).hexdigest()[:10], rows


def completed_split_evidence(tasks):
    prefix='aws_split_event_20261001T230029Z_'
    rows={r['args']['tag'][len(prefix):-len('_s6_pilot')]:r
          for r in tasks.get('split_screen',[]) if r['args']['tag'].startswith(prefix)
          and r['args']['tag'].endswith('_s6_pilot')}
    if len(rows)!=11:return {}
    for first,second in (('paired_timing_S4_private_P0_observed','paired_timing_S4_private_P0_rank'),
                         ('order_S16_shared_P0','order_S16_private_P0')):
        if rows[first]['data_sha256']!=rows[second]['data_sha256']:
            raise ValueError('Opening mechanism evidence requires identical fit/dev data')
    return rows


def figures(M, tasks, ev):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    blue, orange, gray = M["BLUE"], M["ORANGE"], M["GRAY"]
    FIG.mkdir(exist_ok=True)
    runpy.run_path(str(ROOT/'report/current_language_status.py'))['figure'](tasks['current_language_status'], FIG/'current_native_language_status.png')
    runpy.run_path(str(ROOT/'report/current_language_status.py'))['fitting_figure'](tasks['current_language_status'], FIG/'latest_native_language_fitting.png')
    def save(fig, name):
        if name != "accomplishments":
            fig.text(.01, 1.015, "Ours = Sleeping Machines", color=blue, fontsize=8,
                     fontweight="bold", ha="left")
        fig.savefig(FIG/(name+".png"), dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    for r in tasks.get('value_credit_pair',[]):
        f,a=plt.subplots(figsize=(7.2,2.45))
        labels={'native_full':'Native full','addressed_full':'Stored projection full',
            'late_full':'Late projection full','late_shallow':'Late same-width shallow','late_minimal':'Late minimal'}
        for row,color in zip(r['common_unit_ledger'],[blue,orange,'#1baf7a','#7057a3',gray]):
            a.plot([x['epoch'] for x in row['curve']],[x['cold_bpc'] for x in row['curve']],
                marker='o',color=color,label=labels[row['arm']],linewidth=1.5,markersize=3)
        a.set(xlabel='Fitting passes over1,024 characters',ylabel='Cold development bpc ↓',xticks=range(1,5),
              title='Matched integrated value-credit learning; seed6,2,047 dev targets')
        a.legend(fontsize=7,ncol=2);f.tight_layout()
        save(f,r['args']['tag'].replace('local_value_credit_analysis_','value_credit_learning_'))
    for key in ('balanced_joint','joint_outcome'):
        for r in tasks.get(key,[]):
            f,a=plt.subplots(figsize=(7.2,2.45))
            for row,color in zip(r['common_unit_ledger'],[blue,orange,'#1baf7a']):
                a.plot([x['epoch'] for x in row['curve']],[x['query_bits'] for x in row['curve']],
                    marker='o',markersize=2,linewidth=1.5,color=color,label=row['arm'].replace('_',' '))
            a.axhline(1,color=gray,linestyle='--',linewidth=1,label='Restricted query-count bound')
            a.set(xlabel='Fitting passes',ylabel='Development bits / query',title='Balanced distant dependency: '+key.replace('_',' '))
            a.legend(fontsize=7,ncol=2);f.tight_layout();save(f,r['args']['tag']+'_learning')
    for r in tasks.get('dvs_practical_native',[]):
        f,a=plt.subplots(figsize=(7.2,2.45))
        a.plot([x['epoch'] for x in r['curve']],[x['nll'] for x in r['curve']],
            marker='o',color=blue,label='Native p16/L2/H2',linewidth=1.5,markersize=3)
        by={x['arm']:x for x in r['common_unit_ledger']}
        for name,color,label in [('time_binned_naive_bayes',gray,'Calibrated time-aware counts'),
                                 (r['selected_control'],orange,'Selected nonlinear control')]:
            a.axhline(by[name]['development_nll'],color=color,linestyle='--',linewidth=1,label=label)
        a.set(xlabel='Fixed fitting passes over984 gestures',ylabel='Subject-disjoint development NLL',
            xticks=range(1,9),title='Real causal packets: completed learning versus strong controls')
        a.legend(fontsize=7);f.tight_layout();save(f,r['args']['tag']+'_learning')
    for r in tasks.get('dvs_credit_comparisons',[]):
        f,a=plt.subplots(figsize=(7.2,2.3))
        for row,color in zip(r['common_unit_ledger'],[blue,orange]):
            a.plot([x['epoch'] for x in row['curve']],[x['nll'] for x in row['curve']],
                marker='o',color=color,label=row['arm'].replace('_',' '),linewidth=1.5,markersize=3)
        a.set(xlabel='Fixed passes over256 fitting gestures',ylabel='Development NLL',xticks=range(1,5),
            title='Matched sparse native credit screen: '+r['args']['kind'].replace('_',' ')+f" / seed{r['args'].get('seed',6)}")
        a.legend(fontsize=7);f.tight_layout();save(f,r['args']['tag']+'_learning')
    for r in tasks.get('dvs_evolution_offset_comparisons',[]):
        f,a=plt.subplots(figsize=(7.2,2.3))
        for row,color in zip(r['common_unit_ledger'],[blue,orange,'#34856d']):
            a.plot([x['epoch'] for x in row['curve']],[x['nll'] for x in row['curve']],marker='o',
                color=color,label=row['arm'].replace('_',' '),linewidth=1.5,markersize=3)
        a.set(xlabel='Fixed passes over256 fitting gestures',ylabel='Development NLL',xticks=range(1,5),
            title=f"Physical-time phase offsets: completed seed{r['args']['seed']}")
        a.legend(fontsize=7);f.tight_layout();save(f,r['args']['tag']+'_learning')
    for r in tasks.get('dvs_state_partition_probes',[]):
        parent=read(str((ROOT/r['args']['state_probe']).relative_to(RES)))
        reference=next(x for x in parent['rows'] if x['encoder']=='selected')
        by={x['partition']:x for x in r['rows'] if x['encoder']=='selected'}
        labels=['Query C1','Query C.1','+ payloads','+ layer0','+ layer1','+ all state']
        values=[reference['query_probe_development']['nll'],by['query_with_state_setting']['development']['nll'],
            by['payloads']['development']['nll'],by['layer0']['development']['nll'],
            by['layer1']['development']['nll'],reference['augmented_probe_development']['nll']]
        f,a=plt.subplots(figsize=(7.2,2.35))
        a.bar(range(len(values)),values,color=[orange,gray,blue,blue,blue,blue])
        for i,v in enumerate(values):a.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=7)
        a.set(xticks=range(len(values)),xticklabels=labels,ylabel='Development NLL',ylim=(0,1.18),
            title='Frozen-state diagnosis: regularization explains most of the apparent gain')
        f.tight_layout();save(f,r['args']['tag']+'_learning')
    banknote=[r for r in tasks.get('native_tabular',[]) if r['args']['dataset']=='banknote'
              and r['args']['tag'].startswith('aws_fast_matrix_recovery_20261001T213409Z_')
              and r['args']['clock_features']==0]
    if banknote:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.35))
        names=['Ours: native' if r['args']['model']=='ours' else 'Boosted trees' for r in banknote]
        colors=[blue if r['args']['model']=='ours' else orange for r in banknote]
        for axis,metric,scale,label in zip(axes,('accuracy','nll'),(100,1),('Development accuracy (%) ↑','Development log loss ↓')):
            values=[r['final']['dev'][metric]*scale for r in banknote]
            axis.bar(range(len(values)),values,color=colors,width=.6)
            axis.set_xticks(range(len(values)),names,fontsize=8)
            axis.set_ylabel(label);axis.grid(axis='x',visible=False)
            axis.set_ylim(0,max(values)*1.2)
            for j,value in enumerate(values):axis.text(j,value+max(values)*.025,f'{value:.2f}',ha='center',fontsize=9)
        f.tight_layout();save(f,'banknote_first_screen')
    confirmation={family:[r for r in tasks.get('tabular_confirmation',[])
        if r['args']['model']==family and r['args']['tag'].startswith('aws_banknote_confirmation_20261001T234000Z_')]
        for family in ('ours','trees','catboost','logistic')}
    complete=[family for family,rows in confirmation.items() if sorted(r['args']['seed'] for r in rows)==[6,7,8]]
    if 'ours' in complete and 'trees' in complete:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.35))
        labels={'ours':'Ours: native','trees':'Boosted trees','catboost':'CatBoost','logistic':'Logistic'}
        for ax,metric,scale,title in zip(axes,('accuracy','nll'),(100,1),('Reserved-test accuracy (%) ↑','Reserved-test log loss ↓')):
            vals=[[r['final']['test'][metric]*scale for r in confirmation[k]] for k in complete]
            means=[sum(v)/len(v) for v in vals]
            ax.bar(range(len(means)),means,color=[blue if k=='ours' else orange for k in complete],width=.6)
            for j,v in enumerate(vals):
                ax.plot([j]*len(v),v,'o',color='#172431',markersize=3)
                ax.text(j,max(v)+max(means)*.03,f'{means[j]:.3f}' if metric=='nll' else f'{means[j]:.1f}%',ha='center',fontsize=8)
            ax.set_xticks(range(len(complete)),[labels[k] for k in complete],fontsize=8)
            ax.set_ylim(0,max(max(v) for v in vals)*1.19);ax.set_ylabel(title)
        f.tight_layout();save(f,'banknote_reserved_test')
    credited=tasks.get('state_credit',[])
    if credited:
        f,ax=plt.subplots(figsize=(7.2,2.5))
        for r in credited:
            x=r['work']['total_training_unit_special_flops']/1e9;y=100*r['final']['dev']['accuracy']
            label='Ours: baseline' if not r['args']['state_credit'] else 'Ours: state-write credit'
            ax.scatter([x],[y],s=55,color=blue if r['args']['state_credit'] else orange,label=label)
            ax.annotate(f'{label}\n{y:.2f}%, {x:.3f}GF', (x,y),xytext=(0,15 if r['args']['state_credit'] else -27),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xlabel('Whole fitting work (GFLOPs) ↓');ax.set_ylabel('Development accuracy (%) ↑')
        ax.set_ylim(0,110);ax.margins(x=.4);f.tight_layout();save(f,'state_credit_quality_work')
    learning=tasks.get('language_learning_audit')
    if learning:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        for row,color in zip(learning['frozen_models'],(orange,blue)):
            label='Ours: carrier (131K fit)' if '/parallel_language/' in row['result'] else 'Ours: native (8K fit)'
            histories=sorted(int(k) for k in row['history_interventions'])
            evidence=row['history_interventions']
            axes[0].plot(histories,[evidence[str(k)]['bpc'] for k in histories],'o-',color=color,label=label)
            axes[1].plot(histories,[evidence[str(k)]['mean_kl_from_64_character_history'] for k in histories],'o-',color=color)
        for axis in axes:
            axis.set_xscale('log',base=2);axis.set_xticks([1,2,4,8,16,64],[1,2,4,8,16,64])
            axis.set_xlabel('Retained history (characters)')
        axes[0].set_ylabel('Diagnostic slice bpc ↓');axes[1].set_ylabel('KL from64-character history (nats)')
        axes[0].legend(fontsize=7);f.tight_layout();save(f,'language_learning_context')
    split=completed_split_evidence(tasks)
    if split:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.75))
        values=[100*split['paired_timing_S4_private_P0_'+mode]['final']['dev']['accuracy']
                for mode in ('observed','rank')]
        axes[0].bar([0,1],values,color=[blue,orange],width=.6)
        axes[0].set_xticks([0,1],['Ours: elapsed time','Ours: order only'],fontsize=8)
        axes[0].set_title('Timing makes the answer identifiable',fontsize=9)
        for j,v in enumerate(values):axes[0].text(j,v+2,f'{v:.1f}%',ha='center',fontsize=9)
        for j,mode in enumerate(('private','shared')):
            values=[100*split[f'order_S{s}_{mode}_P0']['final']['dev']['accuracy'] for s in (4,16)]
            positions=np.arange(2)+(j-.5)*.36
            axes[1].bar(positions,values,width=.34,color=orange if j==0 else blue,label='Ours: '+mode+' rules')
            for x,v in zip(positions,values):axes[1].text(x,v+2,f'{v:.1f}',ha='center',fontsize=8)
        axes[1].set_xticks([0,1],['4 occupied sources','16 occupied sources'],fontsize=8)
        axes[1].set_title('Shared rules; private memories',fontsize=9)
        axes[1].legend(loc='upper left',fontsize=7,frameon=False)
        for axis in axes:
            axis.set_ylim(0,115);axis.set_yticks([0,25,50,75,100]);axis.set_ylabel('Development accuracy (%) ↑',fontsize=8)
            axis.grid(axis='x',visible=False)
        f.tight_layout();save(f,'native_mechanism_evidence')
    for domain in ('temporal','language'):
        records=[r for r in tasks.get('gym_screen',[]) if r['job']['domain']==domain]
        if records:
            f,axes=plt.subplots(1,2,figsize=(7.2,2.6))
            for entry in records:
                r=entry['result'];w=r['work'];name='Ours '+entry['job']['variant'].replace(domain+'_','').replace('_',' ')
                quality=100*r['final']['dev']['accuracy'] if domain=='temporal' else r['final']['dev']['bpc']
                cost=w['total_training_unit_special_flops'] if domain=='temporal' else w['cpu_emulator']['total_training_unit_special_flops']
                infer=(w['inference_arithmetic_flops_per_query']+w['inference_special_functions_per_query']) if domain=='temporal' else (w['cpu_emulator']['inference_arithmetic_flops_per_character']+w['cpu_emulator']['inference_special_functions_per_character'])
                for axis,x in zip(axes,(cost/1e9,infer/1e6)):
                    axis.scatter(x,quality,color=blue,s=30);axis.annotate(name,(x,quality),xytext=(3,3),textcoords='offset points',fontsize=6)
            axes[0].set_xlabel('Whole neural fitting GFLOPs');axes[1].set_xlabel('Inference MFLOPs / target')
            for axis in axes:axis.set_ylabel('Development accuracy (%)' if domain=='temporal' else 'Development bpc');axis.grid(alpha=.2)
            f.tight_layout();save(f,'aws_fast_screen_'+domain)
    hierarchy=[r for r in tasks.get('aws_hierarchy',[]) if r['result']['config']['train']==64000]
    if hierarchy:
        f,a=plt.subplots(figsize=(7.2,2.6))
        for residual in sorted({r['result']['config']['residual'] for r in hierarchy}):
            rows=sorted([r['result'] for r in hierarchy if r['result']['config']['residual']==residual],key=lambda r:r['config']['depth'])
            a.plot([r['config']['depth'] for r in rows],[100*r['test_acc'] for r in rows],marker='o',
                label='Ours plain race' if residual==0 else f'Ours residual-{residual}',color=blue if residual==0 else orange)
        a.set(xlabel='Race model depth (RHM hierarchy depth fixed at 3)',ylabel='Final held-out accuracy (%)',xticks=[1,2,3,4])
        a.grid(alpha=.2);a.legend(fontsize=8);f.tight_layout();save(f,'aws_hierarchy_depth')
    if tasks.get('native_language'):
        f,axes=plt.subplots(1,2,figsize=(7.2,2.6))
        for r in tasks['native_language']:
            a=r['args'];label=f"Ours d{a['payload']}/p{a['pool']}/{a['fit']//1024}K/s{a['seed']}"
            axes[0].plot([v['epoch'] for v in r['curve']],[v['dev']['bpc'] for v in r['curve']],marker='o',label=label)
            axes[1].scatter(r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9,r['final']['dev']['bpc'],label=label)
        axes[0].set(xlabel='Fitting passes',ylabel='Frozen development bpc ↓');axes[0].legend(fontsize=6)
        axes[1].set(xlabel='Whole fitting GFLOPs ↓',ylabel='Frozen development bpc ↓')
        for ax in axes:ax.grid(alpha=.2)
        f.tight_layout();save(f,'native_language_quality_work')
    # Analytical construction, not a fitted task score.
    f,axes=plt.subplots(1,3,figsize=(7.2,2.1))
    t=np.linspace(0,1,200)
    axes[0].plot(t,np.cos(np.pi*t),label='First content direction')
    axes[0].plot(t,np.sin(np.pi*t),label='Second content direction')
    axes[0].set(xlabel='Local normalized elapsed time',ylabel='Receptive coefficient',title='Time rotates relevance')
    axes[0].legend(fontsize=5.5)
    age=np.linspace(-.1,1.1,200);u=np.clip(age,0,1)
    axes[1].plot(age,u**2*(1-u)**2*np.exp(-.4*u))
    axes[1].set(xlabel='Age / learned window width',ylabel='Contribution weight',title='Smooth event integration')
    theta,beta,prefix,split=.4,.7,2.3,.25
    first=-np.log1p(-beta*theta/prefix)/beta
    voltage=prefix/beta*(-np.expm1(-beta*(split-first)))
    for y,current in ((1,2.),(0,.8)):
        second=split+np.log((current-beta*voltage)/(current-beta*theta))/beta
        times=[first]+([second] if second<.45 else [])
        axes[2].vlines(times,y-.2,y+.2,color=blue if y else orange,lw=2)
    axes[2].axvline(split,color=gray,ls=':',lw=1)
    axes[2].axvline(.45,color=gray,ls='--',lw=1)
    axes[2].set(xlim=(0,.48),ylim=(-.5,1.5),yticks=[0,1],yticklabels=['Later weak input','Later strong input'],xlabel='Local event time',title='Same first spike; new evidence')
    axes[2].tick_params(axis='y',labelsize=5.5)
    f.tight_layout();save(f,'temporal_reception_windows_trains')
    if tasks.get('delay_language'):
        f,axes=plt.subplots(1,2,figsize=(7.2,2.5))
        records=tasks.get('native_language',[])+tasks['delay_language']
        for r in records:
            a=r['args'];label=('Ours native' if 'clock_features' not in a else f"Ours R{a['clock_features']} {a.get('clock_allocation','uniform')} {'reception' if a['clock_readout'] else 'waiting'}")+f"/{a['fit']//1024}K/s{a['seed']}"
            axes[0].scatter(r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9,r['final']['dev']['bpc'],label=label)
            axes[1].plot([v['epoch'] for v in r['curve']],[v['dev']['bpc'] for v in r['curve']],label=label)
        axes[0].set(xlabel='Whole fitting GFLOPs ↓',ylabel='Frozen development bpc ↓')
        axes[1].set(xlabel='Passes over fitting data',ylabel='Frozen development bpc ↓');axes[1].legend(fontsize=5)
        f.tight_layout();save(f,'delay_language_quality_work')
    if tasks.get('native_event'):
        f,axes=plt.subplots(1,2,figsize=(7.2,2.6))
        for ax,task in zip(axes,('order','timing')):
            for r in tasks['native_event']:
                a=r['args']
                if a['task']!=task:continue
                label=f"Ours S{a['sources']}/{'CF' if a['credit']=='counterfactual' else 'PW'}/{'time' if a['time_input']=='observed' else 'rank'}/s{a['seed']}"
                quality=r['final']['dev']['accuracy']
                lo,hi=r['final']['dev']['episode_bootstrap_95_percent_interval']
                ax.errorbar(r['work']['total_training_unit_special_flops']/1e9,quality,
                            yerr=[[max(0,quality-lo)],[max(0,hi-quality)]],fmt='o',capsize=2,label=label)
            ax.set(xlabel='Whole fitting GFLOPs ↓',ylabel='Frozen development accuracy ↑',title=task)
            ax.grid(alpha=.2)
            if ax.containers:ax.legend(fontsize=5.5)
            else:ax.text(.5,.5,'No completed '+task+' pilot',transform=ax.transAxes,ha='center',fontsize=8)
        f.tight_layout();save(f,'native_event_quality_work')
    for name,rows in historical_write_groups(tasks):
        f,axes=plt.subplots(1,2,figsize=(7.2,2.4))
        labels=['Ours parent' if 'write_credit' not in r['args'] else f"Ours α={r['args']['write_credit']:g}" for r in rows]
        colors=[gray]+[blue,orange,blue][:max(0,len(rows)-1)]
        for i,r in enumerate(rows):
            w=r['work']['cpu_emulator'];bpc=r['final']['dev']['bpc']
            axes[0].scatter(w['total_training_unit_special_flops']/1e9,bpc,color=colors[i])
            axes[0].annotate(labels[i],(w['total_training_unit_special_flops']/1e9,bpc),xytext=(3,5),textcoords='offset points',fontsize=7)
        for i,r in enumerate(rows):
            axes[1].plot([v['epoch'] for v in r['curve']],[v['dev']['bpc'] for v in r['curve']],marker='o',label=labels[i],color=colors[i])
        axes[0].set(xlabel='Whole fitting GFLOPs ↓',ylabel='Frozen development bpc ↓')
        axes[1].set(xlabel='Passes over fitting data',ylabel='Frozen development bpc ↓')
        axes[1].legend(fontsize=7)
        for ax in axes:ax.grid(alpha=.2)
        f.tight_layout();save(f,name)
    f,ax=plt.subplots(figsize=(7.2,2.3));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    def box(x,y,w,h,t):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#edf3fb',edgecolor=blue))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
    box(.1,.8,1.55,1.3,'Incoming content\n+ persistent state')
    box(2.1,1.85,2.35,.9,'Head 1: own Q / K / V\nReceiver + historical race')
    box(2.1,.25,2.35,.9,'Head 2: own Q / K / V\nReceiver + historical race')
    box(4.95,1.85,2,.9,'Channel 1 / arrival t₁\nEvolves while waiting')
    box(4.95,.25,2,.9,'Channel 2 / arrival t₂\nEvolves while waiting')
    box(7.5,.8,2.2,1.3,'Read at max(t₁,t₂)\nKeep separate channels\nLearned next-block mix')
    for start,end in [((1.7,1.8),(2,2.3)),((1.7,1.1),(2,.7)),((4.5,2.3),(4.9,2.3)),((4.5,.7),(4.9,.7)),((7,2.3),(7.45,1.8)),((7,.7),(7.45,1.1))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=11,color=blue))
    f.tight_layout();save(f,'parallel_temporal_heads')
    f,ax=plt.subplots(figsize=(7.2,2.4));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    box(.1,1.5,2.2,1.,'Historical write\nSave key, value and φ')
    box(3.2,1.5,2.4,1.,'Later temporal query\nSame cached race / winner')
    box(6.6,1.5,3.1,1.,'Prediction + local teacher\nCurrent Q and live state learn')
    box(3.2,.12,2.4,.95,'Old admitted keys\nSum weighted saved features')
    box(6.6,.12,3.1,.95,'Old write maps learn\nOne key outer product\nOne winning-value outer product')
    for start,end in [((2.4,2),(3.1,2)),((5.7,2),(6.5,2)),((7.4,1.45),(5.7,.6)),((5.7,.55),(6.5,.55))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=11,color=blue))
    ax.text(.1,.45,'Ours: +50% K/V tensor storage\nNo old representation graph',fontsize=8,color=gray)
    f.tight_layout();save(f,'historical_write_credit_flow')
    pilots=[r for r in tasks['episodic_language'] if r['args'].get('heads',1)>1 and r['args']['fit']==2048 and r['args']['dev']==8192 and 'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and r['args'].get('chunk',16)==16]
    if pilots:
        f,ax=plt.subplots(figsize=(7.2,2.5))
        for row in pilots:
            a=row['args'];ax.plot([c['epoch'] for c in row['curve']],[c['dev']['bpc'] for c in row['curve']],marker='o',label=f"Ours H{a['heads']} / U{a.get('update_targets',16)} / lr {a['lr']:g}")
        ax.set(xlabel='Passes over 2,048 fitting characters',ylabel='Frozen 8K development bpc ↓');ax.grid(alpha=.2);ax.legend(fontsize=7)
        f.tight_layout();save(f,'parallel_temporal_head_pilots')
    scaled_heads=[r for r in tasks['episodic_language'] if r['args'].get('heads',1)>1 and
        'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and r['args'].get('update_targets')==128 and
        r['args']['lr']==.004 and r['args']['seed']==6 and r['args']['dev']==8192 and
        r['args']['fit'] in (2048,8192)]
    if any(r['args']['fit']==8192 for r in scaled_heads):
        f,ax=plt.subplots(figsize=(7.2,2.8))
        for heads,color in ((2,blue),(4,orange)):
            rows=sorted([r for r in scaled_heads if r['args']['heads']==heads],key=lambda r:r['args']['fit'])
            if rows:
                ax.plot([r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9 for r in rows],
                    [r['final']['dev']['bpc'] for r in rows],marker='o',color=color,label=f'Ours H{heads} / d32 per head')
                for r in rows:
                    ax.annotate(f"{r['args']['fit']//1024}K fit",(r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9,r['final']['dev']['bpc']),xytext=(5,5),textcoords='offset points',fontsize=8)
        controls=[r for r in tasks['episodic_language'] if r['args'].get('heads',1)==1 and r['args']['fit']==8192 and r['args']['dev']==8192]
        for r in controls:
            name='indexed KV' if r['args']['memory']=='kv' else 'receiver'
            ax.scatter([r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9],[r['final']['dev']['bpc']],marker='s',color='#278477')
            ax.annotate('Ours earlier '+name,(r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9,r['final']['dev']['bpc']),xytext=(7,-10 if name=='receiver' else 4),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Whole fitting GFLOPs (unit-weight specials)',ylabel='Frozen 8K development bpc ↓')
        ax.grid(alpha=.2);ax.legend(fontsize=8);f.tight_layout();save(f,'parallel_head_data_work')
    f,ax=plt.subplots(figsize=(7.2,2.15));ax.set(xlim=(0,10),ylim=(0,3.7));ax.axis('off')
    def modality_box(x,y,w,h,t):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor='#edf3fb',edgecolor=blue))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
    for y,t in [(2.8,'Language / instructions'),(1.65,'Irregular sensor events'),(.5,'Action queries / deadlines')]:
        modality_box(.1,y,2.3,.65,t)
        ax.add_patch(FancyArrowPatch((2.5,y+.325),(3.2,1.95),arrowstyle='->',mutation_scale=11,color=blue))
    modality_box(3.3,.6,3.5,2.7,'Learned input adapters\nContent + time + source\n\nPersistent evolving state\nTrainable delays / parallel races\nCounterfactual route credit\nSelectively recruited modules')
    for y,t in [(2.8,'Grounded predictions'),(1.65,'Temporal world estimates'),(.5,'Timed actions')]:
        modality_box(7.5,y,2.3,.65,t)
        ax.add_patch(FancyArrowPatch((6.9,1.95),(7.4,y+.325),arrowstyle='->',mutation_scale=11,color=blue))
    f.tight_layout();save(f,'general_temporal_interface')
    repeat_rows=[r for r in tasks['episodic_language'] if r['args'].get('arrivals',1)>1]
    repeat_fields=('heads','payload','depth','pool','matching','recent','fit','dev','epochs','chunk','update_targets','warmup_targets','seed','lr')
    for group in sorted({tuple(r['args'][k] for k in repeat_fields) for r in repeat_rows}):
        trials=[r for r in repeat_rows if tuple(r['args'][k] for k in repeat_fields)==group]
        a=trials[0]['args']
        reference=[r for r in tasks['episodic_language'] if 'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and all(r['args'].get(k)==a.get(k) for k in repeat_fields)]
        if not reference:continue
        rows=sorted(reference+trials,key=lambda r:r['args'].get('arrivals',1));base=reference[0]
        def total(r):return r['work']['cpu_emulator']['total_training_unit_special_flops']
        def inference(r):
            w=r['work']['cpu_emulator'];return w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character']
        marks=[r['args'].get('arrivals',1) for r in rows]
        f,axes=plt.subplots(1,2,figsize=(7.2,2.5))
        axes[0].plot(marks,[total(r)/total(base) for r in rows],marker='o',color=blue,label='Ours: whole fitting')
        axes[0].plot(marks,[inference(r)/inference(base) for r in rows],marker='s',color=orange,label='Ours: inference trace')
        axes[0].set(ylabel='Counted work / one-arrival reference',xlabel='Winner values / historical query',xticks=marks)
        axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
        axes[1].plot(marks,[r['final']['dev']['bpc'] for r in rows],marker='o',color=blue)
        axes[1].set(ylabel='Frozen development bpc ↓',xlabel='Winner values / historical query',xticks=marks);axes[1].grid(alpha=.2)
        for m,r in zip(marks,rows):axes[1].annotate(f"{r['final']['dev']['bpc']:.3f}",(m,r['final']['dev']['bpc']),xytext=(2,5),textcoords='offset points',fontsize=8)
        axes[1].margins(y=.25);f.tight_layout()
        save(f,'repeated_arrival_quality_work_'+hashlib.sha256(repr(group).encode()).hexdigest()[:10])
    optimizer_pilots=[r for r in tasks['episodic_language'] if r['args'].get('heads')==2 and r['args']['fit']==2048 and r['args']['dev']==8192 and 'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and r['args'].get('chunk',16)==16]
    if optimizer_pilots:
        optimizer_pilots.sort(key=lambda r:(r['args'].get('update_targets',16),r['args']['lr']))
        f,ax=plt.subplots(figsize=(7.2,2.8));bottom=np.zeros(len(optimizer_pilots))
        colors=['#89b9e7',blue,'#86c6bd','#aa9fdb',orange,gray]
        names=[('forward_and_loss','Forward/loss'),('backward','Backward'),('gradient_normalization','Gradient averaging'),('gradient_clipping','Clipping'),('optimizer','Adam'),('special','Unit-weight special functions')]
        for (stage,label),color in zip(names,colors):
            costs=[(r['work']['cpu_emulator']['training_special_function_evaluations'] if stage=='special' else r['work']['cpu_emulator']['training_stages'].get(stage,0))/1e9 for r in optimizer_pilots]
            ax.bar(range(len(costs)),costs,bottom=bottom,label=label,color=color);bottom+=costs
        ax.set(xticks=range(len(optimizer_pilots)),xticklabels=[f"Ours U{r['args'].get('update_targets',16)}\nlr {r['args']['lr']:g}" for r in optimizer_pilots],ylabel='Whole fitting GFLOPs (unit-weight specials)',ylim=(0,max(bottom)*1.17))
        for i,(r,total) in enumerate(zip(optimizer_pilots,bottom)):
            ax.text(i,total+.25,f"{r['final']['dev']['bpc']:.3f} bpc",ha='center',fontsize=8)
        ax.legend(fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.25));f.tight_layout();save(f,'parallel_optimizer_work')
    # Full-bank architectural comparison: keep every query/key match.
    f, axes = plt.subplots(2, 2, figsize=(7.2, 4.3))
    contexts = np.array([64,128,256,512,1024,2048,4096,8192,16384,32768])
    rows = [full_bank_comparison(context=int(n)) for n in contexts]
    for name,color,label in [('transformer',gray,'Transformer'),('ours',blue,'Ours: temporal race')]:
        axes[0,0].loglog(contexts,[r[name+'_inference']/1e6 for r in rows],color=color,label=label)
        axes[1,0].loglog(contexts,[r[name+'_training']/1e6 for r in rows],color=color,label=label)
    axes[0,0].set(xlabel='All historical keys scored / query',ylabel='Inference MFLOPs / token')
    axes[1,0].set(xlabel='All historical keys scored / query',ylabel='Training MFLOPs / target')
    axes[0,1].semilogx(contexts,[r['transformer_inference']/r['ours_inference'] for r in rows],color=blue,label='Inference')
    axes[0,1].semilogx(contexts,[r['transformer_training']/r['ours_training'] for r in rows],color=orange,label='Training incl. credit / Adam')
    axes[0,1].set(xlabel='All historical keys scored / query',ylabel='Transformer / ours work ratio')
    depths=np.array([2,4,8,16,32,64])
    for name,color,label in [('transformer',gray,'Transformer'),('ours',blue,'Ours: temporal race')]:
        axes[1,1].plot(depths,[full_bank_comparison(depth=int(l))[name+'_inference']/1e6 for l in depths],color=color,label=label)
    axes[1,1].set(xlabel='Depth (N = 4,096)',ylabel='Inference MFLOPs / token')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    f.tight_layout();save(f,'full_bank_temporal_scaling')
    f,axes=plt.subplots(1,2,figsize=(7.2,2.2))
    axes[0].loglog(contexts,[r['transformer_value_bytes']/1024 for r in rows],color=gray,label='Transformer: all values')
    axes[0].loglog(contexts,[r['ours_value_bytes']/1024 for r in rows],color=blue,label='Ours: one value / head')
    axes[0].set(xlabel='Historical keys / query',ylabel='Logical value reads (KiB / token)')
    axes[1].semilogx(contexts,[r['transformer_key_value_bytes']/r['ours_key_value_bytes'] for r in rows],color=blue)
    axes[1].set(xlabel='Historical keys / query',ylabel='Total K/V read ratio: Transformer / ours',ylim=(0,2.2))
    axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    f.tight_layout();save(f,'full_bank_temporal_traffic')
    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    for depth, color in ((1, gray), (8, blue)):
        row = tasks["generic_language"][depth]
        curve = [row["initial"]["dev"]["bpc"]] + [e["dev"]["bpc"] for e in row["curve"]]
        axes[0].plot(range(len(curve)), curve, "o-", color=color, label=f"Ours: {depth} layer" + ("s" if depth > 1 else ""))
        work = tasks["generic_language_audit"]["rows"][str(depth)]["wall_s"]
        final = curve[-1]
        axes[1].scatter(work, final, color=color, s=65)
        axes[1].annotate(f"{depth} layer" + ("s" if depth > 1 else "") + f"\n{final:.3f} bpc", (work,final),
                         xytext=(0,12), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set(xlabel="Passes over 8,192 training characters", ylabel="Validation bpc (lower is better)",
                title="Learned prediction through event layers", xticks=range(5))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Total CPU wall time (s; fitting + evaluation)",
                ylabel="Validation bpc (lower is better)", title="Quality / observed time tradeoff",
                ylim=(3.32,3.58))
    f.tight_layout()
    save(f, "e133_generic_language")
    stream = tasks["stream_training"]
    f, a = plt.subplots(figsize=(7.2, 2.65))
    a.plot(range(5), [stream["initial"]["bpc"]]+[row["dev"]["bpc"] for row in stream["curve"]],
           "o-", color=blue, label="Ours: validation")
    a.plot(range(1,5), [row["fit"]["bpc"] for row in stream["curve"]],
           "s--", color=gray, label="Ours: fitting, frozen evaluation")
    token = tasks["token_training"]
    a.plot(range(5), [token["initial"]["bpc"]]+[row["dev"]["bpc"] for row in token["curve"]],
           "^-", color=orange, label="Ours: prefix tokens, validation")
    a.set(xlabel="Passes over 8,192 training characters", ylabel="Bits per character ↓",
          title="Eight-layer persistent language stream", xticks=range(5))
    a.legend(fontsize=8)
    f.tight_layout()
    save(f, "e176_stream_language_learning")
    f = accomplishments_figure(M, ev, tasks)
    save(f, "accomplishments")

    f, a = plt.subplots(figsize=(7.2, 2.65))
    a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis("off")
    boxes = [(0.1, 1.55, 1.8, .9, "Input events\nvector + time"),
             (2.45, 1.55, 2.4, .9, "Mix input + memory\nGated residual vector"),
             (5.5, 1.55, 1.8, .9, "Clock race\nwinning arrival"),
             (7.95, 1.55, 1.9, .9, "Next layer\nthen a query"),
             (2.45, .05, 2.4, .85, "Alternatives + loss\nvector + clock credit")]
    for x, y, w, h, label in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",facecolor="#edf3fb",edgecolor=blue))
        a.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    for x1, x2 in ((1.95, 2.4), (4.9, 5.45), (7.35, 7.9)):
        a.add_patch(FancyArrowPatch((x1,2),(x2,2),arrowstyle="-|>",mutation_scale=14,color=blue))
    a.annotate("", (3.65,.94), (3.65,1.5), arrowprops={"arrowstyle":"->","linestyle":"--","color":orange})
    a.text(5.65,.47,"One message is emitted per retained arrival.\nLocal state survives between events.",fontsize=9,va="center")
    a.text(.1,3.3,"Ours: a learned event-state block",fontsize=11,fontweight="bold")
    a.text(.1,2.85,"Source identity, content and elapsed time determine the next vector and arrival.",fontsize=9)
    save(f,"shared_architecture")

    f,a=plt.subplots(figsize=(7.2,2.65))
    a.set(xlim=(0,10),ylim=(0,4));a.axis('off')
    boxes=[(.1,1.3,1.7,1.,'Observed event\naddress, time, content'),
           (2.25,1.3,1.9,1.,'Addressed memory\nevolves on read'),
           (4.6,2.05,2.1,.75,'Head 1: local race\nselected content'),
           (4.6,.75,2.1,.75,'Head 2: local race\nselected content'),
           (7.25,1.3,2.5,1.,'Align and mix channels\nnext block / query')]
    for x,y,w,h,label in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor='#edf3fb',edgecolor=blue))
        a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    for start,end in (((1.85,1.8),(2.2,1.8)),((4.2,1.8),(4.55,2.4)),
                      ((4.2,1.8),(4.55,1.1)),((6.75,2.4),(7.2,1.8)),((6.75,1.1),(7.2,1.8))):
        a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,color=blue))
    a.text(.1,3.5,'Ours: native temporal receiver heads',fontsize=11,fontweight='bold')
    a.text(.1,3.,'Eight blocks; two independent heads; local state persists between events.',fontsize=9)
    a.text(.1,.15,'Learning charges losing-route teachers and producer credit; no per-position KV bank.',fontsize=8,color=orange)
    save(f,'native_addressed_event_path')

    f, a = plt.subplots(figsize=(7.2,2.85))
    values=[100*tasks['shd_full_values']['rows']['parent']['held_accuracy'],
            100*tasks['shd_key_value']['rows']['separate']['held_accuracy'],
            100*tasks['shd_full_values']['rows']['fixed']['held_accuracy']]
    labels=["Starting\ncheckpoint","New context maps\n6,336 taught parameters","All value layers\n58,048 taught parameters"]
    if tasks['shd_content'] is not None:
        values.append(100*tasks['shd_content']['rows']['content']['held_accuracy'])
        labels.append("Content retrieval\n60,096 taught parameters")
    a.bar(range(len(values)),values,color=["#c5ced8",gray,orange,blue][:len(values)],width=.55)
    a.set(xticks=range(len(values)),xticklabels=labels,
          ylim=(0,100),ylabel="Held-speaker accuracy (%) — higher is better",title="Eight-layer event classifier: same parent, one-pass continuations")
    a.tick_params(axis='x',labelsize=8)
    for i,v in enumerate(values):a.text(i,v+2,f"{v:.1f}%",ha="center",fontsize=11)
    f.tight_layout();save(f,"e122_speech")

    f, axes = plt.subplots(1,2,figsize=(7.2,2.85))
    def held(row):
        parts=[row["final"][k] for k in ("dev_original","dev_additional")]
        return 100*sum(p["correct"] for p in parts)/sum(p["n"] for p in parts)
    parent=held(tasks["shd_scaled"])
    entries=[("Original\nparent",tasks["shd_scaled"],gray)]
    for key,label,color in (("shd_warm","Warm\ncontinuation",orange),
                            ("shd_fine","Fine source\nmessages",blue),
                            ("shd_phase","Fine source +\ntemporal phase","#1baf7a"),
                            ("shd_local","New learner;\nparent frozen","#8766b6")):
        if tasks[key] is not None:entries.append((label,tasks[key],color))
    axes[0].bar(range(len(entries)),[held(r) for _,r,_ in entries],
                color=[c for _,_,c in entries],width=.55)
    for i,(_,r,_) in enumerate(entries):
        axes[0].text(i,held(r)+2,f"{held(r):.2f}%",ha="center",fontsize=8.5)
    axes[0].set(xticks=range(len(entries)),xticklabels=[n for n,_,_ in entries],
        ylim=(0,100),ylabel="Private held-speaker accuracy (%)",title="Recognition quality — higher is better")
    timed=entries[1:]
    for i,(label,r,color) in enumerate(timed):
        axes[1].scatter(r["wall_s"],held(r),s=65,color=color,label=label.replace("\n"," "))
    axes[1].axhline(parent,color=gray,linestyle="--",linewidth=1)
    axes[1].text(.98,.96,f"Original parent: {parent:.2f}%",transform=axes[1].transAxes,
        fontsize=7,ha="right",va="top")
    axes[1].set(xlabel="Recorded continuation wall time (s)",ylabel="Accuracy (%) — higher is better",
        title="One pass; same 6,144 fitting examples",ylim=(65,80))
    axes[1].legend(fontsize=6.1,loc="lower center",ncol=2)
    axes[0].tick_params(axis="x",labelsize=6.1)
    f.tight_layout();save(f,"e139_source_information")

    if tasks["shd_state_residual"] is not None:
        row=tasks["shd_state_residual"]
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        epochs=[0]+[r["epoch"] for r in row["curve"]]
        dev=[row["initial"]["dev"]["accuracy"]]+[r["dev"]["accuracy"] for r in row["curve"]]
        axes[0].plot(epochs,[100*x for x in dev],"o-",color=blue,label="Temporal residual + frozen parent")
        axes[0].axhline(100*dev[0],color=gray,linestyle="--",label="Original parent")
        axes[0].set(xlabel="Passes over 6,144 fitting utterances",ylabel="Private accuracy (%) — higher is better",
            title="Transfer to held speakers",xticks=epochs,ylim=(65,100))
        axes[0].legend(fontsize=6.5,loc="upper left")
        for split,label,color in (("fit","Fitting speakers",orange),("dev","Held speakers",blue)):
            axes[1].plot([r["epoch"] for r in row["curve"]],[r[split]["nll"] for r in row["curve"]],
                "o-",label=label,color=color)
        axes[1].axhline(row["initial"]["dev"]["nll"],color=gray,linestyle="--",label="Parent held NLL")
        axes[1].set(xlabel="Passes over fitting utterances",ylabel="NLL — lower is better",
            title="Prediction quality",xticks=epochs[1:])
        axes[1].legend(fontsize=6.5,frameon=True,facecolor="white",edgecolor="white",framealpha=1.)
        f.tight_layout();save(f,"e143_temporal_residual_learning")

    if tasks["shd_state_ablation"] is not None and tasks["shd_state_summary"] is not None:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        rows=tasks["shd_state_ablation"]["rows"]
        entries=[("trained","Learned model"),("reset_clocks","Initial hidden clocks"),
            ("reset_source_embedding","Initial source marks"),("reset_state_stack","Initial state stack"),
            ("reset_modal_dynamics","Initial modal poles")]
        values=[100*rows[key]["held"]["accuracy"] for key,_ in entries]
        axes[0].barh(range(len(entries)),values,color=[blue,orange,gray,gray,gray])
        axes[0].set(yticks=range(len(entries)),yticklabels=[name for _,name in entries],
            xlim=(60,87),xlabel="Accuracy (%) — higher is better",title="Reset one learned subsystem")
        axes[0].invert_yaxis();axes[0].tick_params(axis="y",labelsize=7)
        for i,v in enumerate(values):axes[0].text(v+.4,i,f"{v:.1f}%",va="center",fontsize=7)
        summary=tasks["shd_state_summary"]
        audit=read("e147/disjoint_speaker_audit_20260930.json")
        for i,(n,gain,lost) in enumerate(((512,summary["new_only_correct"],summary["parent_only_correct"]),
            (audit["n"],audit["new_only_correct"],audit["parent_only_correct"]))):
            axes[1].bar(i-.16,gain,width=.3,color=blue,label="Newly correct" if i==0 else None)
            axes[1].bar(i+.16,-lost,width=.3,color=orange,label="Previously correct, now wrong" if i==0 else None)
            axes[1].text(i,gain+3,f"Net +{gain-lost}",ha="center",fontsize=8)
        axes[1].axhline(0,color=gray,linewidth=.8)
        axes[1].set(xticks=[0,1],xticklabels=["Development\n512 utterances",f"Disjoint audit\n{audit['n']} utterances"],
            ylabel="Changed correct answers",title="Matched gains and regressions",ylim=(-25,82))
        axes[1].tick_params(axis="x",labelsize=7)
        axes[1].legend(fontsize=6,loc="upper left")
        f.tight_layout();save(f,"e145_learned_timing_and_transfer")

    if tasks["shd_single_clean"] is not None and tasks["shd_single_paired"] is not None:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        for key,label,color in (("shd_single_clean","Clean head initialization",orange),
                               ("shd_single_paired","Paired-view initialization",blue)):
            row=tasks[key]
            curve=[r for r in row["curve"] if r["epoch"]>0]
            points=[row["conditioned_initial"]["dev"]]+[r["dev"] for r in curve]
            axes[0].plot([0]+[r["epoch"] for r in curve],[100*p["accuracy"] for p in points],
                "o-",label=label,color=color)
        reference=max(tasks["shd_state_residual"]["curve"],key=lambda r:r["dev"]["correct"])
        axes[0].axhline(100*reference["dev"]["accuracy"],color=gray,linestyle="--",label="Combined model")
        axes[0].set(xlabel="Single-encoder continuation passes",ylabel="Private accuracy (%) — higher is better",
            title="Head initialization and encoder updates",xticks=[0,1,2,3],ylim=(60,100))
        axes[0].legend(fontsize=5.6,loc="upper left",ncol=2)
        row=tasks["shd_single_paired"]
        for i,(key,label) in enumerate((("matched_clean_head","Clean-fit head"),("conditioned_initial","Paired-view head"))):
            part=row[key]
            clean=part.get("clean_fit",part.get("fit"))["nll"]
            aug=part["augmented_fit"]["nll"]
            axes[1].bar(i-.16,clean,width=.3,color=blue,label="Clean fitting speech" if i==0 else None)
            axes[1].bar(i+.16,aug,width=.3,color=orange,label="Augmented fitting speech" if i==0 else None)
        axes[1].set(xticks=[0,1],xticklabels=["Clean-fit head","Paired-view head"],
            ylabel="NLL — lower is better",title="Same frozen temporal features")
        axes[1].legend(fontsize=6.1,loc="upper right")
        f.tight_layout();save(f,"e152_single_encoder_learning")

    if tasks["shd_single_audit"] is not None and tasks["shd_observer_audit"] is not None:
        audit=tasks["shd_single_audit"];directional=tasks["shd_observer_audit"]
        f,axes=plt.subplots(1,2,figsize=(7.2,2.45))
        points=[(audit['rows'][name],label,color) for name,label,color in (
            ('combined','Combined model',gray),('single_paired','Single paired head',blue),
            ('matched_d6','Trained six blocks','#1baf7a'),('grown_d12','Bounded twelve blocks',orange))]
        points.append((directional['rows']['directional_d12'],'Directional twelve blocks','#9154c3'))
        points.append((directional['rows']['directional_prefix'],'Selected trained prefix','#144da1'))
        for row,label,color in points:
            axes[0].scatter(row['forward_wall_s'],100*row['audit']['accuracy'],color=color,s=30,label=label)
        axes[0].set(xlabel='CPU forward seconds / 657 utterances — lower is better',
            ylabel='Reused audit accuracy (%) — higher is better',title='Quality and observed CPU work',ylim=(72,85))
        axes[0].legend(fontsize=5.8,loc='upper right')
        for i,(full,prefix) in enumerate(((audit['rows']['grown_d12']['audit'],audit['rows']['grown_d12_prefix']['audit']),
            (directional['rows']['directional_d12']['audit'],directional['rows']['directional_prefix']['audit']))):
            for offset,row,color,label in ((-.16,full,blue,'Full twelve blocks'),(.16,prefix,gray,'Six appended blocks removed')):
                axes[1].bar(i+offset,100*row['accuracy'],width=.3,color=color,label=label if i==0 else None)
                axes[1].text(i+offset,100*row['accuracy']+.4,str(row['correct']),ha='center',fontsize=7)
        axes[1].set(xticks=[0,1],xticklabels=['Bounded outputs','Directional units'],
            ylabel='Reused audit accuracy (%) — higher is better',title='Fitted contribution of added depth',ylim=(70,86))
        axes[1].legend(fontsize=5.8,loc='upper right')
        f.tight_layout();save(f,'e164_depth_use_and_work')

    if tasks["shd_exchange"] is not None:
        f, axes = plt.subplots(1,2,figsize=(7.2,2.95))
        for name,label,color,style in (("state","Full state query",blue,"-"),
                                       ("packets","Packet query",gray,":")):
            row=tasks["shd_exchange"][name]
            for axis,split in zip(axes,("fit","held")):
                points=[row["initial"][split]["accuracy"]]+[x[split]["accuracy"] for x in row["curve"]]
                axis.plot(range(len(points)),[100*x for x in points],marker="o",linestyle=style,color=color,label=label)
        if tasks["shd_compact"] is not None:
            for mode,label,color,style in (("learned","Compact, learned angles","#1baf7a","-"),
                                          ("frozen","Compact, fixed angles",orange,"--")):
                row=tasks["shd_compact"]["rows"][mode]
                for axis,split in zip(axes,("fit","held")):
                    axis.plot([x["epoch"] for x in row["curve"]],
                              [100*x[split+"_accuracy"] for x in row["curve"]],
                              marker="o",linestyle=style,color=color,label=label)
        for axis,title in zip(axes,("Learning the fitting utterances","Transfer to held speakers")):
            axis.set(xlabel="Passes over 1,024 utterances",ylabel="Accuracy (%) — higher is better",
                     title=title,xticks=range(4),ylim=(0,105))
        axes[0].legend(fontsize=6.8,loc="lower right")
        f.tight_layout();save(f,"e136_scattering_learning")

    f, axes = plt.subplots(1,2,figsize=(7.2,3.15))
    entries=tasks["work_audit"]["rows"]
    names={"shared_phase_only":"Ours: phase rule (69 scalars)","shared_d2_phase":"Ours: encoder + phase rule",
           "shared_d2_recall":"Ours: encoder + pointer","lstm":"LSTM, width 32 / 2 layers",
           "transformer":"Transformer, width 32 / 2 layers"}
    colors={"shared_phase_only":blue,"shared_d2_phase":"#1baf7a","shared_d2_recall":blue,
            "lstm":gray,"transformer":orange}
    for axis,task,split,title in zip(axes,("modular","recall"),("unseen_all","context4x"),
                                    ("Arithmetic: every unseen triple","Recall: four times the context")):
        for row in entries:
            if row["task"]!=task or row["split"]!=split:continue
            label=names[row["model"]];x=row["work"]["estimated_operations"];y=100*row["accuracy"]
            axis.scatter([x],[y],s=55,marker="D" if row["model"].startswith("shared") else
                         "s" if row["model"]=="transformer" else "o",color=colors[row["model"]],label=label,zorder=4)
            offset=(-12,10) if task=="modular" and row["model"]=="lstm" else \
                   (12,10) if task=="modular" and row["model"]=="transformer" else (0,6 if y<90 else -14)
            axis.annotate(f"{y:.1f}%",(x,y),xytext=offset,textcoords="offset points",ha="center",fontsize=8)
        axis.set(xscale="log",ylim=(-3,112),xlabel="Estimated operations per query (log)",title=title)
        if task=="recall":
            from matplotlib.ticker import NullFormatter
            axis.set_xticks([700_000,1_000_000,2_000_000,4_000_000],["0.7M","1M","2M","4M"])
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.legend(fontsize=6.4,loc="center left",bbox_to_anchor=(-.02,.6))
    axes[0].set_ylabel("Development accuracy (%)")
    f.tight_layout();save(f,"consolidated_work_frontiers")
    mechanism = tasks["mechanisms"]
    f, axes = plt.subplots(1,2,figsize=(7.2,3.05))
    groups = [("Timing\npatterns",mechanism["timing"]),
              ("Shared-motif\ncomposition",mechanism["motifs"]),
              ("Depth-four\norder",mechanism["deep_order"])]
    for i,(_,row) in enumerate(groups):
        values = [100*r["final"]["test"] for r in row["rows"]]
        axes[0].scatter([i]*len(values),values,color=blue,s=28,zorder=4)
        axes[0].annotate(f"{min(values):.2f}–{max(values):.2f}%",(i,min(values)),
                         xytext=(0,-15),textcoords="offset points",ha="center",fontsize=7.1)
    axes[0].set(xticks=range(3),xticklabels=[n for n,_ in groups],ylim=(98.5,100.35),
                ylabel="Synthetic evaluation accuracy (%) ↑",title="Five runs per event mechanism")
    axes[0].tick_params(axis='x',labelsize=7)
    curves=mechanism["order_curve"]["rows"]
    steps=[r["step"] for r in curves[0]["curve"]]
    quality=np.array([[100*r["test"] for r in row["curve"]] for row in curves])
    axes[1].fill_between(steps,quality.min(0),quality.max(0),color=blue,alpha=.18)
    axes[1].plot(steps,np.median(quality,0),"o-",color=blue,ms=3,label="Ours: event chain, one pass")
    for n,row in mechanism["order_references"].items():
        axes[1].scatter([n]*len(row["rows"]),[100*r["acc"] for r in row["rows"]],
                        color=orange,s=23,marker="s",label="Transformer; repeated fitting" if n==2000 else None)
    axes[1].set(xscale="log",ylim=(25,103),xlabel="Distinct fitting examples (log)",
                ylabel="Synthetic evaluation accuracy (%) ↑",title="Depth-three sample efficiency")
    axes[1].legend(fontsize=6.4,loc="lower right")
    f.tight_layout(w_pad=2)
    save(f,"supremacy_map")

    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    stages = ("forward_and_loss", "backward", "gradient_clipping", "optimizer")
    names = ("Forward + loss", "Backward", "Clip", "Adam")
    colors = (blue, orange, gray, "#8c73aa")
    rows = tasks["training_work"]["rows"]
    for a, model, title in zip(axes, ("common", "transformer"), ("Ours: event query encoder, 8 layers", "Transformer: 2 layers")):
        bottom = np.zeros(len(rows))
        for stage, label, color in zip(stages, names, colors):
            key = "common_stages" if model == "common" else "reference_stages"
            values = np.array([row[key][stage]/1e9 for row in rows])
            a.barh(range(len(rows)), values, left=bottom, label=label, color=color)
            bottom += values
        if model == "common":
            values = np.array([row["common_setup"]["total_arithmetic_flops"]/1e9 for row in rows])
            a.barh(range(len(rows)), values, left=bottom, label="Evidence + calibration", color="#64a68c")
        a.set_yticks(range(len(rows)), ["Text", "Market", "Composition", "MNIST", "Gestures"])
        a.set_xlim(0, max(max(row["common_total_training_flops"], row["reference_total_training_flops"])
                        for row in rows)/1e9*1.1)
        a.invert_yaxis()
        a.set_xlabel("Whole fitting budget (estimated GFLOPs) ↓", fontsize=8)
        a.set_title(title, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    f.legend(handles, labels, fontsize=6.5, loc="lower center", ncol=3)
    f.tight_layout(w_pad=2, rect=(0, .15, 1, 1))
    save(f, "e172_complete_training_work")

    rows = sorted(tasks["language_scaling"], key=lambda row:row["parameters"])
    f, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for row, color in zip(rows, (gray, orange, blue, "#64a68c")):
        axes[0].plot([point["epoch"] for point in row["curve"]],
                     [point["dev"]["bpc"] for point in row["curve"]], "o-", color=color,
                     label=f"Ours: width {row['args']['width']}")
        total = row["work"]["total_training_arithmetic_flops"]/1e9
        score = row["final"]["dev"]["bpc"]
        axes[1].scatter(total, score, color=color, s=50)
        axes[1].annotate(f"Ours: {row['parameters']/1000:.1f}K parameters",(total,score),
            xytext=(0,10),textcoords="offset points",ha="center",fontsize=7)
    axes[0].set(xlabel="Passes over the same 131,072 characters",ylabel="Development bpc ↓",
                title="Completed language learning curves",xticks=[1,2,3,4])
    axes[0].legend(fontsize=7)
    axes[1].set(xscale="log",xlabel="Total fitting arithmetic (GFLOPs; log) ↓",
                ylabel="Development bpc ↓",title="Capacity costs work and improves quality")
    axes[1].margins(x=.25,y=.3)
    f.tight_layout(w_pad=2)
    save(f,"language_capacity_scaling")

    capacity=[row for row in tasks['language_scaleup']
              if row['args']['fit']==131072 and row['args']['width']==256]
    if capacity and tasks['language_selective']:
        constant=next(row for row in tasks['language_memory'] if row['args']['memory_profile']=='inherited')
        selective=next(row for row in tasks['language_selective'] if row['args']['memory_profile']=='inherited')
        f,a=plt.subplots(figsize=(7.2,2.65))
        points=[constant,selective,capacity[-1]]
        labels=['Ours: constant memory, width 128','Ours: input gates, width 128','Ours: input gates, width 256']
        for row,label,color,offset in zip(points,labels,(gray,blue,orange),((16,13),(16,-16),(-16,12))):
            x=row['work']['total_training_arithmetic_flops']/1e12;y=row['final']['dev']['bpc']
            a.scatter(x,y,color=color,s=55)
            a.annotate(label+f"\n{y:.3f} bpc; {x:.3f} TFLOPs",(x,y),xytext=offset,
                textcoords='offset points',ha='right' if offset[0]<0 else 'left',fontsize=8,
                arrowprops=dict(arrowstyle='-',color=color))
        a.set(xlim=(.75,4.5),ylim=(2.53,2.68),xlabel='Total fitting arithmetic (TFLOPs) ↓',
              ylabel='Development bpc ↓',title='Where additional work helped this fit')
        f.tight_layout();save(f,'language_compute_choices')

    rows=tasks["breadth_work"]["rows"]
    costs={row["task"]:row for row in tasks["training_work"]["rows"]}
    f, axes=plt.subplots(1,2,figsize=(7.2,2.2))
    selected=[next(row for row in rows if row["task"]==task)
              for task in ("language","market","temporal","mnist","dvs")]
    ratios=[row["common_forward_map_scan_flops"]/row["reference_forward_map_attention_flops"]
            for row in selected]
    training=[costs[row["task"]]["common_to_reference_ratio"] for row in selected]
    for axis,values,title in zip(axes,(ratios,training),
        ("Ours / Transformer: inference core FLOPs","Ours / Transformer: total fitting FLOPs")):
        axis.barh(range(5),values,color=blue,height=.65)
        axis.axvline(1,color=orange,linestyle="--",linewidth=1)
        axis.set_yticks(range(5),["Text","Market","Composition","MNIST","Gestures"])
        axis.invert_yaxis();axis.set_xlim(0,2.45)
        axis.set_title(title,fontsize=9)
        axis.set_xlabel("Ratio ↓  (Transformer = 1)",fontsize=8)
        for i,value in enumerate(values):axis.text(value+.04,i,f"{value:.2f}×",va="center",fontsize=8)
    f.tight_layout(w_pad=2)
    save(f,"breadth_work_ratios")

    f,axes=plt.subplots(1,2,figsize=(7.2,2.65))
    a=axes[0];a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
    boxes=[(.08,.84,'Observed character + previous message'),
           (.08,.58,'Contextual keys race through delays'),
           (.08,.32,'One unit mixes content + retained state'),
           (.08,.06,'Winner sends a vector and arrival time')]
    for x,y,label in boxes:
        a.add_patch(FancyBboxPatch((x,y),.83,.12,boxstyle='round,pad=.01',
                                 facecolor='#eef3f7',edgecolor=blue,linewidth=.8))
        a.text(x+.415,y+.06,label,ha='center',va='center',fontsize=7.2)
    for y in (.82,.56,.30):
        a.annotate('',xy=(.50,y-.08),xytext=(.50,y),arrowprops=dict(arrowstyle='->',color=blue))
    a.set_title('Ours: integrated event path',fontsize=9)
    a=axes[1]
    for depth in range(6):
        a.scatter([depth+(j%3-1)*.15 for j in range(54)],
                  [j//3 for j in range(54)],s=6,color='#cdd4db',linewidths=0)
        a.scatter([depth,depth+.15],[3,3],s=15,color=orange,linewidths=0)
        a.scatter([depth+.15*(depth%2)],[3],s=20,color=blue,linewidths=0)
    a.plot([i+.15*(i%2) for i in range(6)],[3]*6,color=blue,linewidth=.8)
    a.set(xlim=(-.5,5.5),ylim=(21,-2),xticks=range(6),
          xticklabels=[str(i) for i in range(1,7)],yticks=[],xlabel='Depth')
    a.set_title('324 available units; 6 state updates / character',fontsize=8.5)
    a.grid(False)
    a.scatter([],[],s=15,color=blue,label='Winner value + state update')
    a.scatter([],[],s=15,color=orange,label='Other addressed key')
    a.scatter([],[],s=10,color='#cdd4db',label='Unaddressed units remain dormant')
    a.legend(loc='lower center',fontsize=6.5,frameon=False)
    f.tight_layout(w_pad=1.6);save(f,'full_sparse_language_path')

    f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
    for a,title,labels,color in (
        (axes[0],'Transformer: attention over token entries',[
            'Current representation produces query',
            'Score separate keys for past positions',
            'Softmax-weighted sum of their values',
            'KV storage grows with retained positions'],orange),
        (axes[1],'Ours: race over persistent memory units',[
            'Incoming content produces query at each depth',
            'Score two state-dependent addressed keys',
            'First arrival selects one content-bearing unit',
            'Mix / update fixed-size state; emit message'],blue)):
        a.set(xlim=(0,1),ylim=(0,1));a.axis('off');a.set_title(title,fontsize=8.4)
        for y,label in zip((.81,.57,.33,.09),labels):
            a.add_patch(FancyBboxPatch((.02,y),.96,.13,boxstyle='round,pad=.01',
                facecolor='#eef3f7',edgecolor=color,linewidth=.8))
            a.text(.5,y+.065,label,ha='center',va='center',fontsize=7)
        for y in (.79,.55,.31):
            a.annotate('',xy=(.5,y-.07),xytext=(.5,y),
                arrowprops=dict(arrowstyle='->',color=color))
    f.tight_layout(w_pad=1.5);save(f,'language_queries_and_memory')

    full=tasks['language_full_sparse']
    if full:
        latest=max(full,key=lambda row:(row['args']['fit'],-row['final']['dev']['bpc']))
        neural={r['model']:r for r in tasks['training_work']['language_rows']}
        ours=latest['work']
        whole=[ours['total_training_unit_special_flops']/1e9]
        fitting=[ours['total_training_unit_special_flops']/ours['fitting_targets']/1e6]
        forward=[(ours['inference_arithmetic_flops_per_character']+
                  ours['inference_special_functions_per_character'])/1e6]
        for name in ('lstm','tf'):
            r=neural[name]
            whole.append(r['total_training_flops']/1e9)
            fitting.append(r['total_training_flops']/r['training_token_positions']/1e6)
            forward.append(r['forward_flops']/r['training_token_positions']/1e6)
        labels=[f"Ours: integrated d{latest['args']['payload']}\n{latest['args']['fit']:,} fit / {latest['args']['epochs']} passes",
                'LSTM\n10M fit / 6 passes','Transformer\n10M fit / 4 passes']
        f,axes=plt.subplots(1,3,figsize=(7.2,2.65),sharey=True)
        for i,(a,values,title,unit,limits) in enumerate(zip(axes,(whole,fitting,forward),
            ('Whole fitting run','Fitting per target','Forward per position'),
            ('GFLOPs; log scale ↓','MFLOPs / target; log scale ↓','MFLOPs / position; log scale ↓'),
            ((10,3e7),(.05,80),(.015,50)))):
            a.barh(range(3),values,color=[blue,gray,orange],height=.65)
            a.set_yticks(range(3),labels);a.set_xscale('log')
            a.tick_params(axis='y',labelleft=i==0,labelsize=7.2)
            a.set_xlim(*limits);a.set_xlabel(unit,fontsize=7.2)
            a.set_title(title,fontsize=9)
            for j,v in enumerate(values):
                label=f'{v:,.1f}' if i==0 else f'{v:.3f}'
                a.text(v*1.13,j,label,va='center',fontsize=7.5)
        axes[0].invert_yaxis()
        f.suptitle('Work estimates; data, model size and quality differ',fontsize=9,y=1.01)
        f.tight_layout(w_pad=.9);save(f,'integrated_language_work_progress')

    points=language_work_points(tasks,ev)
    f,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
    styles={'integrated':(blue,'s','Ours: integrated'),
            'native_batched':('#7655b2','D','Ours: later native'),
            'carrier':('#1baf7a','o','Ours: earlier carrier'),
            'lstm':(gray,'o','LSTM'),'tf':(orange,'^','Transformer')}
    for a,split,title in zip(axes,('dev','test'),('Cold development scores','Saved test scores')):
        subset=[r for r in points if r['split']==split]
        for family,(color,marker,label) in styles.items():
            group=[r for r in subset if r['family']==family]
            if group:a.scatter([r['total']/1e9 for r in group],[r['bpc'] for r in group],
                color=color,marker=marker,s=25,label=label,zorder=3)
        for i,r in enumerate(subset):
            offset=(5,7 if i%2==0 else -12)
            # Separate the matched 2K depth/KV points; their work/quality are
            # close enough that alternating offsets alone overlap the IDs.
            if r['family']=='integrated' and r['fit']==2048:
                if r['label'].startswith('IKVS'):offset=(35,8)
                elif r['label'].startswith('IKV') and 'D8/' in r['label']:offset=(35,-8)
                elif r['label'].startswith('IKV'):offset=(15,12)
                elif 'D8/' in r['label']:offset=(-6,-18)
                else:offset=(-10,12)
            a.annotate(str(points.index(r)+1),(r['total']/1e9,r['bpc']),xytext=offset,
                textcoords='offset points',fontsize=7,
                arrowprops=dict(arrowstyle='-',color='#8a8984',lw=.4))
        a.set_xscale('log');a.set_xlim(.8,2e7);a.set_ylim(1.45,max(3.75,max(p['bpc'] for p in points)+.2))
        a.set_title(title,fontsize=9);a.set_xlabel('Whole fitting GFLOPs; log scale ↓',fontsize=8)
        a.legend(loc='upper right',fontsize=6.3,frameon=False)
    axes[0].set_ylabel('Bits per character ↓',fontsize=8)
    f.suptitle('Quality versus fitting work; scoring protocols and data budgets differ',fontsize=9,y=1.01)
    f.tight_layout(w_pad=.8);save(f,'language_quality_vs_work')

    f,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
    for a,split,title in zip(axes,('dev','test'),('Cold development scores','Saved test scores')):
        subset=[r for r in points if r['split']==split]
        for family,(color,marker,label) in styles.items():
            group=[r for r in subset if r['family']==family]
            if group:a.scatter([r['inference']/1e6 for r in group],[r['bpc'] for r in group],
                color=color,marker=marker,s=25,label=label,zorder=3)
        for i,r in enumerate(subset):
            offset=(5,7 if i%2==0 else -12)
            if r['family']=='integrated' and r['fit']!=2048:
                offset=(-13,8) if r['fit']>8192 else (6,9)
            if r['family']=='integrated' and r['fit']==2048:
                if r['label'].startswith('IKVS'):offset=(35,8)
                elif r['label'].startswith('IKV') and 'D8/' in r['label']:offset=(35,-8)
                elif r['label'].startswith('IKV'):offset=(12,12)
                elif 'D8/' in r['label']:offset=(-12,-18)
                else:offset=(-10,12)
            a.annotate(str(points.index(r)+1),(r['inference']/1e6,r['bpc']),xytext=offset,
                textcoords='offset points',fontsize=7,
                arrowprops=dict(arrowstyle='-',color='#8a8984',lw=.4))
            if r['family']=='tf':
                left=r['cached_inference']/1e6;right=r['inference']/1e6
                a.plot([left,right],[r['bpc'],r['bpc']],color=orange,lw=.7,alpha=.5,zorder=1)
                a.scatter([left],[r['bpc']],facecolors='none',edgecolors=orange,marker='^',s=25,zorder=2)
        if split=='test':a.scatter([],[],facecolors='none',edgecolors=orange,marker='^',
            s=25,label='Transformer: cache scenario')
        a.set_xscale('log');a.set_xlim(.015,30)
        a.set_ylim(1.45,max(3.75,max(p['bpc'] for p in points)+.2))
        a.set_title(title,fontsize=9);a.set_xlabel('Inference MFLOPs / predicted character; log scale ↓',fontsize=7.5)
        a.legend(loc='upper right',fontsize=6.1,frameon=False)
    axes[0].set_ylabel('Bits per character ↓',fontsize=8)
    f.suptitle('Quality versus inference work; estimates, scoring protocols and data differ',fontsize=8.8,y=1.01)
    f.tight_layout(w_pad=.8);save(f,'language_quality_vs_inference')

    if tasks['integrated_online_language']:
        row=tasks['integrated_online_language'][-1]
        f,axes=plt.subplots(1,2,figsize=(7.2,2.6))
        for name,color,label in [('frozen',gray,'Ours: frozen'),('online',blue,'Ours: online')]:
            n,total,x,y=0,0.,[],[]
            for block in row['blocks']:
                n+=block['n'];total+=block['bpc'][name]*block['n'];x.append(n);y.append(total/n)
            axes[0].plot(x,y,color=color,label=label,lw=1.2)
        axes[0].set(xlabel='New stream targets',ylabel='Cumulative bpc ↓',title='Prediction before feedback')
        axes[0].legend(fontsize=7,frameon=False)
        costs=[row['work'][name]['unit_special_flops']/1e6 for name in ('frozen','online')]
        axes[1].barh(range(2),costs,color=[gray,blue],height=.6)
        axes[1].set_yticks(range(2),['Ours: frozen','Ours: online']);axes[1].invert_yaxis()
        axes[1].set(xlabel='Whole stream MFLOPs ↓',title='Scoring plus adaptation work')
        axes[1].set_xlim(0,max(costs)*1.3)
        for i,cost in enumerate(costs):axes[1].text(cost+max(costs)*.02,i,f'{cost:,.1f}',va='center',fontsize=8)
        f.tight_layout(w_pad=1.2);save(f,'integrated_online_language')

    pairs=episodic_pairs(tasks)
    for pair in pairs:
        f,axes=plt.subplots(1,2,figsize=(7.2,2.7))
        for mode,color,label in [('receiver',gray,'Ours: receiver memory'),('kv',blue,'Ours: episodic race KV')]:
            r=pair[mode]
            axes[0].plot([0]+[row['epoch'] for row in r['curve']],
                [r['initial_dev']['bpc']]+[row['dev']['bpc'] for row in r['curve']],
                'o-',color=color,label=label,ms=3)
        axes[0].set(xlabel='Fitting passes',ylabel='Frozen development bpc ↓',title='Matched small-data intervention')
        axes[0].legend(fontsize=6.4,loc='upper right')
        a=pair['kv']['final']['dev']['activity'];mean=a['kv_scores']/max(1,a['kv_queries'])
        axes[1].bar([0,1],[mean,1],color=[orange,blue],width=.6)
        axes[1].set(xticks=[0,1],xticklabels=['Dense aggregation\n(same shortlist)','Ours: one\nrace winner'],
            ylabel='Value vectors read / retrieval query ↓',title='KV value reads at inference')
        for i,value in enumerate([mean,1]):axes[1].text(i,value+.15,f'{value:.2f}',ha='center',fontsize=8)
        axes[1].set_ylim(0,mean*1.2);axes[1].tick_params(axis='x',labelsize=7)
        a=pair['kv']['args']
        index=a.get('candidate_index','character')
        f.tight_layout(w_pad=1.1);save(f,f"episodic_language_comparison_D{a['fit']}_depth{a['depth']}_{index}")

    if tasks['language_representation']:
        audit=tasks['language_representation']
        names=('full','reset_history_every_token','zero_incoming_embeddings','remove_all_memory_corrections')
        f,a=plt.subplots(figsize=(7.2,2.45))
        for j,(row,color,label) in enumerate(zip(audit['rows'],(gray,blue),
            ('Ours: constant memory','Ours: input-gated memory'))):
            values=[row['interventions'][name]['bpc'] for name in names]
            positions=np.arange(4)+(j-.5)*.35
            a.bar(positions,values,width=.35,color=color,label=label)
            for x,value in zip(positions,values):a.text(x,value+.08,f"{value:.3f}",ha='center',fontsize=7)
        a.set_xticks(range(4),['Full content\nand history','Reset history\nevery character','Zero incoming\nembeddings','Remove all\nmemory corrections'])
        a.set(ylabel='Development bpc ↓',ylim=(0,8.5),title='Fitted predictions depend on content and history')
        a.legend(fontsize=7,loc='upper left');a.grid(axis='x',visible=False)
        f.tight_layout();save(f,'language_content_memory_audit')


def blocks(M, tasks, ev):
    """Project entry point: capabilities, evidence, principles and applications."""
    def compact_work(value):
        if value >= 1e15:
            return f"{value/1e15:.2f}P"
        if value >= 1e12:
            return f"{value/1e12:.2f}T"
        return f"{value/1e9:.2f}G" if value>=1e9 else f"{value/1e6:.2f}M"
    phase=tasks["phase_only"]["final"]["dev"]
    work=tasks["work_audit"]["rows"]
    phase_work=next(r for r in work if r["model"]=="shared_phase_only")["work"]["estimated_operations"]
    recall_work=next(r for r in work if r["model"]=="shared_d2_recall" and r["split"]=="context4x")["work"]["estimated_operations"]
    pool_mean,pool_weighted=tasks["shd_pool_mean"],tasks["shd_pool_weighted"]
    def pooled(row,endpoint="final"):
        if "dev" in row[endpoint]:
            part=row[endpoint]["dev"]
            return part["correct"]/part["n"]
        parts=[row[endpoint][k] for k in ("dev_original","dev_additional")]
        return sum(p["correct"] for p in parts)/sum(p["n"] for p in parts)
    pages=[]
    completed_stage = tasks['language_selective']+[row for row in tasks['language_scaleup']
                                                 if row['args']['fit']==131072]
    stage_bpc=min((row['final']['dev']['bpc'] for row in completed_stage),default=None)
    data_stage=[row for row in tasks['language_scaleup'] if row['args']['fit']==1048576]
    data_bpc=min((row['final']['dev']['bpc'] for row in data_stage),default=None)
    lm_costs={row["model"]:row for row in tasks["training_work"]["language_rows"]}
    banknote_confirmation={family:[r for r in tasks.get('tabular_confirmation',[])
        if r['args']['model']==family and r['args']['tag'].startswith('aws_banknote_confirmation_20261001T234000Z_')]
        for family in ('ours','trees','catboost','logistic')}
    banknote_control_stopped=(ROOT/'experiments/gym/plans/aws_banknote_confirmation_20261001T234000Z/aws_banknote_confirmation_20261001T234000Z_catboost_s8_pilot.failure.json').exists()
    banknote_last_cell=('CatBoost seed8 was stopped without a score; full confirmation is incomplete. '
                        if banknote_control_stopped else 'The last CatBoost cell remains pending. ')
    replicated_banknote=all(sorted(r['args']['seed'] for r in banknote_confirmation[k])==[6,7,8]
                            for k in ('ours','trees'))
    def test_mean(family,metric):return sum(r['final']['test'][metric] for r in banknote_confirmation[family])/len(banknote_confirmation[family])
    banknote_scope=('Competitive accuracy, no confirmed advantage: the three-seed reserved test does not sustain the development lead. ' if replicated_banknote else '')
    if replicated_banknote:
        banknote_scope+=f'Ours averages {100*test_mean("ours","accuracy"):.1f}% versus {100*test_mean("trees","accuracy"):.1f}% for the original trees. '
        if len(banknote_confirmation['logistic'])==3:banknote_scope+='Logistic regression has lower test log loss. '
        banknote_scope+=('All twelve final cells are completed. ' if all(len(v)==3 for v in banknote_confirmation.values())
                         else 'The full four-family confirmation remains incomplete. ')
    else:banknote_scope+='Independent confirmation is pending. '
    pages.append([
        ("title", "Sleeping Machines"),
        ("sub", "A general-purpose architecture that computes with time"),
        ("small", "Tero Keski-Valkama and Karoliina Salminen · Research report · 3 October 2026"),
        ("h1", "Our ambition: a universal learning substrate"),
        ("p", "<b>One architecture for content, time and selective computation.</b> "
         "The ambition is a broadly capable learner that combines language, dense synchronous observations "
         "and sparse asynchronous streams, including their joint arrival into shared persistent state. "
         "Statistical memory, deep learned representations and Transformer-capable retrieval belong "
         "to the same family. Time performs computation; hard routes learn from unrealized alternatives; "
         "small messages mix incoming content with private memory; keys and values remain distinct. "
         "Useful stored capacity can exceed the work recruited for an observation."),
        ("h1", "Already partly demonstrated"),
        ("bullets", [
         "<b>Statistical prediction.</b> Completed count/copy race language comparisons are competitive "
         "with strong counting and dense references. Counts are particularly strong where evidence "
         "supports local statistics; this is family evidence, not a result of the native deep learner.",
         "<b>Learned temporal computation.</b> Deep event chains, temporal rules, persistent vector "
         "representations and counterfactual route credit have completed positive tests, with their "
         "task boundaries and negative confirmations retained in this report.",
         "<b>Native language at 10M.</b> The depth-8/payload-32 route-credit model scores "
         "2.326 test bpc versus the saved one-pass Transformer's 2.427, at the same T256 evaluation "
         "window. The wider depth-4 model scores 2.183 versus the one-pass LSTM's 2.171. "
         "Single seeds, differing training segment lengths; the count reference remains stronger.",
         "<b>A Transformer-capable function class.</b> Delay-coded aggregation reproduces deterministic "
         "softmax attention under its stated conditions (theory §105); the broader event family has "
         "an in-principle emulation path. The current native streaming candidate does not yet "
         "implement the complete Transformer-equivalent stack."]),
        ("h1", "The opportunity and the remaining bridge"),
        ("p", "<b>We know of no mathematical obstruction to this architectural direction.</b> "
         "The research question is whether native learning realizes this breadth efficiently at scale. "
         "Expressivity alone does not guarantee optimization, generalization or lower total resource use. "
         "Joint multimodal learning and comparable-quality large-data advantage remain to be demonstrated."),
        ("p", "Race selection has the exact softmax winner probabilities, but one winning value matches "
         "attention only in expectation; subsequent nonlinear layers do not generally commute with that "
         "expectation. Exact delay-coded aggregation instead pays deliveries, normalization, latency and "
         "precision. Roughly halving attention aggregation arithmetic is a conditional inference "
         "opportunity, not a demonstrated halving of complete-model inference. The completed "
         "approximately 49.66% saving concerns replay fitting work, a separate result. "
         "The target is better prediction at a fully counted resource budget (§§280,317).")])
    pages.extend(runpy.run_path(str(ROOT/'report/current_language_status.py'))['pages'](tasks['current_language_status']))
    pages.append([
        ("h1", "Deep learning that computes with time"),
        ("p","Messages carry content and an arrival time. Nodes mix incoming vectors with persistent memory, "
         "gate their updates and compete through learned delays. Arrival order and winning races determine the computation. "
         "The goal is useful intelligence with much less active work."),
        ("h1","The differentiators at a glance"),
        ("bullets",[
         "<b>Time performs computation.</b> Delays, races and phase transformations implement useful functions.",
         "<b>Hard routes can learn.</b> Winning messages execute; unrealized alternatives receive counterfactual credit.",
         "<b>Deep, persistent event representations.</b> Vector messages and local memory carry information and credit through layers; retrieval and temporal primitives share the model family.",
         "<b>Capacity beyond activity.</b> The scaling goal is more useful dormant capacity, selectively recruited and judged by prediction quality at a given total work budget."]),
        ("h1","The strongest demonstrated results"),
        ("bullets",[
         "<b>Generalization.</b> Race retrieval reaches <b>100% at four times the training context</b> "
         "within 4,000 examples in all five runs. A learned phase rule solves <b>all 3,440 unseen "
         "modular triples</b>, using the supplied period 17.",
         "<b>Learning from fewer examples.</b> Depth-three event chains reach <b>99.73–99.93%</b> "
         "after 2,000 examples seen once; saved Transformer controls reach <b>33.25–40.80%</b> "
         "with the same number of distinct examples and repeated fitting. Depth-four chains reach 99.9–100%.",
         "<b>Learned representations.</b> Completed temporal-carrier development screens reach "
         +(f"<b>{stage_bpc:.3f} bpc at 131K</b> " if stage_bpc is not None else "")+
         (f"and <b>{data_bpc:.3f} at 1M fitting characters</b>, four passes. " if data_bpc is not None else "fitting characters. ")+
         "A learned speech encoder reaches <b>79.69%</b> on 512 private development utterances. "
         "Embeddings, temporal state and vector maps learn."
         +(f" Calibration: closed-form Kneser–Ney counts of the same fitting data score {count_reference_bpc(131072):.3f} / "
           f"{count_reference_bpc(1048576):.3f} bpc on the same targets and lead these small-data comparisons. "
           "They are strong references where local statistics are well supported; architecture advantage "
           "requires practical headroom (§§376,393–394)." if count_reference_bpc(131072) is not None and count_reference_bpc(1048576) is not None else "")]),
        ("figure",("accomplishments",174)),
        ("small","Left: means and recorded ranges, five event runs and two Transformer runs; "
         "2,000 distinct examples, seen once / presented 400,000 times. Right: all five event runs "
         "reach 100% within 4,000 examples; the control is the best saved result across seven "
         "Transformer configurations and their learning curves. These synthetic tasks use different "
         "architectures and structural priors. Sources: E53/E36 and E61.")])

    banknote={r['args']['model']+str(r['args']['clock_features']):r for r in tasks.get('native_tabular',[])
              if r['args']['dataset']=='banknote' and r['args']['tag'].startswith('aws_fast_matrix_recovery_20261001T213409Z_')}
    if 'ours0' in banknote and 'trees0' in banknote:
        if any(banknote['ours0']['protocol'][k]!=banknote['trees0']['protocol'][k]
               for k in ('raw_sha256','fit_sha256','dev_sha256','fit_indices','dev_indices')):
            raise ValueError('Opening banknote comparison requires identical fitting/development data')
        ours,trees=banknote['ours0']['final']['dev'],banknote['trees0']['final']['dev']
        gain=100*(1-ours['nll']/trees['nll'])
        pages.append([
            ('h1','New evidence: quality and complete work'),
            ('p',f'<b>Tabular: competitive accuracy, no confirmed win.</b> '+
             (f'Ours averages <b>{100*test_mean("ours","accuracy"):.1f}%</b> reserved-test accuracy versus '
              f'<b>{100*test_mean("trees","accuracy"):.1f}%</b> for boosted trees across three seeds. '
              f'Logistic regression reaches <b>{100*test_mean("logistic","accuracy"):.1f}%</b> and lower log loss '
              f'(<b>{test_mean("logistic","nll"):.3f}</b> versus ours <b>{test_mean("ours","nll"):.3f}</b>). '
              if replicated_banknote and len(banknote_confirmation['logistic'])==3 else banknote_scope)+
             'The native eight-block model mixes content and memory through parallel temporal receiver heads.'),
            ('figure',('banknote_reserved_test' if replicated_banknote else 'banknote_first_screen',174)),
            ('small',('Means and individual seeds6/7/8 on281 reserved rows (270 feature groups). '
              '128 fitting/128 development rows; four fixed selection opportunities. '
              'Accuracy uncertainty includes zero difference, but does not establish statistical equivalence. '
              if replicated_banknote else '128 fitting/128 development rows, four passes, seed6. ')+
             f'The original development lead, {100*ours["accuracy"]:.1f}% versus {100*trees["accuracy"]:.1f}%, is retained in Appendix B. '
             +(banknote_last_cell if not all(len(v)==3 for v in banknote_confirmation.values()) else '')+
             'Tree FLOPs are unavailable and CPU fits are faster; no resource advantage over trees is established.'),
            ('p',f'<b>Statistical memory where counting is strong (10M characters).</b> On the same999,999 test targets, '
             f'ours count/copy race mixture scores <b>{ev["native10"]:.3f}bpc</b> versus '
             f'<b>{ev["lstm10"]:.3f}</b> for LSTM and <b>{ev["tf10"]:.3f}</b> for Transformer'
             +(f'; closed-form counts alone (untuned {count_scale_reference(10_000_000)["method"]}, order {count_scale_reference(10_000_000)["order"]}) '
               f'score <b>{count_scale_reference(10_000_000)["bpc"]:.3f}</b>' if count_scale_reference(10_000_000) else '')+'. '
             'Here counting statistics are near-optimal and the dense controls sit at their level; learned models overtake '
             'them only with far more data and parameters (Theory §§381, 394). Our statistical memory therefore adds a useful '
             'information path on top of near-optimal counts. It is not the learned native model. Capacity and fitting budgets differ. '
             'Appendix B charges floating mixing work and reports integer table work separately.'),
            ('p','<b>Work between two learned language models.</b> Ours native2K uses <b>3.78 whole-fit GFLOPs</b> versus '
             '<b>22.75 GFLOPs</b> for the saved KV2K construction: <b>6.02× less counted work</b>, '
             'at 3.765 versus 3.733 development bpc (0.032 worse). Both use four passes and 8,191 scored development targets; '
             'width, capacity and memory construction differ. Complete CPU fitting traces include counterfactual learning and Adam. '
             'This compares two learned models with each other. Near-optimal count references for this small-data regime '
             'are shown in Appendix B as calibration (Theory §§393–394).'),
            ('small','This banknote comparison concerns one task. Strong synthetic order/retrieval evidence '
             'on the preceding page remains valid under its own protocols. Appendix B retains the full cross-domain comparisons and resource ledgers.')])
        nlb={r['label']:r for r in tasks.get('native_language_batched',{}).get('native',[])}
        ctl={r['label']:r for r in tasks.get('native_language_batched',{}).get('controls',[])}
        credited=[r for k,r in nlb.items() if 'route credit' in k and r.get('sparse')]
        best=min(credited,key=lambda r:r['test']) if credited else None
        if best and 'p32/d4' in nlb and 'p32/d4 + route credit' in nlb and 'LSTM-256' in ctl:
            pages[-1].insert(-1,('p','<b>Learned native language at 10M, one pass (THEORY §413).</b> Alternative-value credit '
                '(forward values unchanged) improves the integrated '
                f'native core from <b>{nlb["p32/d4"]["test"]:.3f} to {nlb["p32/d4 + route credit"]["test"]:.3f}</b> '
                f'test bpc at the same size. More width reaches <b>{best["test"]:.3f}</b> '
                f'(T256 {best["test256"]:.3f}) versus <b>{ctl["LSTM-256"]["test"]:.3f}</b> for LSTM-256 and '
                f'<b>{ctl["Transformer-256x2"]["test"]:.3f}</b> for Transformer. Winner-only trace: '
                f'<b>{best["sparse"]/1e6:.2f}</b> versus {ctl["LSTM-256"]["infer"]/1e6:.2f} MFLOPs/position; '
                f'fitting {best["fit"]/1e6:.2f} versus {ctl["LSTM-256"]["fit"]/1e6:.2f} MFLOPs/character. '
                'Traced/estimated conventions differ; single seeds, more work than LSTM, trained sparse parity pending. '
                'The native appendix retains every arm and failed write credit.'))
        native8=[r for r in tasks.get('native_language',[]) if r['args']['fit']==8192
                 and r['args']['seed']==6 and (r['args']['heads'],r['args']['payload'],r['args']['depth'])==(2,16,8)]
        if native8:
            scaled=native8[0]
            pages[-1].insert(-1,('p',f'<b>Native data scaling.</b> The same 54,907-parameter construction improves '
                f'from <b>3.765 to {scaled["final"]["dev"]["bpc"]:.3f} bpc</b> when fitting data grows from2K to8K characters, '
                f'using <b>{scaled["work"]["cpu_emulator"]["total_training_unit_special_flops"]/1e9:.2f} whole-fit GFLOPs</b>. '
                'Both use four passes and the same 8,191 development targets; this is one-seed completed data-scaling evidence.'))

    split=completed_split_evidence(tasks)
    if split:
        timed=split['paired_timing_S4_private_P0_observed']
        rank=split['paired_timing_S4_private_P0_rank']
        shared=split['order_S16_shared_P0'];private=split['order_S16_private_P0']
        rows=[]
        for label,r in (('Ours: elapsed time',timed),('Ours: order-only control',rank),
                        ('Ours: 16-source shared rules',shared),('Ours: 16-source private rules',private)):
            w=r['work']
            rows.append([label,f'{100*r["final"]["dev"]["accuracy"]:.2f}',f'{r["parameters"]:,}',
                         f'{w["total_training_unit_special_flops"]/1e9:.3f}',
                         f'{w["total_training_unit_special_flops"]/w["fitting_query_targets"]/1e6:.3f}',
                         f'{(w["inference_arithmetic_flops_per_query"]+w["inference_special_functions_per_query"])/1e6:.3f}'])
        pages.append([
            ('h1','Native strengths: useful time and private state'),
            ('p',f'<b>Elapsed time carries useful information.</b> Ours reaches <b>{100*timed["final"]["dev"]["accuracy"]:.2f}%</b> '
             'on paired short/long sequences with identical marks, addresses and event order but opposite labels. '
             'The refitted order-only control reaches exactly <b>50%</b>; paired noise makes that ceiling exact. '
             'Clearing persistent state also reduces ours to 50%.'),
            ('figure',('native_mechanism_evidence',174)),
            ('p',f'<b>Learning can be shared while memories stay private.</b> At 16 occupied sources, shared processing '
             f'raises accuracy from <b>{100*private["final"]["dev"]["accuracy"]:.2f}% to {100*shared["final"]["dev"]["accuracy"]:.2f}%</b>, '
             f'with <b>{private["parameters"]/shared["parameters"]:.1f}× fewer parameters</b> and '
             f'<b>{100*(1-shared["work"]["total_training_unit_special_flops"]/private["work"]["total_training_unit_special_flops"]):.1f}% less whole-fit work</b>. '
             'All 512 receiver slots remain available; each event commits 16 states and scores 32 keys in both constructions. '
             'Sharing also removes private source embeddings; these effects are tested together.'),
            ('table',(['Construction','Dev accuracy<br/>%','Parameters','Whole fit<br/>GFLOPs','Fit/query<br/>MFLOPs','Infer/query<br/>MFLOPs'],rows,[51,23,24,24,26,26])),
            ('small','Eight blocks, two independent heads, d8, pool 2; 128 fitting queries/pass, four passes, 256 development queries, seed 6. '
             'Timing uses 32 independent population pairs; 16-source order uses 16 populations. Exploratory population-bootstrap gains '
             'are 45.31 pp [42.97, 47.66] for time and 31.25 pp [25.39, 37.11] for shared rules; 95% intervals condition on selected checkpoints, '
             'not independent seeds or confirmation. Complete counted fitting includes losing proposals, backward, clipping and Adam; '
             'special functions have unit weight. These are synthetic mechanism advantages, not superiority over time-aware dense models or measured energy.'),
            ('small','Sources: <a href="experiments/AWS_SPLIT_EVENT_BATTERY.md">frozen11-pilot protocol</a> and '
             '<a href="experiments/SPLIT_SCREEN_FINDINGS_20261002.md">validated findings</a>. Full variants and gap-retention diagnostics remain in Appendix B.')])

    reference_rows=[
        ["Ours: learned event-state model (planned)", "10M / four passes", "Pending", "Pending"],
        ["LSTM; width 512, one recurrent layer", "10M / six passes", f"{ev['lstm10']:.3f}",
         compact_work(lm_costs['lstm']['total_training_flops'])],
        ["Transformer; width 256, four layers", "10M / four passes", f"{ev['tf10']:.3f}",
         compact_work(lm_costs['tf']['total_training_flops'])],
    ]
    reference_sources=[
        '<a href="experiments/results/e174/aligned_lstm_10m_20260930.json">10M LSTM aligned result</a>',
        '<a href="experiments/results/e174/aligned_tf_10m_20260930.json">10M Transformer aligned result</a>',
    ]
    official=[row for row in tasks['language_scaleup']+tasks['language_full_sparse'] if row['args'].get('official_test')]
    if official:
        reference_rows.pop(0)
        for row in official:
            protocol=row['protocol']
            if (protocol['fitting']!=[0,10_000_000] or protocol['development']!=[90_000_000,90_200_000]
                    or protocol['test']!=[95_000_000,96_000_000] or not protocol['official_test_read']
                    or not protocol['weights_frozen_on_test'] or protocol['statistical_experts']
                    or row['final']['official_test']['n']!=999_999):
                raise ValueError('Completed learned language result does not match the comparison protocol')
            label=(f"Ours: integrated races; payload {row['args']['payload']}" if 'payload' in row['args'] else
                   f"Ours: input-gated event state; width {row['args']['width']}")
            reference_rows.insert(0,[label,
                '10M / four passes',f"{row['final']['official_test']['bpc']:.3f}",
                compact_work(row['work']['total_training_unit_special_flops'])])
    for model, label in (("lstm", "LSTM; width 512, one recurrent layer"),
                         ("tf", "Transformer; width 256, four layers")):
        reference = ev["aws_references"][model]
        if reference:
            row = reference["result"]
            cost = row.get("training_flops_estimate", {}).get("total_training_flops")
            reference_rows.append([label, f"90M / {row['args']['passes']:g} passes",
                                   f"{row['test_bpc']:.3f}", compact_work(cost) if cost else "Not audited"])
            reference_sources.append(f'<a href="{reference["path"]}">90M {model.upper()} saved result</a>')
    new_aws_pages=[]
    aws_sparse=[r for r in tasks['language_full_sparse'] if r['args']['tag'].startswith('aws_')]
    if aws_sparse or any(ev['aws_references'].values()):
        latest=[];resources=[]
        for r in aws_sparse:
            a=r['args'];w=r['work'];latest.append([f"Ours sparse d{a['payload']}/L{a['depth']}",f"{a['fit']:,}/{a['epochs']}",f"{r['final']['dev']['bpc']:.3f}",'Not scored',f"{w['total_training_unit_special_flops']/1e9:,.2f}",f"{w['total_training_unit_special_flops']/w['fitting_targets']/1e6:.3f}"])
            resources.append(['Ours sparse',f"{r['parameters']:,}",f"{r['wall_s']/3600:.2f}",f"{r['max_rss_kb']/1024:.1f}",f"{(w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'])/1e6:.4f}"])
        for family,label in (('lstm','LSTM512'),('tf','Transformer256/L4')):
            ref=ev['aws_references'][family]
            if not ref:continue
            r=ref['result'];w=r['training_flops_estimate'];meta=json.loads((ROOT/ref['path']).with_name('provenance.json').read_text())
            latest.append([label,f"90M/{r['args']['passes']:g}",f"{r['best_valid_bpc']:.3f}",f"{r['test_bpc']:.3f}",f"{w['total_training_flops']/1e9:,.2f}",f"{w['total_training_flops']/w['training_token_positions']/1e6:.3f}"])
            resources.append([label,f"{r['params']:,}",f"{meta['wall_s']/3600:.2f}",f"{meta['peak_rss_kb']/1024:.1f}",f"{w['forward_flops']/w['training_token_positions']*(2 if family=='tf' else 1)/1e6:.4f}"])
        new_aws_pages.append([
            ('h1','Appendix B. New completed AWS language evidence'),
            ('p','The six-block sparse receiver model reaches 3.106 development bpc on 32K fitting characters. '
             'It learns meaningful prediction, but this is the older single-head, observed-character-pool variant, '
             'not the newer eight-block native/content-gated reception model. The large dense controls are completed '
             'reference targets with much better held-out quality and much larger fitting budgets.'),
            ('table',(['Model','Fit chars / passes','Dev bpc ↓','Test bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓'],latest,[39,29,23,24,31,28])),
            ('table',(['Model','Parameters','Guarded wall h','Peak RSS MiB','Infer MFLOPs / char ↓'],resources,[45,31,29,29,40])),
            ('p','All rows use the same units and whole-fit/per-target denominators within each column. Sparse work '
             'is a representative full-step arithmetic estimate with unit-weight special functions; dense work '
             'uses shapes and backward = twice forward, including clipping/Adam. Both exclude evaluation and traffic. '
             'Dense inference uses the saved shape convention, including Transformer window overlap; sparse inference is a winner-only trace. '
             'Guarded wall and RSS include different simulator/runtime overheads and are not energy measurements.'),
            ('small','The sparse development set has 8,191 targets; dense selection uses 200,000 validation characters '
             'and the saved 1M test interval. Neither their development scores nor their fitting budgets are matched. '
             'Raw resource gaps cannot establish comparable-quality or iso-FLOP supremacy. The interrupted Transformer '
             'attempt remains preserved; only its completed retry appears as a quality point. All completed points, '
             'including the 90M Transformer, are retained in the common quality/work and inference figures.')])
    hierarchy=tasks.get('aws_hierarchy',[])
    large=[r for r in hierarchy if r['result']['config']['train']==64000]
    if large:
        new_aws_pages.append([
            ('h1','Appendix B. AWS hierarchy depth: a useful constraint'),
            ('p','These are older race-network diagnostics on the fixed depth-three Random Hierarchy Model. '
             'They test composition and depth, not the newer integrated language/event construction. Every shown '
             'model uses width 200, 64K fitting examples, ten passes, architecture seed 0 and rule seed 0.'),
            ('figure',('aws_hierarchy_depth',160)),
            ('table',(['Model','Depth','Final held-out accuracy','Guarded wall s','Peak RSS MiB'],[
                ['Ours plain race' if r['result']['config']['residual']==0 else f"Ours residual-{r['result']['config']['residual']}",
                 str(r['result']['config']['depth']),f"{100*r['result']['test_acc']:.2f}%",f"{r['provenance']['wall_s']:.1f}",f"{r['provenance']['peak_rss_kb']/1024:.1f}"]
                for r in sorted(large,key=lambda r:(r['result']['config']['residual'],r['result']['config']['depth']))],[46,18,46,34,30])),
            ('p','Plain depth 1/2/3/4 gives 70.12/83.72/85.06/84.12%: depth helps to three blocks, then regresses '
             'slightly. The completed residual-2 depth 2/3/4 variants give 72.72/74.54/72.56%, below their plain '
             'counterparts. Depth-four residual accuracy rises throughout its ten passes; its longer-budget '
             'convergence is untested. This constrains the tested implementation and budget, not all residual paths.'),
            ('small','Only completed provenance files produce rows. Plain and residual depth-four runs are distinct. '
             'Final epoch is reported; evaluation curves '
             'are visible throughout fitting, so this is exploratory held-out evidence, not independent confirmation. '
             'Single rule/model seed; no 64K matched dense-control or complete FLOP/energy supremacy is inferred.')])
    language_reference_page=[
        ("h1","Appendix B (continued). Completed language references"),
        ("p","The Transformer and LSTM benchmarks have already been run. Their completed result files "
         "remain in the repository and are reused as reference targets for the full learned-event benchmark. "
         "Lower bits per character (bpc) means better prediction."),
        ("table",(["Model","Fitting characters / passes","Test bpc ↓","Full training FLOPs ↓"],
                  reference_rows,[60,46,24,44])),
        ("p","The 10M references score exactly the same 999,999 text8 targets in [95M,96M), "
         "with frozen validation-selected weights and cold initial context. The 90M LSTM uses the same "
         "test interval and its saved recurrent scoring protocol. All use the historical 27-character alphabet "
         "and 200,000-character validation selection; data budgets, capacities and fitting passes differ."),
        ("p","Training estimates include every fitting step, forward/loss, backpropagation, gradient clipping "
         "and Adam. Backward is approximated as twice forward; multiply-add counts as two FLOPs. "
         "G/T/P mean billion/trillion/quadrillion. Validation/test evaluation, memory traffic and runtime "
         "are outside these arithmetic totals."),
        ("h2","Ours: learned-language benchmark status"),
        ("p",("The completed learned-model row above scores the same cold-context character targets as "
         "the saved controls, with weights frozen. Its fitting arithmetic and additional special functions "
         "are recorded separately; the table includes unit-weight specials for comparison with the older "
         "neural estimates. Architecture/capacity and optimization differ. " if official else
         "The planned comparison uses 10M fitting characters, four passes, 200,000 validation characters "
         "and the same 1M test interval. Priority is the integrated sparse/timed architecture documented "
         "in the next appendix. Completed dense-carrier fits remain diagnostic controls; their quality "
         "does not establish sparse-model performance. Numerical contracts and progressively larger "
         "integrated development fits precede promotion. The aligned full-test result and its fitting "
         "work remain pending. ")+
         "The preserved 28,403-parameter pilot's 3.351 development bpc uses a smaller fitting budget "
         "and different split; it is not a comparable test result."),
        ("p",f"Earlier 1M-character references also remain saved: LSTM <b>{ev['lstm1']:.3f}</b> "
         f"and Transformer <b>{ev['tf1']:.3f} test bpc</b>, each with twenty fitting passes. "
         "The separate count/copy baseline and the cross-task Transformer/retrieval LSTM comparisons "
         "remain in their labeled sections and Appendix B."),
        ("small","Saved evidence: "+"; ".join(reference_sources)+". An earlier 90M Transformer attempt was "
         "interrupted by its RSS watchdog before producing a completed test result; its provenance is "
         "preserved.")]

    pages.append([
        ("h1","Why this research matters"),
        ("p","Sparse neural computation promises to spend work only where information changes. The hard "
         "part is teaching useful deep representations when routes can be silent, discrete or absent. "
         "A cheap forward pass is insufficient if discovering those routes consumes the savings."),
        ("table",(["Research lineage","What it established","The question we pursue"],[
         ["Neuromorphic and spiking networks","Event-driven signals and trained spike timing; EventProp differentiates at events.","How do inactive alternatives receive useful credit without exhaustive replay?"],
         ["Temporal logic and learned delays","Race/delay algebra computes with time; delay learning already improves SNN recognition.","Can temporal computation coexist with rich vector content and deep learned state?"],
         ["Sparse conditional models / MoE","Selected experts allow capacity to grow faster than active work; Switch trains at scale.","Can message timing, communication and correction work also become selective?"],
         ["Asynchronous state-space models","EventSSM learns asynchronous streams with parallel scans; this is a strong precedent.","Can hard races and counterfactual alternatives add quality per unit of total work?"],
        ],[39,66,69])),
        ("h2","What is distinctive here"),
        ("p","We combine computation through trainable time, content-bearing messages, persistent local state "
         "and credit to unrealized alternatives. Optionality asks whether distinct, reachable future corrections "
         "remain available under a work budget. The theory connects temporal algebra, key/value separation, "
         "credit transport and supervision that includes silence. The contribution is this construction and "
         "its tested consequences; learned delays and sparse capacity are established ideas."),
        ('p','The integrated language candidates now exercise learned content, temporal races, sparse '
         'persistent receivers and counterfactual credit together. An eight-block variant also races '
         'historical key/value messages. These small-data experiments explore only a small part of '
         'the design space; larger-scale quality and complete resource advantages remain under test.'),
        ("p","Our earlier deep sparse-routing pilots often lost activity and useful credit before the final "
         "layers. Counterfactual proposals alone did not reliably fix that. The subsequent vector-state and "
         "persistent-memory work addresses those observed obstacles. Completed structured-task gains motivate "
         "the larger learned-model tests; broad quality, training efficiency and energy must still be measured together."),
        ("small",'Primary precedents: <a href="https://www.nature.com/articles/s41598-021-91786-z">EventProp</a>; '
         '<a href="https://arxiv.org/abs/2001.04242">Space-Time Algebra</a>; '
         '<a href="https://arxiv.org/abs/2306.17670">Learning Delays in SNNs</a>; '
         '<a href="https://www.jmlr.org/papers/v23/21-0998.html">Switch Transformers</a>; '
         '<a href="https://arxiv.org/abs/2404.18508">EventSSM</a>. '
         'Our routing failures and revised interpretations remain in the theory index and findings.')])

    pages.append([
        ("h1","One architecture, several learned computations"),
        ("p","The common idea is local computation triggered by an arrival: retain memory, combine the "
         "incoming vector with that memory, and choose an outgoing time. A delay changes which messages "
         "meet and which race finishes first. This makes timing part of the learned function. "
         "The model family implements this idea at several levels of generality."),
        ("figure",("shared_architecture",154)),
        ("table",(["Model","Mechanism","Evidence","What it establishes"],[
         ['Ours: integrated sparse temporal language','Learned content, state-dependent key races, selected receiver updates and episodic KV','Completed 32K receiver / 2K depth and KV screens','Combined mechanisms train; bounded candidate coverage and causal schedule'],
         ["Ours: learned event-state encoders","Source embeddings, temporal modes, nonlinear vector maps and competing clocks","Language E176; speech E165","Learned representations and persistent state"],
         ["Ours: routed event query encoders","Candidate payloads, receiver memory and hard value/time races","Breadth E120; language E133","Trainable event depth across tasks; text/market breadth variants add statistical evidence"],
         ["Ours: structured event mechanisms","Temporal chains, relative pointers and phase composition","E34/E53/E54, E61, E124","Sample efficiency and generalization with declared structural priors"],
         ["Ours: statistical controls","Conditional counts, backoff and copy probabilities","E173; online pilot","Separate baselines; no event backbone"],
        ],[38,57,33,46])),
        ("p","<b>Incoming content is retained.</b> In the event-state encoder, an incoming vector is "
         "projected into rotating, decaying memory. A learned memory read is mixed with a direct input path, "
         "normalized and gated. The outgoing vector adds that correction to the incoming vector. The payload "
         "therefore depends on both current content and history; timing supplies an additional control."),
        ("small","Each task has separately fitted weights. Current learned encoders use fixed depth and locally "
         "dense vector maps; the language scheduler retains state and delayed messages across chunks. "
         "The routed query encoder instead rebuilds a supplied context per query. Clock learning uses "
         "declared surrogate credit through a hard schedule; it is not an exact derivative of every order change.")])

    pages.append([
        ("h1","How the model computes and learns"),
        ("p","<b>Time performs computation.</b> An arrival time is a computational value. A delay adds to "
         "that value; a first-arrival race computes a minimum and selects a payload; coincidence detects "
         "the latest required arrival. Inhibition can veto a path. With a declared periodic reference, "
         "phase transformations compose reusable modular relations. Learned timing therefore changes "
         "the function and its causal paths, beyond deciding when a fixed dense calculation runs."),
        ("bullets",[
         "<b>Local temporal memory</b> accumulates observed content and elapsed time without evaluating empty time ticks.",
         "<b>Computation through delays</b> uses waiting times, arrival order and clock races to transform information and select outcomes.",
         "<b>Hard races and counterfactual credit</b> choose the emitted vector and delay. Losing alternatives teach better routes while remaining distinct from the winning forward message.",
         "<b>Separate keys and values</b> let content-dependent keys set routes and clocks while value learning preserves the selected schedule.",
         "<b>Trainable depth</b> carries representations and learning credit through a hierarchy. The reversible construction preserves conditional norms under its stated assumptions; readout visibility and routing remain necessary.",
         "<b>Structured memories</b> include relative pointers, learned phase transformations and conditional evidence. Their task gains retain their declared priors; statistical experts are labeled separately.",
         "<b>Appropriate supervision</b> teaches a completed class decision or the next event's type and waiting time, including information carried by silence."]),
        ("p","For a temporal pattern such as A followed by B, a learned delay can bring A's trace into "
         "coincidence with B. A competing path can veto the match when C intervenes. The timing-pattern "
         "and compositional experiments test these mechanisms; the language and speech encoders learn "
         "richer vector messages and temporal state."),
        ("small",'The simulator stores arrival coordinates on a common axis; the native function uses local elapsed intervals, precedence and causal joins, not a globally ticking execution clock. A time-origin shift preserves predictions. CPU serialization and a fabricated clockless ASIC are separate implementation claims. The primitives and their symmetry limits are developed in '
         '<a href="experiments/theory/05_temporal_computation_and_scaling.md">temporal computation theory, §56</a>; '
         '<a href="experiments/theory/01_foundations_and_counterfactual_credit.md">counterfactual learning</a>, '
         '<a href="experiments/theory/22_key_value_separation_and_race_boundaries.md">key/value separation</a> '
         'and <a href="experiments/theory/25_reversible_event_memory_and_depth.md">reversible depth</a> '
         "give the learning contracts. Clockless delay/race networks compute relative timing relations; "
         "phase arithmetic requires its reference. Each implemented model uses a declared subset. "
         "Candidate discovery and training alternatives are charged to the work ledger.")])

    pages.append([
        ('h1','Useful functions from time, reception and repeated events'),
        ('p','A dot product is a compatibility score along one direction. Projecting a message into a learned '
         'two-dimensional plane and rotating a local clock vector changes which content direction is receptive. '
         'Separate clock modes and independent heads can gate different components, then compose a new vector '
         'while retaining the incoming content. This creates content–time interactions, rather than just delaying a fixed computation.'),
        ('figure',('temporal_reception_windows_trains',145)),
        ('table',(['Ours: construction','Analytical function and learning','Implementation status'],[
            ['Temporal reception','Content dot a rotating learned direction; exact ordinary phase/projection gradients within a route history','Full native candidate; two/four-mode fits and same-clock waiting control'],
            ['Learnable integration window','Compact smooth kernel; exact membership gradients; five moment vectors, no silent-time ticks','Primitive contracts passed; full-model integration remains'],
            ['Information-bearing train','Same first spike, different later evidence, distinguishable train readout; exact fixed-count gradients','Closed-form event solver and train/window contracts passed']], [35,83,56])),
        ('p','Parameters are useful when their interactions preserve relevant information, reach the readout and '
         'receive adequate credit and exposure. Consecutive affine maps can fuse into one; gated products and '
         'temporal state create new interactions. The appropriate mode count or local rank is determined by '
         'marginal held-out quality per complete work, not by maximizing parameter count.'),
        ('small','The figure is an analytical construction, not measured model performance. Hard destination '
         'changes and spike creation/deletion still require boundary or counterfactual credit. More emissions '
         'are charged; no biological rate-code or free-energy claim. Theory §§337–352 gives the proof, '
         'costs, timing-noise limits and frozen/refitted ablation protocol.')])
    race_rows=tasks['language_race']
    pages.append([
        ('h1','The ambition: useful capacity without proportional activity'),
        ('p','The proposed shift is to compute through event timing and selectively active paths. '
         'A larger network should be able to retain more useful dormant structure while spending work '
         'on the paths a query needs. Counterfactual credit must teach those hard choices, including '
         'useful alternatives that did not win. The earlier temporal-chain, pointer and phase results '
         'test parts of this case and remain central evidence.'),
        ('h2','What temporal softmax actually provides'),
        ('p','If candidate clocks have rates exp(score), their first-arrival winner has exactly the '
         'softmax choice probabilities. Competition supplies normalization in time. It can avoid an '
         'explicit normalizing sum/division in the winner path when those rates are physically available. '
         'Score formation, candidate discovery, value delivery and learning still cost work. This identity '
         'is not a measured near-zero-energy attention system.'),
        ('table',(['Ours: mechanism','Completed evidence or status','What remains'],[
         ['Temporal chains / hard pointers / phase rules','Strong structured-task accuracy, transfer and work comparisons','Transfer the useful priors to broad learned representations'],
         ['Temporal content carrier','Learned embeddings, state, gates and delay-dependent transport','Every layer executes; does not demonstrate dormant-unit scaling'],
         ['Integrated sparse temporal model','Hard races, content memory, sparse state updates and counterfactual teachers pass contracts','Scaling quality and complete learning work are under test'],
         ['Capacity beyond activity','Gains depend on useful sparsity and learning','Measure marginal useful capacity with bounded active work'],
        ],[45,67,62])),
        ('h2','A direct mechanism experiment'),
        ('p','The native candidate combines the mechanisms: an addressed content/time event enters '
         'eight blocks with two independent receiver heads, selecting one persistent content-bearing '
         'unit per head at each depth. Separate state-dependent keys '
         'set rates; the winner mixes incoming content and retained memory, then emits a vector and '
         'learned arrival time. Addressed losing values receive counterfactual score credit during '
         'training. The dense carrier and carrier-plus-retrieval variants remain diagnostic controls.'),
        ('small','Observed-source pools and fixed depth are declared priors; learned topology growth '
         'and unrestricted asynchronous schedules remain open. RNG and physical traffic are additional. The old '
         'time-normalized value sum has a shared random amplitude; its covariance and cutoff claims '
         'are corrected beside the original theory, not silently deleted. A centered, conserved teacher '
         'is now tested. Its fixed-error expected Jacobian is not an unbiased sampled-loss gradient. '
         'Theory §§294–298: experiments/theory/45_race_attention_and_resource_identity.md.')])

    pages.append([
        ('h1','A native path to more capability per unit of work'),
        ('p','The next integrated experiments test what this substrate does naturally: '
         'learned time computation, sparse addressed memory and hard choices trained through '
         'counterfactual credit. A derived gated-state kernel accumulates ordered interactions '
         'without enumerating past pairs; learning useful such representations is the target. Episodic race attention remains a useful preserved comparison; '
         'the native branch does not require a per-position attention bank.'),
        ('figure',('native_addressed_event_path',174)),
        ('table',(['Ours: native construction','Exact activity boundary'],[
            ['S observed sources; L blocks; H heads; P candidates','Available receivers: S × L × H × P'],
            ['One selected receiver per head/block','Selected state commits per event: L × H'],
            ['All addressed candidates are scored','Key scores and training proposals per event: L × H × P']], [88,86])),
        ('p','Useful capacity can grow while selected activity stays fixed, but its value must be '
         'learned. At a fixed data budget more local maps receive fewer examples. Sharing learned '
         'maps while retaining separate state is one way to improve learning exposure; the native '
         'language adapter tests this directly. It changes capacity and is a whole-construction comparison.'),
        ('p','Analytic state evolution avoids periodic simulation during silence. Decay can still '
         'erase information, so long-gap accuracy is measured separately from operation count. '
         'A protected-content subspace alongside evolving time modes is a derived next hypothesis, '
         'to test if the current construction loses useful memory.'),
        ('small','The current eight-block/two-head/pool2 model selects 16 commits and scores 32 keys '
         'per event. Shared maps, content transforms, losing proposals, backward, Adam and source-local '
         'causal waits remain paid. The ordered kernel is a restricted algebraic identity, not an '
         'achieved capability of the fitted model. Full-depth contracts and accounting smokes pass; quality pilots, '
         'refitted controls and independent seeds determine further scaling. Theory §§330–336; '
         'completed results and the executable priority appear in Appendix B.')])

    if tasks.get('joint_outcome'):
        relation=tasks['joint_outcome'][-1]
        rows={row['arm']:row for row in relation['common_unit_ledger']}
        j,l,s=[rows[key] for key in ('joint_full','local_full','joint_shallow')]
        blocks=[('h1','Useful old evidence: a completed joint-learning intervention'),
            ('p',f"Protected outcome state and two learned key/value races reach "
             f"{100*j['fresh']['accuracy']:.2f}% on new suffixes, versus "
             f"{100*l['fresh']['accuracy']:.2f}% with matched local-credit training. "
             'The query suffix and its actual count inputs are identical within each opposite-label '
             'group. Predicting the distant relation therefore requires additional observed evidence.'),
            ('figure',('report/figures/'+relation['args']['tag']+'_learning.png',173)),
            ('p',f"Reserved loss:{j['fresh']['query_bits']:.3f} versus {l['fresh']['query_bits']:.3f}bits/query. "
             f"All1,024 fitting presentations, complete prefix and optimizer work charged: "
             f"{j['whole_fit_gflops_estimate']:.4f} versus {l['whole_fit_gflops_estimate']:.4f}GFLOPs estimated. "
             'Joint training teaches the loss of candidate pairs; inference delivers only two values '
             'after scoring all occupied keys. Initial models and inference mechanisms match.'),
            ('p',f"Shallow joint reaches{100*s['fresh']['accuracy']:.2f}% at "
             f"{s['whole_fit_gflops_estimate']:.4f}GFLOPs. This task supports protected evidence and "
             'terminal joint learning; useful extra core depth is not established. Earlier native/tapped '
             'fits and frozen readout failures remain in the appendix.'),
            ('small','One fitted seed,128 reserved synthetic queries. Observed predecessor addresses '
             'are fixed; terminal content-risk derivatives are exact conditionally, earlier native '
             'route derivatives remain scoped. The bound concerns query-count inputs, not all counting. '
             'No natural-language or iso-quality resource superiority is inferred.')]
        if tasks.get('joint_replication'):
            confirmation=tasks['joint_replication'][-1]
            blocks.insert(-1,('p',f"Unchanged seed7/8 confirmation: "
                f"{sum(x['joint_credit_gate_passed'] for x in confirmation['comparisons'])} of2 declared joint-versus-local gates pass. "
                'All four arms and delivered-value interventions are reported separately in the appendix; '
                'the new suffix set does not select settings.'))
        pages.append(blocks)
    full_rows=tasks['language_full_sparse']
    full_blocks=[
        ('h1','Ours: the integrated sparse temporal language experiment'),
        ('p','This candidate has no dense language carrier. Each event mixes its embedding with the '
         'previous deep message and traverses contextual key races. Only selected receivers update '
         'their persistent rotating/decaying state and emit values. Time is part of the computation; '
         'inactive receivers are not evaluated on empty ticks.'),
        ('figure',('full_sparse_language_path',164)),
        ('p','The first six-depth, 16-dimensional candidate provides 324 units but updates only six '
         'states per character. Each depth scores two addressed keys; training additionally evaluates '
         'both candidate values to teach hard choices. Unaddressed pools remain dormant. The capacity '
         'experiment doubles available units to 648 while retaining six selected state updates. '
         'Key scoring and counterfactual work still grow and are charged.'),
        ('p','Checks establish causal predictions, identical chunked execution, precise clocks at '
         '10M positions, equality of training forward values and winner-only inference, learned '
         'key/value/memory gradients and conserved route credit. The smoke fit learns, but is not a '
         'quality benchmark. Completed data, depth and memory interventions test progress before larger promotion.'),
        ('small','Fixed observed-character pools, bounded delays and six sequential event depths; '
         'no learned topology or complete frontier-language claim. Interior arrival-time derivatives '
         'and counterfactual score surrogates have distinct scope. FLOPs include teaching alternatives '
         'and optimizer work; representative sparse traces do not certify whole-run instruction or '
         'energy counts. Theory §§299–302; source: sleeping_machines/sparse_race_language.py.')]
    depth_rows={r['args']['depth']:r for r in tasks['episodic_language']
        if r['args']['memory']=='receiver' and r['args']['fit']==2048
        and r['args']['payload']==32 and r['args']['seed']==6}
    if 6 in depth_rows and 8 in depth_rows:
        six,eight=depth_rows[6],depth_rows[8]
        extra=eight['work']['cpu_emulator']['total_training_unit_special_flops']/six['work']['cpu_emulator']['total_training_unit_special_flops']-1
        full_blocks.insert(-1,('p',f"A matched payload-32 / 2K depth screen improves "
            f"{six['final']['dev']['bpc']:.3f} to {eight['final']['dev']['bpc']:.3f} development bpc "
            f"from six to eight receiver blocks, for {100*extra:.1f}% more fitting work. "
            'The current eight-block KV variant adds historical races, giving sixteen selection '
            'steps after warmup. Fixed data/passes/seed do not isolate depth from increased capacity; '
            'this is exploratory evidence, not a scaling law.'))
    pages.append(full_blocks)
    pages.append([
        ('h1','Ours: queries, memory and context'),
        ('p','Race attention specifies how a candidate wins; memory organization specifies what '
         'the candidates represent. The integrated language model races between compressed '
         'persistent receivers. It does not replace a Transformer token KV cache entry for entry.'),
        ('figure',('language_queries_and_memory',164)),
        ('p','At each depth, a learned map turns the incoming message into a query. A candidate key '
         'combines its learned prototype and a read of its retained state. Query–key compatibility '
         'sets an exponential clock rate; the first arrival wins with the corresponding softmax '
         'probability. The observed character addresses two candidates per depth. Queries cannot '
         'search arbitrary past-token keys in this construction. Winner-only delivery also differs '
         'from a deterministic softmax-weighted sum, although its one-step expectation equals that sum.'),
        ('table',(['Model / storage formula','Raw state / one stream','Forward history'],[
            ['Ours: width 16, 324 states + clocks + last message','22.84 KiB','Carried until stream reset'],
            ['Ours: width 32, 324 states + clocks + last message','43.16 KiB','Carried until stream reset'],
            ['Transformer: conceptual FP32 KV, 4 layers × 256 positions × width 256','2,048 KiB','At most 256 characters'],
        ],[80,38,56])),
        ('p','The raw width-32 state is about 47× smaller than this conceptual KV allocation and '
         'does not grow with history length. This is a storage-formula comparison, not matched recall '
         'capacity or measured total RAM: ours compresses history, whereas KV entries retain separate '
         'position-addressable representations. The saved Transformer actually recomputes windows '
         'without an implemented KV cache. Its 256-character window is a model setting, not an '
         'intrinsic dataset limit. Ours retains forward state beyond its 16-character training-credit horizon.'),
        ('small','Ours raw bytes = 324 × (4d + 8) + 4d; conceptual Transformer KV bytes = '
         '2 × 4 × 256 × 256 × 4. Excludes weights, gradients, optimizer, activations, object/index '
         'overhead and traffic. Long-range recall and comparable-quality memory advantages remain '
         'to be measured. Recurrent compression resembles the memory organization of selective '
         'state-space models (Mamba, Gu & Dao, arXiv:2312.00752); our hard temporal races and '
         'counterfactual route teacher are separate mechanisms. Compression is not required by '
         'races: a separate integrated per-position KV experiment retains historical entries and '
         'tests sparse value delivery. Theory §§308–310.')])
    if full_rows:
        pages.append([
            ('h1','Ours: completed integrated-language stages'),
            ('table',(['Ours: fit / payload / pool','Development bpc ↓','Fitting GFLOPs ↓','Capacity / selected states'],[
             [f"{row['args']['fit']:,} / {row['args']['payload']} / {row['args']['pool']}",f"{row['final']['dev']['bpc']:.3f}",
              f"{row['work']['total_training_arithmetic_flops']/1e9:.2f}",
              f"{row['capacity_units']} / {row['args']['depth']}"] for row in full_rows],[44,38,44,48])),
            ('p','These are the integrated model stages, with identical cold development targets. '
             'Each character selects one unit at each depth; addressed alternatives teach the races. '
             'Capacity and selected activity are different counts. The fitting ledger includes '
             'counterfactual values, backward, clipping and Adam.'),
            ('p','Quality, data efficiency and work must be judged together. Completed earlier carrier '
             'results remain preserved. Comparing these models also changes payload size, capacity '
             'and truncated credit, so a score difference does not isolate one mechanism. '
             'No official test or energy measurement is implied.'),
            ('small','One seed; observed-character index; no statistical expert. Whole-stream candidate '
             'scores, selected updates and teaching visits are recorded. Arithmetic is a representative '
             'saved-parameter extrapolation; sparse optimizer activity depends on the actual input '
             'and credit history. RNG, indexing and memory traffic are separate. Source: '
             'experiments/results/parallel_language/local_full_sparse_language_*.json.')])

    if race_rows:
        pages.append([
            ('h1','Ours: matched indexed language retrieval'),
            ('p','Completed development-only results. The unchanged carrier is fitted jointly with '
             'retrieval maps. Race and softmax arms share the index, payload and initialization. '
             'These scores do not establish official-test or frontier superiority.'),
            ('table',(['Ours: attention / fit','Development bpc ↓','Fitting TFLOPs ↓','Value deliveries / query ↓'],[
             [row['args']['attention']+f" / {row['args']['fit']:,}",
              f"{row['final']['dev']['bpc']:.3f}",
              f"{row['work']['total_training_arithmetic_flops']/1e12:.3f}",
              f"{row['final']['dev']['delivered_values']/max(1,row['final']['dev']['retrieval_queries']):.2f}"]
             for row in race_rows],[60,36,37,41])),
            ('p','Winner delivery limits forward value messages, while counterfactual score teaching '
             'visits shortlisted alternatives during training. The retrieval index is supplied by '
             'observed character identity; candidate quality is an empirical question. All six carrier '
             'layers still execute. Four winners need not be cheaper than short candidate lists.'),
            ('small','Identical cold development intervals, first target excluded; frozen development '
             'weights with a fixed race noise stream; 64-character truncated credit. FLOPs are '
             'representative saved-parameter complete-step estimates. Candidate occupancy is not '
             'traced exactly throughout the fit; result files also include a separately labelled '
             'synthetic saturated-index scenario. RNG and emulator memory traffic are additional. '
             'Source: experiments/results/parallel_language/local_indexed_language_*.json.')])

    pages.append([
        ("h1","A demonstrated advantage: generalization with less work"),
        ("p","The retrieval and phase computations preserve useful rules when the evaluation extends beyond "
         "the fitting examples. The saved compact Transformer and LSTM controls use the same synthetic "
         "evaluation targets. <b>Higher and further left is better:</b> more accurate answers from less "
         "estimated inference work. All points use one logical operation ledger."),
        ("figure",("consolidated_work_frontiers",174)),
        ("table",(["Ours: event computation","Held-out capability","Estimated work per query"],[
         ["Periodic path; 69 learned scalars",f"{phase['correct']:,}/{phase['n']:,} unseen triples",f"{phase_work:,.0f} logical operations"],
         ["Two-layer carrier + hard pointer","100% at four times context",f"{recall_work:,.0f} logical operations"],
        ],[62,57,55])),
        ("p","The phase rule costs 188 logical operations per triple; the saved LSTM and Transformer "
         "cost 104,518 and 155,592 and score 1.95% and 3.60%. At four times the recall context, the event "
         "encoder plus learned pointer scores 100% at 728,602 operations; both compact controls score "
         "7.42% at 2.24M and 4.46M operations. The phase model receives a periodic representation with "
         "period 17, and retrieval has a pointer mechanism. These useful priors explain the task advantage "
         "and are part of what must transfer to harder tasks."),
        ("p","<b>Learning work is also selective.</b> The periodic teacher makes 29,003 mistaken-example updates "
         "and 145,015 learned-scalar update visits. The dense arithmetic controls make 4,800 Adam steps: "
         "92.2M parameter visits for the LSTM and 132.5M for the Transformer. These count parameter updates, "
         "excluding optimizer state and backward arithmetic; they are not training FLOPs or joules."),
        ("small",f"Arithmetic: 1,473 fitting triples, a 200-epoch budget, all 3,440 unseen triples; supplied period 17. "
         f"The phase-only path stops after {tasks['phase_only']['actual_epochs']} passes, when an entire fitting pass makes no updates. Recall: "
         "4,000 pointer-fitting examples plus 512 neural-fitting examples; the dense controls receive all 4,512 "
         "examples for eight epochs. Width 32 and two generic layers where present, seed 6, one small dense setting. "
         "Work is an analytic logical-operation estimate, including configured vector maps, routers, scans, "
         "normalization, clock candidates and pointer search. These inference counts are not measured joules or "
         "backward/optimizer counts. Full work definitions appear in the evidence appendix.")])

    pages.append([
        ("h1","A demonstrated advantage: learning temporal structure"),
        ("p","Event chains learn timing patterns and compose recognizable parts into ordered structures. "
         "The preserved five-run results show accurate recognition and strong sample efficiency. The right "
         "panel compares distinct examples rather than incompatible activity counters."),
        ("figure",("supremacy_map",174)),
        ("table",(["Preserved comparison","Ours","Transformer","Fitting protocol"],[
         ["Timing patterns","99.95–100%; five runs","99.60–99.80%; two runs","200k examples once / 1M with relative-time bias"],
         ["Shared-motif composition","99.00–99.93%; five runs","99.60–99.85%; two runs","40k examples once / 1M with relative-time bias"],
         ["Depth-four order","99.90–100%; five runs","98.95–99.05%; two runs","Same 40k distinct examples; one pass / 50 passes"],
         ["Depth-three order at 2k examples","99.73–99.93%; five runs","33.25–40.80%; two runs","One pass / repeated fitting on the same 2k examples"],
        ],[45,41,41,47])),
        ("p","The strongest depth-four comparison has approximately ten times fewer classification errors "
         "despite the event learner seeing each fitting example once. The sample-efficiency curve makes the "
         "next question concrete: can learned embeddings and broader event representations retain that advantage "
         "when the inputs no longer supply known temporal parts?"),
        ("small","Completed E35, E34, E53/E54 and E36 files. Ranges describe the recorded runs, not confidence "
         "intervals. Architectures and optimization differ; synthetic evaluation sets were reused during research. "
         "Event deliveries remain activity measurements. Total arithmetic, backward and optimizer work require "
         "a declared ledger; activity divided by dense MACs is not a training-cost or power ratio.")])

    pages.append([
        ("h1","Why asynchronous, sparse computation matters"),
        ("p","Persistent local state can retain experience without repeatedly reconstructing a whole history. "
         "An asynchronous node updates when useful information arrives. A hard race emits one chosen message. "
         "These mechanisms create a path to spending less computation and moving less data per useful answer. "
         "On an event-oriented processor, inactive nodes and communication links could remain idle."),
        ("h2","Spend the next unit of work where it helps"),
        ("p","A larger budget can buy a longer memory, better retrieval, a deeper representation for difficult "
         "inputs or more local adaptation. The development rule is to measure the held-out improvement from "
         "each choice and allocate work to the most useful one. The phase and pointer results show why an "
         "appropriate computation can be much cheaper than a generic dense approximation. The scaling "
         "program tests how much of this flexibility survives when that computation must itself be learned."),
        ("table",(["Architecture","Fitting known sequences","Generating or processing a stream"],[
         ["LSTM","Gates depend on prior hidden state; recurrent work is sequential","Retained hidden state; dense gate updates per token"],
         ["Transformer","Causal attention permits sequence-parallel fitting","Sequential token generation with a key/value cache; adaptation is possible"],
         ["Ours: Sleeping Machines","Affine event memory supports parallel scans once incoming values/times are known","Retained local state and delayed-message queue; only due arrivals execute"],
        ],[38,68,68])),
        ("p",f"Parallel fitting and sequential generation are compatible. The precise-clock language scan "
         f"gives a {tasks['parallel_contract']['measured_step_speedup']:.2f}× complete-step CPU speedup "
         "against serial execution of the same width-256 model, including backward, clipping and Adam. "
         "Predictions, states, gradients, chunk boundaries and causality are checked. This follows a "
         "bounded-delay schedule; arbitrary reordering networks require their own contract."),
        ("p",'Parallel recurrent computation also appears in '
         '<a href="https://proceedings.mlr.press/v119/katharopoulos20a.html">linear attention</a> and '
         '<a href="https://arxiv.org/abs/2312.00752">selective state-space models</a>. '
         'EventSSM already processes asynchronous events with scans. These are important controls. '
         'The distinctive hypothesis here is the combination of learned timing, sparse communication, '
         'credit to alternatives and independent compute budgets; it must earn its advantage empirically.'),
        ("small","Primary FLOPs describe the declared event algorithm, including required vector maps, "
         "candidate computation, scans, backward and optimizer updates. Simulator padding and dispatch are "
         "separate implementation overhead. Physical power also depends on memory, queues, communication "
         "and hardware utilization. Demonstrated work savings and projected power savings are distinguished; "
         "total device joules have not yet been measured.")])

    pages.append([
        ("h1","A mathematical foundation for trainable computation"),
        ("table",(["Principle","What it enables"],[
         ["Stable transport through depth","The reversible packet/memory construction preserves conditional value and credit norms under its stated operator and boundary assumptions. Readout visibility, routing and optimization remain separate requirements."],
         ["Active communication support","Inputs need causal paths through which to interact. A context channel supplies joint information when sparse packets leave local groups disconnected."],
         ["Credit to unrealized alternatives","A losing payload or timing choice can show how a different route would change the outcome, while forward computation remains a hard race."],
         ["Periodic state as an isometry","Learned rotations/reflections have unit-magnitude occurrence derivatives. Their composition supports reusable arithmetic instead of a table of observed tuples."],
         ["Certified composition","Target-constrained min/max composition of phase errors certifies the fitted modular rule across all 4,913 possible tuples; exhaustive checking confirms it."],
         ["Natural supervised credit","Categorical and event likelihoods both credit predicted sufficient statistics minus observations. Silence enters through integrated exposure."],
         ["Useful optionality","Reserve consists of distinct, attainable future corrections under a causal work budget. Reachability and transferable learning matter alongside immediate loss."],
         ["Statistically useful credit","Expected improvement must overcome the curvature cost of fitting noise. Cross-example teacher agreement separates reproducible correction from raw gradient magnitude."],
        ],[57,117])),
        ("h2","From mathematics to an engineering discipline"),
        ("p","The theory connects representation, topology, clocks and optimization. Expressivity describes what "
         "a network can compute; transport describes whether information and credit survive; the objective describes "
         "what the teacher asks it to learn. These pieces must agree. The periodic certificate is one concrete case "
         "where the formal model explains and verifies a learned computation."),
        ("p","A common implementation makes these principles reusable across tasks. Efficient primitives can own "
         "a computation when its structure is known; a deep carrier can learn representations when it is not. "
         "The research objective is to combine this flexibility with affordable route discovery and increasingly "
         "capable models."),
        ("small","Formal derivations and their assumptions are indexed in the project's theory notes. Conditional "
         "stability and a certificate for a fitted rule do not establish global optimizer convergence or a scaling law. "
         "The program builds on established deep spiking and sparse conditional computation; its focus is the "
         "combination of useful temporal operators, hard causal routing and counterfactual learning. Related survey: "
         '<a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full">'
         'Direct training of deep spiking networks</a>.')])

    pages.append([
        ("h1","Potential grounded in the completed evidence"),
        ("p","The objective is capable models that spend computation where it improves an answer. The saved "
         "structured-task results and learned stream pilots provide specific starting points. Moving from "
         "those mechanisms to frontier prediction requires useful representations, longer context and "
         "measured quality at a fixed total resource budget."),
        ("table",(["Opportunity","Present foundation","What would establish the larger case"],[
         ["Compact prediction and memory","Persistent learned state; pointers generalize to longer contexts.","Competitive held-out language quality, retention and complete fitting/inference cost."],
         ["Continuous perception","Learned temporal speech representations and composition.","Full speech/vision tests and confidence-based early decisions at measured latency."],
         ["More useful training per budget","Structured-task sample efficiency; exact causal parallel scans.","Quality improvements at equal total fitting work, including route discovery."],
         ["Dormant skills and selective depth","Hard race routing and compute-allocation theory.","Learned marginal work allocation that outperforms fixed allocation on real tasks."],
         ["Mobile and industrial autonomy","Local state and event-triggered updates.","Device joules, memory traffic and task quality measured on deployable implementations."],
        ],[40,62,72])),
        ("h2","Why the hardware and economic implications could be large"),
        ("p","If comparable quality needs less total training and inference energy, a fixed power and capital "
         "budget can support more capable models, more research or continuous adaptation on robots, phones "
         "and instruments. Persistent local state and selective communication would favor hardware that "
         "handles message delivery, queues and memory efficiently. These are conditional consequences of "
         "measured savings, rather than savings inferred from event counts."),
        ("p","The near-term experiment asks where the next unit of computation helps: longer memory, "
         "richer content transformations, retrieval, depth or local learning. Held-out gains and complete "
         "work determine promotion. A successful scaling result must show that those gains continue across "
         "independently fitted sizes and budgets.")])

    scaling=sorted(tasks["language_scaling"],key=lambda row:row["parameters"])
    memory=sorted(tasks["language_memory"],key=lambda row:row["args"]["memory_profile"])
    memory+=sorted(tasks["language_selective"],key=lambda row:row["args"]["memory_profile"])
    memory_blocks = ([
        ("h2","From longer memory to selective content"),
        ("table",(["Ours: memory variant","Parameters","Development bpc ↓","Fitting GFLOPs ↓"],[
         [("Input gates / " if row['args'].get('content_memory') else "Constant / ")+
          row['args']['memory_profile'].replace('_',' '),f"{row['parameters']:,}",
          f"{row['final']['dev']['bpc']:.3f}",f"{row['work']['total_training_arithmetic_flops']/1e9:,.2f}"]
         for row in memory],[64,32,37,41])),
        ("small","Identical width, seed, data, four passes and 64-character credit horizon. Constant-memory "
         "arms vary initial timescales/frequencies. Input-gated arms add content-dependent write/forget controls "
         "(0.50% more parameters, 1.41% more fitting arithmetic). Longer decay alone worsens this fit. "
         "These are single-seed development comparisons; complete numerical contracts precede training.")
    ] if memory else [("p","The inherited event initialization leaves individual modal timescales at only "
         "a few character intervals after fitting. A matched small ablation now tests longer decay times and "
         "resolved temporal periods before committing to the large run. Modal decay is a diagnostic, not a "
         "hard bound on the complete stack's context.")])
    pages.append([
        ("h1","Ours: language learning and staged scale-up"),
        ("p","These learned models use character embeddings, gated residual content transformations and "
         "persistent rotating/decaying state. Temporal modes encode relative token distance. No statistical "
         "count, copy or word experts provide their predictions. The goal is competitive quality from a "
         "learned backbone before claiming a language compute advantage."),
        ("figure",("language_capacity_scaling",166)),
        ("table",(["Ours: width","Parameters","Development bpc ↓","Total fitting GFLOPs ↓"],[
         [str(row['args']['width']),f"{row['parameters']:,}",f"{row['final']['dev']['bpc']:.3f}",
          f"{row['work']['total_training_arithmetic_flops']/1e9:.2f}"] for row in scaling],[35,42,45,52])),
        ("small","All curves use six layers, seed 6, 131,072 fitting characters, four passes and the same "
         "8,191 cold-context development targets. Credit is truncated every 64 characters; memory persists. "
         "FLOPs include forward/loss, backward, clipping and Adam; special functions are reported separately "
         "in each result. These runs vary capacity at equal data/passes, rather than equal compute, and "
         "establish neither a scaling law nor official-test superiority."),
    ]+memory_blocks)

    if tasks['language_representation']:
        audit=tasks['language_representation'];gated=audit['rows'][-1]['interventions']
        pages.append([
            ('h1','Ours: content and memory in fitted language models'),
            ('p','An event carries information about its input. The learned vector is a transformation of '
             'incoming content and persistent state, with a residual path and an output gate. The new '
             'candidate also gates memory writing and forgetting from incoming content. These checks '
             'measure whether the fitted predictions use those paths.'),
            ('figure',('language_content_memory_audit',170)),
            ('p',f"Resetting all history before each character preserves its current embedding and learned "
             f"content transformations, but raises the gated model's development loss from "
             f"{gated['full']['bpc']:.3f} to {gated['reset_history_every_token']['bpc']:.3f} bpc. "
             f"Zeroing incoming embeddings raises it to {gated['zero_incoming_embeddings']['bpc']:.3f}. "
             'Both present input and earlier messages contribute to prediction.'),
            ('h2','Selective retention adds a useful control'),
            ('p','The two input gates start at one, preserving the constant-memory model exactly at '
             'initialization. During fitting they learn different write strengths and forgetting factors '
             'for different incoming vectors. A factor below one slows decay; above one accelerates it. '
             'The controls are known from the causal previous layer, so serial execution and parallel '
             'affine scans retain their checked outputs and teachers.'),
            ('p','In the matched small fit, gates improve 2.643 to 2.587 bpc for 0.50% more parameters '
             'and 1.41% more fitting arithmetic. This is a local quality/work improvement, with one seed. '
             'All six layers still execute for every character; dormant-unit scaling remains a separate target.'),
            ('small','Frozen selected checkpoints, identical 8,191 cold development targets, no training '
             'or official-test reads. These interventions disrupt a trained model; they establish fitted '
             'dependence, not the quality of retrained ablated architectures or lossless storage. Source: '
             'parallel_language/local_language_representation_20260930T162337Z.json.')])

    if tasks['language_scaleup']:
        rows=sorted(tasks['language_scaleup'],key=lambda row:(row['args']['fit'],row['args']['width']))
        larger_blocks=[
            ('h1','Ours: larger language development stages'),
            ('p','Each stage fits independently from initialization. Data, capacity and memory controls are '
             'declared below. Development selects weights within the fixed four-pass budget; ongoing '
             'training logs are never substituted for a completed result.'),
            ('table',(['Ours: memory / width','Fit characters','Parameters','Development bpc ↓','Fitting TFLOPs ↓'],[
             [('Input gates' if row['args'].get('content_memory') else 'Constant')+f" / {row['args']['width']}",
              f"{row['args']['fit']:,}",f"{row['parameters']:,}",f"{row['final']['dev']['bpc']:.3f}",
              f"{row['work']['total_training_arithmetic_flops']/1e12:.3f}"] for row in rows],[46,34,29,34,31]))]
        capacity=[row for row in rows if row['args']['fit']==131072 and row['args']['width']==256]
        if capacity:
            smaller=next(row for row in tasks['language_selective'] if row['args']['memory_profile']=='inherited')
            larger=capacity[-1]
            extra=smaller['final']['dev']['bpc']-larger['final']['dev']['bpc']
            ratio=larger['work']['total_training_arithmetic_flops']/smaller['work']['total_training_arithmetic_flops']
            larger_blocks += [
                ('figure',('language_compute_choices',170)),
                ('p',f"On the identical 131K-character/four-pass screen, widening the gated model buys "
                 f"{extra:.3f} bpc for {ratio:.2f}× the total fitting arithmetic. Adding input gates at "
                 'width 128 instead improves 0.057 bpc for 1.41% more arithmetic. This makes the '
                 'allocation question quantitative: measure useful correction before spending broadly '
                 'on width. These are finite, single-seed interventions, rather than a scaling law.')]
        larger_blocks += [
            ('p','The fixed-capacity data comparison and fixed-data capacity comparison answer different '
             'questions. Equal passes and data do not imply equal compute. The pipeline checks finite '
             'learning, trained value blocks, complete work and source provenance before promotion. '
             'A development gain is not an official-test or frontier claim.'),
            *([('p',"Count reference (Theory §§376–380): on the same 8,191 development targets, interpolated "
              f"Kneser–Ney counts of the same fitting characters score {count_reference_bpc(131072):.3f} bpc at 131K and "
              f"{count_reference_bpc(1048576):.3f} at 1M, with one counting pass and no gradient work. Every completed "
              'earlier fit without count-carrying receivers from 2K to 1M characters is above this bar. '
              'The fixed-step estimator analyzed in Theory §377 has a variance floor; this is not an impossibility '
              'theorem for learned recurrent gates or short credit. It motivates testing '
              'count-carrying receivers with escape races and a learned base measure. Counts are reference '
              'predictors, not neural controls or a large-data comparison.')] if count_reference_bpc(1048576) is not None else []),
            ('small','Precise clocks; float32 payloads; causal persistent state; 64-character credit horizon. '
             'Special functions, evaluation passes and physical traffic are separate from the arithmetic '
             'ledger. One seed, no statistical experts. Source: experiments/results/parallel_language.')]
        pages.append(larger_blocks)

    pages.append([
        ("h1","What establishes the larger advantage"),
        ("p","The ambition is a common model family whose strongest mechanisms remain useful as tasks, data and "
         "capacity grow. Arithmetic and retrieval retain their demonstrated strengths in the consolidated "
         "implementation. The strongest language mixture is specialized; the generic backbone still needs to "
         "demonstrate competitive learned representations. Character and subword-token budgets must be distinguished."),
        ("table",(["Objective","Decisive evidence"],[
         ["Generic language scaling","Train a learned event backbone that owns the prediction. Scale through declared data budgets with matched Transformer, recurrent and state-space references; record loss, capacity, complete training work, memory traffic, time and joules."],
         ["Preserve capabilities","Repeat established generalization and sample-efficiency results within the common model family, with task-appropriate depth and explicit resource accounting."],
         ["Strong real-event recognition","Accurate speech and event-camera decisions on complete held-out benchmarks; calibrated confidence and time-to-answer."],
         ["Learn routes and representations at scale","Reliable deep credit and useful counterfactual alternatives as width, depth, memory and data increase."],
         ["Efficient persistent operation","Maintain local state and pending messages across queries, preserving causal predictions while reducing repeated work."],
         ["Lower total energy at useful quality","Measure training and inference joules, memory traffic, latency and communication under declared hardware and quality targets."],
        ],[57,117])),
        ("p","Success on these dimensions would turn the current task-level advantages into a broader foundation "
         "for frontier models. The project's distinctive resources—timing, local memory, hard selection and credit "
         "to alternatives—remain the guide for architecture and learning.")])

    speech_runs=[("Original eight-layer parent",tasks["shd_scaled"])]
    for key,label in (("shd_warm","Warm larger-data continuation"),
                      ("shd_fine","Fine source messages"),
                      ("shd_phase","Fine sources + temporal phase"),
                      ("shd_local","Fine/phase learner; parent frozen")):
        if tasks[key] is not None:speech_runs.append((label,tasks[key]))
    state_residual=tasks["shd_state_residual"]
    audit_sentence=""
    if tasks["shd_state_summary"] is not None:
        audit=tasks["shd_state_summary"]["disjoint_audit"]
        audit_sentence=(f" On {audit['n']} disjoint utterances from the same held speakers, accuracy improves "
            f"from {100*audit['parent_correct']/audit['n']:.2f}% to {100*audit['residual_correct']/audit['n']:.2f}%.")
    if state_residual is not None:
        best_row=max(state_residual["curve"],key=lambda r:r["dev"]["correct"])
        speech_runs.append((f"Parent + parallel six-block residual; pass {best_row['epoch']}",{"final":best_row}))
    single_paired=tasks["shd_single_paired"]
    if single_paired is not None:
        selected=min(single_paired["curve"],key=lambda r:(-r["dev"]["correct"],r["dev"]["nll"]))
        speech_runs.append((f"Single six-block temporal encoder; pass {selected['epoch']}",{"final":selected}))
    for key,label in (("shd_calibrated_d6","Calibrated six-block continuation"),
                      ("shd_calibrated_d12","Bounded twelve-block encoder"),
                      ("shd_observer_depth","Directional twelve-block encoder")):
        if tasks[key] is not None:speech_runs.append((label,tasks[key]))
    if tasks['shd_selected_prefix'] is not None:
        speech_runs.append(('Selected single six-block prefix; trained with twelve blocks',tasks['shd_selected_prefix']))
    best_label,best_speech=max(speech_runs,key=lambda entry:(pooled(entry[1]),
        -entry[1]['final'].get('dev',{}).get('nll',float('inf'))))
    speech_rows=[]
    display_runs=speech_runs
    if state_residual is not None:
        display_runs=[speech_runs[0]]+[(f"Parent + six-block temporal residual; pass {r['epoch']}",{"final":r})
            for r in state_residual["curve"]]
        if single_paired is not None:
            display_runs=[speech_runs[0],(f"Parent + six-block residual; pass {best_row['epoch']}",{"final":best_row})]
            for label,run in speech_runs:
                if label.startswith(("Single six","Selected single","Calibrated","Bounded","Directional")):display_runs.append((label,run))
    for label,row in display_runs:
        parts=[row["final"]["dev"]] if "dev" in row["final"] else [row["final"][k] for k in ("dev_original","dev_additional")]
        correct=sum(p["correct"] for p in parts)
        speech_rows.append([label,f"{correct}/512",f"{100*pooled(row):.2f}%"])
    pages.append([
        ("h1","Appendix A. Deep event recognition"),
        ("p",f"The strongest completed speech result in the model family is <b>{100*pooled(best_speech):.2f}%</b> "
         f"on 512 private held-speaker utterances ({best_label.lower()}). "
         "Published official-test results below use a different partition; they are reference targets."),
        ("table",(["Ours: private development configuration","Correct","Accuracy ↑"],speech_rows,[108,32,34])),
        ("figure",("e143_temporal_residual_learning" if state_residual is not None else "e139_source_information",152)),
        ("p",("A six-block width-128 temporal encoder learns corrections while the inherited eight-layer parent "
         "stays frozen. It adds 395,814 parameters to the parent's 53,296. Signed modal states, nonlinear gates "
         "and residual vectors learn from all source identities and original event times before causal pooling. "
         "This is a larger parallel model, not fourteen sequential layers; completed-utterance supervision "
         "does not yet teach calibrated early answers."+audit_sentence if state_residual is not None else
         "Raw channel identities and original source times now enter learned vector messages before "
         "coalescing. Signed temporal rotations extend the receiver memory, with the old mean as its zero-phase "
         "case. Initial predictions, hard winners and clocks match the trained parent exactly. Each packet "
         "still emits one winning vector and delay; source and temporal transformations add measured work.")),
        ("table",(["Published reference; official-test protocol","Reported accuracy ↑"],[
         ["Ours: full official SHD comparison","Pending"],
         ["EventSSM: asynchronous learned state-space layers","95.9%"],
         ["S7: input-dependent temporal state","96.3%"],
        ],[131,43])),
        ("small","Our private sample uses training-file speakers 3/6; official test accuracy is unmeasured. "
         + ("The temporal residual uses three passes and a fresh optimizer; its larger capacity and budget "
            "are not a matched single-factor comparison. Its listed score selects the best private-development "
            "epoch; the curve shows all three. " if state_residual is not None else "") +
         "The original parent was fitted on 4,096 examples. Continuation timers exclude loading; the residual "
         "timer includes it. Both include evaluation and are not energy measurements. One seed. References: "
         '<a href="https://arxiv.org/html/2404.18508v2">EventSSM</a>, '
         '<a href="https://arxiv.org/html/2410.03464v1">S7</a>. '
         "These published scores were checked against the original papers; they are targets for a full "
         "official-test comparison, not scores on our private split.")])

    if tasks["shd_state_ablation"] is not None and tasks["shd_state_summary"] is not None:
        summary=tasks["shd_state_summary"]
        reset=tasks["shd_state_ablation"]["rows"]
        reset_rows=[]
        for key,label in (("trained","All learned parameters retained"),("reset_clocks","Only hidden clocks reset"),
            ("reset_source_embedding","Only source vectors reset"),("reset_state_stack","Only state stack reset"),
            ("reset_modal_dynamics","Only decay/frequency parameters reset")):
            part=reset[key]["held"]
            reset_rows.append([label,f"{part['correct']}/512",f"{100*part['accuracy']:.2f}%"])
        pages.append([
            ("h1","Appendix A (continued). Learned timing and transfer"),
            ("p","<b>Learned delays contribute to the answer.</b> Restoring the hidden clocks to their initial "
             "values, with source vectors, state/value maps and the trained classifier retained, loses 13 correct "
             "answers. Timing changes the temporal interactions used by the representation; it performs computation."),
            ("figure",("e145_learned_timing_and_transfer",174)),
            ("table",(["Ours: same trained readout; subsystem reset","Correct","Accuracy ↑"],reset_rows,[108,32,34])),
            ("p","The temporal stack and source vectors also learn useful coordinated representations. Resetting "
             "the stack loses 50 correct answers; resetting sources loses 35. Decay/frequency resets change one "
             "decision. These changes depend on the fitted solution's coordination; their effects cannot be added "
             "or treated as a matched comparison of retrained architectures."),
            ("p","The 657-utterance audit is disjoint from fitting and the development sample and uses the selected "
             "checkpoint unchanged. It gains 62 correct answers and loses 15, for a net improvement of 47. "
             "Both samples use the same two held training speakers. Official-test and additional-speaker "
             "generalization are the next evaluation targets."),
            ("h2","A stronger mathematical account of routing"),
            ("p","An affine packet summary can preserve both the final state and the average of raw temporal "
             "states, including their teachers. This permits richer pooling before expensive nonlinear maps. "
             "A second derivation shows why many almost-equal delays may offer little usable choice: their "
             "effects point in nearly the same direction. Diverse payloads and temporal modes, sufficient delay "
             "spread and downstream visibility determine useful route reserve."),
            ("small","One exploratory run: 449,110 total parameters, inherited parent plus a trained six-block "
             "encoder. Three passes use about 33 minutes including preparation/evaluation and 1.79 GiB peak RSS; "
             "the guarded host retains at least 10,361 MiB sampled available memory. Complete physical work and "
             "joules remain unmeasured. Formal statements and numerical contracts are in THEORY §§226–236.")])
    elif tasks["shd_exchange"] is not None:
        state=tasks["shd_exchange"]["state"]["final"]
        packets=tasks["shd_exchange"]["packets"]["final"]
        ablation=tasks["shd_exchange"]["ablation"]
        query_rows=[
            ["Full retained-state query","140,428",f"{100*state['fit']['accuracy']:.1f}%",f"{100*state['held']['accuracy']:.1f}%"],
            ["Emitted-packet query","2,188 active",f"{100*packets['fit']['accuracy']:.1f}%",f"{100*packets['held']['accuracy']:.1f}%"],
        ]
        compact_blocks=[]
        if tasks["shd_compact"] is not None:
            rows=tasks["shd_compact"]["rows"]
            learned,frozen=rows["learned"],rows["frozen"]
            for mode,label in (("learned","Compact; learned angles"),("frozen","Compact; fixed angles")):
                last=rows[mode]["curve"][-1]
                query_rows.append([label,f"{rows[mode]['train_parameters']:,}",
                                   f"{100*last['fit_accuracy']:.1f}%",f"{100*last['held_accuracy']:.1f}%"])
            compact_blocks=[("p",f"A rank-16 bank/channel/class query uses 4,968 decoder parameters. Learned exchanges "
                f"reach {100*learned['curve'][-1]['held_accuracy']:.1f}% held accuracy versus "
                f"{100*frozen['curve'][-1]['held_accuracy']:.1f}% with fixed angles. These compact arms share initial "
                "predictions, calibration, keys, query capacity, examples and optimizer budget. Both embeddings and "
                "queries learn; only angle adaptation is disabled. This isolates useful exchange adaptation under constrained supervision."),
                ("small","Compact/full terminal queries estimate 115,028/138,900 forward MACs, excluding key/exchange work, "
                 "normalization, backward, optimizer and traffic. Parameter compression is much larger than this query-work saving; energy is unmeasured.")]
        pages.append([
            ("h1","Appendix A (continued). Trainable event memory at twelve layers"),
            ("p","A winning key selects one memory bank. Its orthogonal exchange stores and emits vector information "
             "with the winning delay. Conditional packet/state norms survive depth; the completed query reads retained "
             "memory. All twelve exchange layers receive credit."),
            ("figure",("e136_scattering_learning",174)),
            ("table",(["Ours: completed three-pass query","Trainable parameters","Fit: higher is better","Held: higher is better"],query_rows,[63,35,38,38])),
            ("p",f"Resetting learned angles preserves all 1,024 full-query fitting decisions, "
             f"while held accuracy changes from {100*ablation['trained']['held']['accuracy']:.1f}% to "
             f"{100*ablation['all_angles_reset_same_decoder']['held']['accuracy']:.1f}%. This frozen-checkpoint probe shows "
             "angle contribution/coadaptation, not a retrained control. The full-state and packet queries differ in active decoder capacity."),
            *compact_blocks,
            ("small","Exploratory seed 6: 1,024 unaugmented fitting utterances; 512 held training-file speakers; "
             "three passes, width 32. Each model retains 53,296 frozen key parameters from the eight-layer/4,096-fit checkpoint. "
             "Query-fitting examples are a subset of its fitting data. These are not from-scratch or matched continuations. All-layer state queries provide "
             "direct supervision. Official SHD test data and calibrated early decisions remain untested.")])

    if tasks["shd_single_clean"] is not None and single_paired is not None:
        clean_run=tasks["shd_single_clean"]
        selected=min(single_paired["curve"],key=lambda r:(-r["dev"]["correct"],r["dev"]["nll"]))
        model_rows=[]
        for label,part in (("Combined model: parent + temporal correction",best_row["dev"]),
            ("Single encoder: clean head, before continuation",clean_run["conditioned_initial"]["dev"]),
            ("Single encoder: paired head, selected pass zero",single_paired["conditioned_initial"]["dev"])):
            model_rows.append([label,f"{part['correct']}/512",f"{100*part['accuracy']:.2f}%",f"{part['nll']:.3f}"])
        if tasks['shd_selected_prefix'] is not None:
            part=tasks['shd_selected_prefix']['final']['dev']
            model_rows.append(['Single encoder: selected trained prefix',f"{part['correct']}/512",
                f"{100*part['accuracy']:.2f}%",f"{part['nll']:.3f}"])
        transfer_text=""
        timing_text=""
        if tasks["shd_single_audit"] is not None:
            audit=tasks["shd_single_audit"]
            original= audit["rows"]["combined"]
            single= audit["rows"]["single_paired"]
            transfer_text=(f" On the reused 657-utterance disjoint audit, the single encoder reaches "
                f"{100*single['audit']['accuracy']:.2f}% versus {100*original['audit']['accuracy']:.2f}% "
                "for the combined model. This audit is excluded from updates and checkpoint selection.")
            timing_text=(f" One CPU forward evaluation of those utterances takes {single['forward_wall_s']:.2f} s "
                f"for the single encoder and {original['forward_wall_s']:.2f} s for the combined model. "
                "This includes packing/query work and excludes loading; it is one timing observation, not joules.")
        if tasks['shd_selected_prefix'] is not None:
            selected_prefix=tasks['shd_selected_prefix']
            transfer_text=(f" The selected trained prefix reaches {selected_prefix['reused_audit']['correct']}/657 "
                f"({100*selected_prefix['reused_audit']['accuracy']:.2f}%) versus 510/657 (77.63%) for the combined model "
                "on the reused disjoint audit. Audit labels do not choose the checkpoint.")
            timing_text=(f" One CPU forward evaluation takes {selected_prefix['observed_forward_wall_s']:.2f} s for "
                "the selected prefix versus 23.71 s for the combined model. Packing/query included, loading excluded; "
                "one timing observation, not joules.")
        pages.append([
            ("h1","Appendix A (continued). One temporal encoder"),
            ("p","A six-block temporal encoder retains nearly all the combined model's development accuracy "
             "through one ordinary query head. Deployment removes the frozen parent: 395,814 parameters "
             "replace 449,110. Modal states, gated vector messages and winning delays remain. Its weights "
             "inherit earlier encoder training; combined teacher predictions are used only to initialize the head."),
            ("table",(["Ours: private development configuration","Correct","Accuracy ↑","NLL ↓"],model_rows,[98,27,27,22])),
            ("figure",("e152_single_encoder_learning",174)),
            ("p","Fitting the head on clean and transformed speech improves held accuracy by 32 answers "
             "with the temporal features frozen. Its covariance penalty suppresses class-visible nuisance "
             "variation. The right panel compares the heads on the same features; the left shows all subsequent "
             "unrestricted encoder passes."+transfer_text),
            ("p","Training the directional twelve-block extension, then selecting its six-block prefix on development, "
             "retains the combined model's 408 correct answers with lower NLL and 395,814 deployed parameters. "
             "The extra training blocks are removed after their learned contributions reduce held accuracy. "
             "This improves deployment quality/work; it does not establish a positive deep-block accuracy gain."),
            ("small","Seed 6, private train-file speakers 3/6; official-test parity remains unmeasured. "
             "Head fitting uses 6,144 unique fitting utterances: one clean view for the first arm, clean plus "
             "one transformed view for the paired arm. The arms also change regularization and use three/two "
             "encoder passes respectively, so total budgets are not matched. All continuation epochs are plotted; "
             "selection uses development accuracy, then NLL. Extra teacher/cache/head work and inherited fitting "
             "must be charged. Modal/vector maps are locally dense; no empty ticks or event-pair attention are added. "
             "The selected prefix additionally inherits the full twelve-block fitting pass; pruning does not erase "
             "that training cost. Prefix/full choice is post-hoc private-development selection. "
             "Formulae and numerical checks: THEORY §§237–264."+timing_text)])

    if tasks['shd_observer_depth'] is not None and tasks['shd_observer_audit'] is not None:
        depth_rows=[]
        for key,label in (('shd_calibrated_d6','Six-block control'),('shd_calibrated_d12','Twelve blocks: bounded outputs'),
                          ('shd_observer_depth','Twelve blocks: directional units')):
            run=tasks[key];final=run['final']
            depth_rows.append([label,f"{run['deployed_parameters']:,}",f"{100*final['fit']['accuracy']:.2f}%",
                f"{100*final['dev']['accuracy']:.2f}%",f"{final['dev']['nll']:.3f}"])
        audit=tasks['shd_observer_audit'];paired=audit['paired']['its_trained_prefix']
        pages.append([
            ('h1','Appendix A (continued). Making depth useful'),
            ('p','Identity growth preserves the classifier and old teachers while added output maps receive '
             'label credit. They must also change useful features. A tightly bounded twelve-block extension '
             'learns weights but changes no audit decisions when its six appended blocks are removed.'),
            ('table',(['Ours: one matched fitting pass','Parameters','Fit accuracy ↑','Private accuracy ↑','NLL ↓'],depth_rows,[68,29,26,29,22])),
            ('figure',('e164_depth_use_and_work',174)),
            ('p','Directional conditioning normalizes temporal-state features before their output map. An '
             'invertible coordinate change rescales classifier-sensitive directions and preserves hidden null '
             'directions for later computation. The fixed transform folds into an ordinary map at deployment. '
             'Initial outputs and old teachers remain exact; fitting replays verify the actual proposed update.'),
            ('p',f"The directional model reaches {audit['rows']['directional_d12']['audit']['correct']}/657 on the reused audit. "
             f"Removing its six appended blocks changes {paired['changed_predictions']} predictions: "
             f"{paired['full_only_correct']} are correct only with the blocks and {paired['reference_only_correct']} only without them. "
             'This measures fitted contribution with the trained prefix/head retained; it is not a retrained architecture comparison.'),
            ('small','All arms inherit the paired-head checkpoint and use 6,144 fitting utterances, the same order, '
             'channel/time transformations and one encoder pass. Old-group LR is 0.0000203125; directional new groups '
             'use 0.0001953125 from fitting-only replay. Changed normalization and update coordinates form one '
             'intervention. New blocks initially add 36 ms latency; labels supervise completed untimed utterances. '
             'Audit reuse is explicit; official-test parity remains unmeasured. CPU points are one warmed observation '
             'per model, packing/query included and loading excluded; energy is unmeasured. Weight-coordinate folding, '
             'all source work and added depth must be charged during training. Local maps remain dense, with no empty '
             'ticks or event-pair attention. Theory §§249–264.')])

    coverage=[]
    breadth={row["task"]:row for row in tasks["breadth_work"]["rows"]}
    total_work={row["task"]:row for row in tasks["training_work"]["rows"]}
    for task,label in (("language","Text8"),("market","Market event prediction"),("temporal","Temporal composition"),
                       ("mnist","MNIST"),("dvs","Event-camera gestures")):
        row=breadth[task]
        def score(metric):
            if task=="language":return f"{metric['nll']/math.log(2):.3f} bpc"
            if "accuracy" in metric:return f"{100*metric['accuracy']:.1f}%"
            return f"{metric['nll']:.3f} nats/event"
        direction = "bpc ↓" if task=="language" else "nats/event ↓" if task=="market" else "accuracy ↑"
        coverage.append([label+"<br/>"+direction,score(row["common_metric"]),score(row["reference_metric"]),
             f"{row['common_forward_map_scan_flops']/1e6:.2f}",
             f"{row['reference_forward_map_attention_flops']/1e6:.2f}"])
    pages.append([
        ("h1","Appendix B. Breadth of the common implementation"),
        ("p","These small development screens test one implementation across tasks. Text and market variants "
         "include separately fitted statistical evidence. Ours denotes Sleeping Machines; TF is the saved "
         "Transformer. Accuracy improves upward; prediction loss and FLOPs improve downward."),
        ("table",(["Task / quality direction","Ours: quality","TF: quality","Ours: inference MFLOPs ↓","TF: inference MFLOPs ↓"],coverage,[42,29,29,37,37])),
        ("figure",("breadth_work_ratios",158)),
        ("p","Ours uses eight layers and TF two, both width 32, with the same neural-fitting examples, "
         "encoding, objective and eight epochs. Inference counts maps/scans (ours) and maps/attention (TF): "
         "two FLOPs per multiply-add, excluding padding, scalar nonlinearities and expert preparation. "
         "Training includes backward, clipping, Adam and our evidence/calibration; its ledger follows."),
        ("p","<b>Lower core inference work in every screen.</b> Gestures use <b>5.18× less</b> with "
         "59.1% versus 15.9% accuracy. That 44-query screen has an underfitting TF control; a general "
         "vision claim requires complete benchmarks and stronger references."),
        ("p","<b>Why training can cost more:</b> the race core evaluates all three candidate vector payloads "
         "during training, versus only the winner during inference. Its eight layers also exceed the reference's two. "
         "With short contexts, that work outweighs the saved attention cost; these rows do not show a training "
         "efficiency advantage. On the longer event-camera prefixes, the common model's estimated total uses "
         f"{100*total_work['dvs']['common_to_reference_ratio']:.1f}% of the reference training arithmetic, including calibration."),
        ("small","Seed 6; neural fit/development counts: text 2,048/256, market 512/256, temporal 1,024/256, "
         "MNIST 1,024/256, gestures 88/44. The common text model also has a separately fitted 32,768-character "
         "evidence bank; market evidence is fitted on a prior day. The references have no such bank. "
         "MNIST uses pooled training-set images; gestures use first-second prefixes and disjoint users. "
         "Batch 16, or four for gestures. No official real-data test. Market fixed evidence: 3.670 nats/event.")])

    stage_rows=[]
    for row in tasks["training_work"]["rows"]:
        for key,label in (("common","Ours"),("reference","TF")):
            part=row[key+"_stages"]
            stage_rows.append([row["task"].capitalize()+": "+label]+
                [f"{part[stage]/1e9:.3f}"
                 for stage in ("forward_and_loss","backward","gradient_clipping","optimizer")]+
                [f"{row['common_setup']['total_arithmetic_flops']/1e9:.3f}" if key=="common" else "0",
                 f"{row[key+'_total_training_flops']/1e9:.2f}"])
    pages.append([
        ("h1","Appendix B (continued). Total training cost"),
        ("p","The ledger estimates the entire completed fitting budget for each reported model. It includes "
         "prediction and loss, backpropagation, gradient clipping and Adam across all eight epochs. "
         "The common model also pays for its evidence bank and initial readout calibration. "
         "These models were trained from initialization; there is no inherited neural fitting to omit."),
        ("figure",("e172_complete_training_work",150)),
        ("table",(["Model/task","Forward + loss","Backward","Clip","Adam","Evidence + calibration","Total"],stage_rows,[43,25,24,16,19,23,24])),
        ("small","All table values are estimated GFLOPs for the whole fitting run, not per query. "
         "A multiply-add counts as two operations. Forward/loss and backward use the saved E172 four-query "
         "operator trace scaled by recorded map/scan work. Ours uses its logged candidate-map/scan "
         "counts. Reference padding is reconstructed from all fitting prefix lengths, the original shuffle seed "
         "and batch sizes, including fused attention products. Clipping and Adam are charged once per original step. "
         "Other arithmetic and the small calibration eigensolver are estimates."),
        ("small","Event-target arithmetic excludes padding and simulator dispatch/allocation; required candidate "
         "maps, losing-value teaching, scans and learning remain charged. References use the same FLOP convention. "
         "Evaluation, search, encoding, special functions, integer/index work, comparisons and memory traffic "
         "are outside these totals. These single-seed screens have different depths/quality and do not measure "
         "event hardware, matched-quality cost or energy. Ledger: "
         '<a href="experiments/estimate_training_work.py">estimate_training_work.py</a>.')])

    pages.append(language_reference_page)
    pages.extend(new_aws_pages)
    for domain in ('temporal','language'):
        records=[r for r in tasks.get('gym_screen',[]) if r['job']['domain']==domain]
        for begin in range(0,len(records),4):
            group=records[begin:begin+4];table=[];capacity=[]
            for entry in group:
                r=entry['result'];a=r['args'];w=r['work'];label='Ours '+entry['job']['variant'].replace(domain+'_','').replace('_',' ')
                if domain=='temporal':
                    fit=a['fit_targets'];dev=a['dev_targets'];cost=w['total_training_unit_special_flops'];targets=w['fitting_query_targets']
                    infer=w['inference_arithmetic_flops_per_query']+w['inference_special_functions_per_query'];quality=f"{100*r['final']['dev']['accuracy']:.2f}%"
                    available=w['available_receivers'];selected=w['selected_updates_per_event'];scores=w['key_scores_per_event']
                else:
                    fit=a['fit'];dev=r['final']['dev']['n'];cost=w['cpu_emulator']['total_training_unit_special_flops'];targets=w['fitting_targets']
                    infer=w['cpu_emulator']['inference_arithmetic_flops_per_character']+w['cpu_emulator']['inference_special_functions_per_character'];quality=f"{r['final']['dev']['bpc']:.4f}"
                    available=a['depth']*a['heads']*a['pool'];selected=a['depth']*a['heads'];scores=selected*a['pool']
                table.append([label,f"{fit}/{a['epochs']}/{dev}",quality,f"{cost/1e9:.3f}",f"{cost/targets/1e6:.3f}",f"{infer/1e6:.4f}"])
                capacity.append([label,f"{r['parameters']:,}",f"{available}/{selected}/{scores}",f"{r['wall_s']:.1f}",str(a['seed'])])
            pages.append([
                ('h1','Appendix B. AWS early '+domain+' mechanism screen'),
                ('p','Small completed integrated pilots from the committed fast matrix. These answer mechanism '
                 'questions before larger scaling; their development budgets differ from the saved main language '
                 'and native512-target results. Contracts and accounting smokes are excluded.'),
                ('figure',('aws_fast_screen_'+domain,125)),
                ('table',(['Model','Fit / passes / dev','Dev quality','Whole fit GFLOPs','Fit MFLOPs / target','Infer MFLOPs / target'],table,[42,27,24,27,27,27])),
                ('table',(['Model','Parameters','Receivers/commits/matches per event','Wall s','Seed'],capacity,[43,29,59,28,15])),
                ('small','FLOPs count MAC as two and special functions once; all losing-value credit and Adam are charged. '
                 'Temporal inference per query includes intervening input events; language per target is per character. '
                 'Capacity/activity counts are per event, not per query. CPU simulation/audit overhead, RNG, traffic and '
                 'physical energy are separate; AWS wall observations may include authorized CPU concurrency. '
                 'Temporal tasks and altered source counts are explicitly named; different '
                 'tasks do not form a single accuracy scaling curve. Single-seed development evidence, not supremacy. '
                 'Paired independent seeds and frozen held-out confirmation precede benchmark promotion.')])
            screen={entry['job']['variant']:entry['result'] for entry in group}
            if 'timing_full' in screen and 'timing_rank' in screen:
                observed=screen['timing_full']['final']['dev']['accuracy']
                rank=screen['timing_rank']['final']['dev']['accuracy']
                pages[-1].append(('small',f'Interpretation: observed-time accuracy {100*observed:.2f}% versus '
                    f'refitted rank-time {100*rank:.2f}% does not yet demonstrate an elapsed-time advantage. '
                    'State-clearing and stretched-gap probes are diagnostic interventions, not refitted controls.'))
            if 'order_sources64' in screen and screen['order_sources64']['final']['dev']['episodes']==1:
                pages[-1].append(('small','Capacity interpretation: commits and matches remain fixed while '
                    'available state grows, but fixed queries reduce per-source training exposure. The 64-source '
                    'development set has one population; its collapsed bootstrap interval is not useful uncertainty.'))
            if domain=='temporal' and begin==0:
                pages[-1].append(('small','Next experimental questions: state clearing and stretched silent gaps damage '
                    'order predictions; private source rules lose training exposure as capacity grows. The next integrated '
                    'battery tests protected memory, shared rules with private state, and paired timing whose labels '
                    'cannot be inferred from rank alone.'))

    for task in ('order','paired_timing'):
        selected=[r for r in tasks.get('split_screen',[]) if r['args']['task']==task]
        for begin in range(0,len(selected),5):
            records=selected[begin:begin+5];rows=[];activity=[]
            for r in records:
                a,w,d=r['args'],r['work'],r['final']['dev']
                name=f'Ours S{a["sources"]} '+('shared' if a['shared_maps'] else 'private')+f'/P{a["protected_pairs"]}/{a["time_input"]}/s{a["seed"]}'
                rows.append([name,f'{100*d["accuracy"]:.2f}',f'{d["nll"]:.4f}',f'{w["total_training_unit_special_flops"]/1e9:.3f}',
                             f'{w["total_training_unit_special_flops"]/w["fitting_query_targets"]/1e6:.3f}',
                             f'{(w["inference_arithmetic_flops_per_query"]+w["inference_special_functions_per_query"])/1e6:.3f}'])
                activity.append([name,f'{a["fit_targets"]}/{a["dev_targets"]}/{a["epochs"]}',f'{r["parameters"]:,}',
                                 str(w['available_receivers']),f'{w["selected_updates_per_event"]}/{w["key_scores_per_event"]}'])
            pages.append([('h1','Appendix B. Protected state/shared rules: '+task),
                ('p','Completed integrated pilots only. Protected modes retain information during silence; temporal modes still evolve. '
                 'Shared learned rules retain private addressed state and remove private source embeddings. Paired timing keeps marks/order '
                 'identical while labels differ; rank-only prediction has an exact 50% paired ceiling under coupled noise.'),
                ('table',(['Construction','Dev accuracy%','Dev NLL','Whole fit GFLOPs','Fit MFLOPs/query','Infer MFLOPs/query'],rows,[48,25,21,27,27,26])),
                ('table',(['Construction','Fit/dev/passes','Parameters','State slots','Updates/scores per event'],activity,[48,32,28,25,41])),
                ('small','Exact full fitting includes producer graphs, losing proposals, backward, clipping and Adam; specials have unit weight. '
                 'Independent population/pair uncertainty is distinct from seed uncertainty. Protected-prefix initialization also removes faster '
                 'initial temporal modes; any timing change is not isolated spectral evidence. Scope remains synthetic pilot quality, not physical energy.')])

    replications=[r for r in tasks.get('split_screen',[])
                  if r['args']['tag'].startswith('aws_event_replication_20261002T005408Z_')]
    if replications:
        shared=[r for r in replications if r['args']['task']=='order' and r['args']['shared_maps']
                and r['args']['protected_pairs']==0]
        timing=[r for r in replications if r['args']['task']=='paired_timing' and r['args']['time_input']=='observed']
        interpretation=[]
        if shared:
            scores=', '.join(f'seed{r["args"]["seed"]}: {100*r["final"]["dev"]["accuracy"]:.2f}%' for r in shared)
            interpretation.append('Shared S16/P0 order replications: '+scores+'. The original seed6 screen was75.39%; '
                'training-seed variation remains material. Private S16 replication controls and fresh-population confirmation '
                'are required for the paired sharing claim; a completed shared-only score cannot pass that gate.')
        if timing:
            scores=', '.join(f'seed{r["args"]["seed"]}: {100*r["final"]["dev"]["accuracy"]:.2f}%' for r in timing)
            interpretation.append('Observed-time paired replications: '+scores+'. Original seed6 was95.31%, '
                'with the exact rank-only paired ceiling50%. These reuse the development distribution and were '
                'chosen after seed6; they are training-seed evidence, not independent confirmation.')
        pages.append([('h1','Appendix B. Native replication scope'),
            *[('p',p) for p in interpretation],
            ('small',f'{len(replications)} of8 reserved replication pilots are complete in this checkout. '
                'Every completed seed is listed in the preceding common-unit tables; pending cells carry no score. '
                'The frozen AWS replication/confirmation chain owns the remaining work; no local duplicates.')])

    confirmations=tasks.get('event_confirmation',[])
    for begin in range(0,len(confirmations),6):
        rows=[]
        for entry in confirmations[begin:begin+6]:
            r,j=entry['result'],entry['job'];w=r['work']
            rows.append(['Ours: '+j['variant'].replace('_',' ')+f'/seed {j["seed"]}',
                f'{100*r["final"]["dev"]["accuracy"]:.2f}',f'{100*r["final"]["confirmation"]["accuracy"]:.2f}',
                f'{r["final"]["confirmation"]["nll"]:.4f}',f'{w["total_training_unit_special_flops"]/1e9:.3f}',
                'Reused; charged' if r.get('extra_optimizer_steps')==0 else 'New fit'])
        pages.append([('h1','Appendix B. Native frozen confirmation'),
            ('p','Completed 1,024-query synthetic holdout evaluations only. All fitted seeds and matched controls '
             'remain visible; selected checkpoints use development NLL before confirmation. Existing seed6 and '
             'AWS-replication weights are reused with their original whole fitting work charged.'),
            ('table',(['Construction','Dev accuracy %','Holdout accuracy %','Holdout NLL','Whole fit GFLOPs','Fitting lineage'],rows,[51,23,27,22,25,26])),
            ('small','Primary gains require the complete three-seed crossed population/pair analysis with correction '
             'across two contrasts. Partial scores cannot pass a gate. Synthetic mechanism confirmation is distinct '
             'from time-aware dense controls, real-data supremacy and physical energy.')])

    route_path=RES/'diagnostics/aws_route_write_decomposition_20261001T235000Z.json'
    if route_path.exists():
        audit=json.loads(route_path.read_text())
        if audit.get('status')=='completed':
            s=audit['summary']
            pages.append([('h1','Appendix B. Persistent-write credit diagnosis'),
                ('p','A frozen selected native checkpoint is replayed with alternative delivered content, '
                 'alternative persistent write, and both. The exact four-corner decomposition separates those '
                 'effects from their interaction and the current message-linearization residual. '
                 'The forward architecture and fitted weights remain unchanged.'),
                ('table',(['Ours: frozen audit component','Mean absolute loss effect'],[
                    ['Full alternative branch',f'{s["total_branch_loss_difference"]:.6f}'],
                    ['Persistent-write effect',f'{s["persistent_commit_effect"]:.6f}'],
                    ['Delivered-value linearization residual',f'{s["value_linearization_residual"]:.6f}'],
                    ['Delivery/write interaction',f'{s["delivery_commit_interaction"]:.6f}']], [115,59])),
                ('p','The two opposed directions in the earlier audit are explained by persistent writes: '
                 'at event0/block4 the value-only change is +0.001504 but the write-only change is −0.016749; '
                 'at event7/block7 they are −0.001280 and +0.033918. The route chooses a memory address as '
                 'well as a message. Training must teach that future state effect.'),
                ('small',f'Twelve fixed probes, one population/address/noise seed, fixed selected-node time. '
                 f'Mean absolute write effect {s["persistent_commit_effect"]:.6f} versus value residual '
                 f'{s["value_linearization_residual"]:.6f}; these absolute summaries are not additive percentages. '
                 'Hybrids are diagnostic interventions, not legal proposed routes. Value residual includes nonlinear '
                 'response and downstream route switches. This is conditional fidelity evidence, not an expected-gradient failure rate.'),
                ('small',f'Charged replay counts:48 forwards,12 backwards,12,288 races; {audit["wall_s"]:.3f}s wall, '
                 f'{audit["max_rss_kb"]/1024:.1f}MiB peak RSS. Arithmetic is uninstrumented. Checkpoint weights and '
                 'outer RNG are preserved. Any new state-aware teacher needs integrated contracts, full accounting and a matched small fit.'),
                ('small','Source: <a href="experiments/results/diagnostics/aws_route_write_decomposition_20261001T235000Z.json">'
                 'completed factorial audit</a>; <a href="experiments/theory/57_full_state_and_joint_clock_credit.md">joint state/time theory</a>.')])

    for dataset in ('banknote','wine_red'):
        rows=[r for r in tasks.get('native_tabular',[]) if r['args']['dataset']==dataset]
        for begin in range(0,len(rows),6):
            records=rows[begin:begin+6]
            pages.append([
                ('h1','Appendix B. Ours and boosted trees: '+dataset),
                ('p','Independent feature-ID rows, state reset between rows, train-only scaling and duplicate-feature '
                 'group isolation. Ours uses eight native event blocks with parallel heads and content/state mixing; '
                 'R2 adds temporal reception. Static processing coordinates are not physical asynchronous samples.'),
                ('table',(['Model','Fit/dev rows','Dev NLL / RMSE ↓','Dev accuracy / MAE','Whole fit GFLOPs','Fit MFLOPs/row','Infer MFLOPs/row'],[
                    [('Ours R'+str(r['args']['clock_features'])) if r['args']['model']=='ours' else 'Boosted trees',
                     f"{r['args']['fit']}/{r['args']['dev']}",f"{r['final']['dev']['nll' if dataset=='banknote' else 'rmse']:.4f}",
                     f"{r['final']['dev']['accuracy' if dataset=='banknote' else 'mae']:.4f}",
                     f"{r['work']['whole_neural_fit_unit_special_flops']/1e9:.3f}" if r['args']['model']=='ours' else 'Not counted',
                     f"{r['work']['neural_fit_unit_special_flops_per_row']/1e6:.3f}" if r['args']['model']=='ours' else 'Not counted',
                     f"{r['work']['inference_unit_special_flops_per_row']/1e6:.3f}" if r['args']['model']=='ours' else 'Not counted'] for r in records],[28,23,25,28,25,23,22])),
                ('table',(['Model','All fit wall s','Peak RSS MiB','Tree nodes / bytes'],[
                    ['Ours R'+str(r['args']['clock_features']) if r['args']['model']=='ours' else 'Boosted trees',
                     f"{r['work']['fit_wall_s' if r['args']['model']=='ours' else 'tree_fit_wall_s_all_candidates']:.2f}",f"{r['max_rss_kb']/1024:.1f}",
                     'Not applicable' if r['args']['model']=='ours' else f"{r['work']['selected_tree_nodes']:,}/{r['work']['selected_tree_node_bytes']:,}"] for r in records],[44,40,40,50])),
                ('small','Four development checkpoints or four separately fitted tree candidates; all candidate tree fitting '
                 'wall time is charged. Neural fit arithmetic is an actual forward/loss/backward/clipping/Adam trace, '
                 'with specials counted once; preprocessing, evaluation and RNG are separate. Tree FLOPs are unavailable '
                 'and are not manufactured. Neural wall time includes CPU simulation/audit instrumentation. '
                 'These are small exploratory development results; reserved test labels are '
                 'not scored. Strong tabular/frontier superiority requires larger frozen protocols and independent seeds.')])
            if dataset=='banknote':
                pages[-1].append(('small','Reception ablation: ours R2 reaches89.06% accuracy /0.270 NLL versus '
                    'native R0 95.31% /0.155. Whole fitting work increases from0.335 to0.379 GFLOPs. '
                    'The added reception capacity has not earned its cost in this single-seed static-data screen.'))
                pages[-1].append(('small','Confirmation protocol: native checkpoint reuse plus seeds7/8, original trees, '
                    'CatBoost and logistic regression; four development selection opportunities per family, then frozen '
                    'reserved-test scoring. See experiments/AWS_BANKNOTE_CONFIRMATION.md. No pending test score is reported.'))
            if dataset=='wine_red' and any(r['args']['model']=='ours' and r['args']['clock_features']==2 for r in records):
                pages[-1].append(('small','Reception helps this regression pilot: ours R2 RMSE0.758 versus R0 0.824 '
                    '(8.0% lower), for0.919 versus0.816 whole-fit GFLOPs (12.6% more). Trees retain lower RMSE0.649. '
                    'This positive within-model effect contrasts with banknote/language reception failures; it is not a cross-family win.'))

    credited=tasks.get('state_credit',[])
    if credited:
        rows=[];activity=[]
        for r in credited:
            w=r['work'];a=r['args'];q=r['final']['dev']
            rows.append(['Ours: baseline' if not a['state_credit'] else 'Ours: write credit',
                f'{100*q["accuracy"]:.2f}',f'{q["nll"]:.4f}',f'{w["total_training_unit_special_flops"]/1e9:.3f}',
                f'{w["total_training_unit_special_flops"]/w["fitting_query_targets"]/1e6:.3f}',
                f'{(w["inference_arithmetic_flops_per_query"]+w["inference_special_functions_per_query"])/1e6:.3f}'])
            activity.append(['Ours: baseline' if not a['state_credit'] else 'Ours: write credit',
                f'{a["fit_targets"]}/{a["dev_targets"]}/{a["epochs"]}',f'{r["parameters"]:,}',
                str(w['available_receivers']),f'{w["selected_updates_per_event"]}/{w["key_scores_per_event"]}'])
        credit_scope=''
        parent=next((r for r in credited if r['args']['state_credit']==0),None)
        teacher=next((r for r in credited if r['args']['state_credit']==1),None)
        settings=lambda r:{k:v for k,v in r['args'].items() if k not in ('tag','state_credit')}
        if parent and teacher and settings(parent)==settings(teacher) and parent['data_sha256']==teacher['data_sha256']:
            gain=teacher['final']['dev']['accuracy']-parent['final']['dev']['accuracy']
            nll_gain=parent['final']['dev']['nll']-teacher['final']['dev']['nll']
            ratio=teacher['work']['total_training_unit_special_flops']/parent['work']['total_training_unit_special_flops']
            passed=gain>=.05 and nll_gain>=.02 and ratio<=2
            credit_scope=(f'Matched gains: {100*gain:.2f} percentage points / {nll_gain:.5f} NLL, '
                f'at {ratio:.3f}× fitting work. The predeclared larger-fit gate '
                f'(5 points/.02 NLL/at most2× work) {"passes" if passed else "fails"}. ')
        pages.append([('h1','Appendix B. Addressed-state write credit'),
            ('p','Completed integrated pilots only: private S4/P0, eight blocks, two independent heads, d8/pool2, '
             '128 fitting queries per pass/four passes,256 development queries, seed6. Both retain hard temporal '
             'races and winner-only inference. The zero-credit model exactly nests the parent; added memory/time '
             'credit is a local surrogate, not an arbitrary unbiased sequence-gradient estimator.'),
            ('figure',('state_credit_quality_work',160)),
            ('table',(['Model','Accuracy %','NLL','Whole fit GFLOPs','Fit MFLOPs/query','Infer MFLOPs/query'],rows,[40,24,23,30,29,28])),
            ('table',(['Model','Fit/dev/passes','Parameters','State slots','Updates/scores per event'],activity,[40,32,28,25,49])),
            ('small','All fitting forward/loss/backward/normalization/clipping/Adam and losing proposals are charged. '
             'Special functions have unit weight beside arithmetic; integer discovery/traffic/energy remain separate. '
             'Training-only auxiliary state views and whole-process RSS are recorded in each result. '
             'Numerical/optimizer prerequisites and accounting smokes are excluded from benchmark plots. '
             +credit_scope+'This reuses exploratory development populations; independent seeds and fresh confirmation remain required.')])
    learning=tasks.get('language_learning_audit')
    if learning:
        c=learning['composition_gradient'];rows=[]
        for r in learning['frozen_models']:
            norms=[d['gradient_norm'] for d in r['layer_gradients']]
            rows.append(['Ours: carrier,131K' if '/parallel_language/' in r['result'] else 'Ours: native,8K',
                str(len(norms)),f'{min(norms):.3f}–{max(norms):.3f}',str(r['forward_input_tokens']),str(r['backwards'])])
        pages.append([('h1','Appendix B. Learned history and credit reach'),
            ('p','Frozen checkpoint audit on32 fixed development positions. Both learned models depend on history; '
             'the native predictions change even when identical16-character suffixes receive the same race noise. '
             'The native model is therefore not strictly a bigram predictor. Equal average loss to a count model '
             'calibrates predictive quality; it does not identify learned features or context dependence.'),
            ('figure',('language_learning_context',168)),
            ('table',(['Saved model','Layers with gradients','Layer norm range','Replay tokens','Backwards'],rows,[48,30,37,30,29])),
            ('p',f'The actual current count-composition logit gradient matches responsibility-weighted cross entropy '
             f'to{c["maximum_absolute_gradient_error"]:.2g}; all six carrier layers receive gradients. '
             f'Mean responsibility is{100*c["mean_responsibility"]:.2f}% on this64-target initialization probe. '
             'This supports investigating attenuated task signal and conditioning, rather than assuming a general gradient disconnect.'),
            ('small','No optimizer steps, weight changes or official-test access. History controls reset state and replay '
             'the retained suffix at its absolute positions; per-position native noise is coupled. This32-position slice '
             'is not the full saved development quality or a matched-data model comparison. Gradient norms aggregate different '
             'parameter groups; they show reach, not superior conditioning or unbiased hard-route credit. '
             'The64-character history is not uniformly better than16 on this slice; useful long-range/semantic features remain open. '
             f'All replays/backwards are counted; arithmetic is uninstrumented. Wall{learning["wall_s"]:.2f}s, '
             f'peakRSS{learning["max_rss_kb"]/1024:.1f}MiB. See experiments/LANGUAGE_LEARNING_DIAGNOSIS_20261002.md.')])
    horizon=tasks.get('language_credit_horizon')
    if horizon:
        values=[[str(r['credit_targets']),f'{r["prediction_bpc"]:.6f}',f'{r["gradient_norm"]:.4f}',
            str(r['graph_input_tokens']),str(r['forward_input_tokens'])] for r in horizon['horizons']]
        contrasts=[[f'{r["short"]} versus {r["long"]}',f'{r["whole_model"]["difference_norm"]:.4f}',
            f'{100*r["whole_model"]["relative_difference_to_long"]:.2f}',f'{r["whole_model"]["cosine_similarity"]:.4f}']
            for r in horizon['comparisons']]
        pages.append([('h1','Appendix B. Credit horizon with fixed context'),
            ('p','Completed frozen native8K/H2/d16/depth8 diagnostic. All three arms retain the same128-token '
             'development context, score the same last16 targets and receive identical per-position race noise. '
             'Only graph reach changes:16,32 or64 tokens. Predictions match exactly; the gradient changes, '
             'separating retained information from the credit used to learn how to retain it.'),
            ('table',(['Graph horizon','Slice bpc','Gradient norm','Graph tokens','Replay tokens'],values,[35,36,36,33,34])),
            ('table',(['Short/long credit','Difference norm','Relative difference %','Gradient cosine'],contrasts,[42,43,48,41])),
            ('p','The16-versus64 gradient difference has norm30.89% of the64-token gradient, with cosine0.9570. '
             'The32-versus64 difference is15.15%. Every layer receives credit, and short-credit norms can be '
             'larger because omitted contributions can cancel retained ones. This is evidence of material '
             'truncation effects on this probe; it does not establish that increasing the horizon improves fitting quality.'),
            ('small',f'Frozen weights; zero optimizer steps and no official-test access. One16-target slice,384 replay '
             f'tokens,112 graph tokens,3 backwards. Wall{horizon["wall_s"]:.2f}s, peakRSS{horizon["max_rss_kb"]/1024:.1f}MiB. '
             'Arithmetic is uninstrumented. Native hard-route credit is a surrogate;64 tokens is a comparison, '
             'not an all-history unbiased reference. Relative difference divides the norm of the gradient difference '
             'by the longer-credit gradient norm, not a percentage of predictive quality or retained features.')])
    confirmed=tasks.get('tabular_confirmation',[])
    labels={'ours':'Ours native','trees':'Boosted trees','catboost':'CatBoost','logistic':'Logistic'}
    order=list(labels)
    confirmed=sorted(confirmed,key=lambda r:(order.index(r['args']['model']),r['args']['seed']))
    if replicated_banknote:
        means=[]
        for family in ('ours','trees','catboost','logistic'):
            rows=banknote_confirmation[family]
            if sorted(r['args']['seed'] for r in rows)==[6,7,8]:
                means.append([labels[family], '3',f'{100*test_mean(family,"accuracy"):.2f}',f'{test_mean(family,"nll"):.4f}'])
            else:means.append([labels[family],f'{len(rows)}/3','Pending','Pending'])
        pages.append([('h1','Appendix B. Banknote: no confirmed advantage'),
            ('p',banknote_scope+'The reserved-test accuracy point estimates are close, but no equivalence margin was specified. '
             'Parity is therefore a descriptive reading, not a proven equivalence claim.'),
            ('figure',('banknote_reserved_test',174)),
            ('table',(['Model','Completed seeds','Mean test accuracy %','Mean test NLL'],means,[45,33,48,48])),
            ('p','For the completed ours/tree comparison, the paired accuracy difference is −2.14 percentage points '
             '(descriptive95% crossed seed/feature-group interval −6.90 to +2.43). The control-minus-ours NLL '
             'difference is −0.0287 (98.33% interval −0.1738 to +0.1277). Logistic regression improves NLL by0.1447 '
             '(98.33% interval0.0338 to0.2660). These three-seed intervals are approximate and share one test split.'),
            ('small','Partial analysis: local_banknote_partial_confirmation_20261002T013000Z.json;11/12 final cells, '
             '4000 bootstrap draws,270 feature groups, three seeds. NLL intervals allow for three control comparisons; '
             'accuracy intervals are descriptive. This does not replace the incomplete full four-family gate. '
             'The original95.3% versus93.0% development screen and all per-seed work remain below.')])
    for begin in range(0,len(confirmed),6):
        records=confirmed[begin:begin+6];quality=[];costs=[]
        for r in records:
            a,w,p=r['args'],r['work'],r['protocol'];model=labels[a['model']]+f'/s{a["seed"]}'
            dev,test=r['final']['dev'],r['final']['test']
            quality.append([model,f'{dev["nll"]:.4f}',f'{test["nll"]:.4f}',f'{100*test["accuracy"]:.2f}',
                f'{w["whole_neural_fit_unit_special_flops"]/1e9:.3f}' if a['model']=='ours' else 'Not counted',
                f'{w["neural_fit_unit_special_flops_per_row"]/1e6:.3f}' if a['model']=='ours' else 'Not counted',
                f'{w["inference_unit_special_flops_per_row"]/1e6:.3f}' if a['model']=='ours' else 'Not counted'])
            costs.append([model,f'{w["fit_wall_s" if a["model"]=="ours" else "control_fit_wall_s_all_candidates"]:.2f}',
                f'{r["confirmation_wall_s"]:.2f}',f'{r["max_rss_kb"]/1024:.1f}',
                'Historical fit reused' if r.get('reused_fit') else 'New fit'])
        pages.append([('h1','Appendix B. Frozen banknote confirmation'),
            ('p','Completed test scores only. Checkpoints/candidates were selected on128 development rows after fitting '
             '128 rows. Reserved feature groups were scored after choices were frozen; all rows start with cold state. '
             'The per-seed ledger preserves completed scores; the summary identifies families with all three seeds.'),
            ('table',(['Model/seed','Dev NLL','Test NLL','Test accuracy%','Whole fit GFLOPs','Fit MFLOPs/row','Infer MFLOPs/row'],quality,[36,22,23,24,23,23,23])),
            ('table',(['Model/seed','Charged fit wall s','Test wall s','Peak RSS MiB','Fit provenance'],costs,[36,36,29,29,44])),
            ('small','Native forward/loss/backward/clipping/Adam are traced. Seed6 reuse retains its original full fitting '
             'charge; it adds no optimizer steps. Control fitting includes all four independent candidates. Their FLOPs '
             'are unavailable. Audit instrumentation, preprocessing and physical energy are separate. Repeated seeds '
             'share test rows and must not be pooled as independent observations. Paired seed/feature-group analysis '
             'and all three prespecified seeds are required for the confirmation claim.')])


    if full_rows:
        latest=max(full_rows,key=lambda row:(row['args']['fit'],-row['final']['dev']['bpc']))
        w=latest['work'];ours_fit=w['total_training_unit_special_flops']/w['fitting_targets']
        ours_forward=w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character']
        tf=lm_costs['tf'];tf_fit=tf['total_training_flops']/tf['training_token_positions']
        tf_forward=tf['forward_flops']/tf['training_token_positions']
        work_rows=[]
        for r in full_rows:
            rw=r['work']
            work_rows.append([
                f"Ours / d{r['args']['payload']} / p{r['args']['pool']}",
                f"{r['args']['fit']:,} / {r['args']['epochs']}",
                f"{r['final']['dev']['bpc']:.3f} / dev",
                f"{rw['total_training_unit_special_flops']/1e9:,.3f}",
                f"{rw['total_training_unit_special_flops']/rw['fitting_targets']/1e6:.3f}",
                f"{(rw['inference_arithmetic_flops_per_character']+rw['inference_special_functions_per_character'])/1e6:.3f}"])
        for name,label,budget,score in [('lstm','LSTM / width 512','10M / six',ev['lstm10']),
                                      ('tf','Transformer / width 256','10M / four',ev['tf10'])]:
            rw=lm_costs[name]
            work_rows.append([label,budget,f"{score:.3f} / test",
                f"{rw['total_training_flops']/1e9:,.3f}",
                f"{rw['total_training_flops']/rw['training_token_positions']/1e6:.3f}",
                f"{rw['forward_flops']/rw['training_token_positions']/1e6:.3f}"])
        pages.append([
            ('h1','Appendix B (continued). Ours: language work as scaling develops'),
            ('p','This ledger updates from completed integrated-model stages. It shows the emerging '
             'work advantage alongside its quality and data budget. Per-target fitting work removes '
             'the difference in the number of presentations; it does not establish equal-quality superiority.'),
            ('figure',('integrated_language_work_progress',152)),
            ('table',(['Model','Fit / passes','bpc / split ↓','Whole fit GFLOPs ↓',
                      'Fitting MFLOPs / target ↓','Forward MFLOPs / position ↓'],
                      work_rows,[38,25,25,27,29,30])),
            ('p','Compare within a column: whole-fit totals use GFLOPs for every model; per-target '
             'and forward work use MFLOPs for every model. One GFLOP is 1,000 MFLOPs. Whole-fit totals '
             'also depend on the number of training presentations; the per-target column divides that out.'),
            ('small','All table values, figures and ratios use arithmetic plus one operation per special '
             'function, matching the historical neural estimate convention. This is not a physical energy '
             'cost. Ours arithmetic-only whole-fit totals (GFLOPs): '+
             '; '.join(f"{r['args']['fit']:,} / pool {r['args']['pool']}: "
                       f"{r['work']['total_training_arithmetic_flops']/1e9:.3f}" for r in full_rows)+
             '. Separate special-function counts are preserved in each result.'),
            ('p',f"<b>The raw work gap is substantial.</b> The completed {latest['args']['fit']:,}-character "
             f"integrated stage's representative forward estimate is <b>{tf_forward/ours_forward:.0f}× smaller</b> "
             f"than the larger saved Transformer estimate; fitting work per target is <b>{tf_fit/ours_fit:.0f}× smaller</b>. "
             'These are configuration-level work ratios. Our development score and the reference official '
             'test score use different targets and data budgets. The gap is not a matched-quality supremacy claim.'),
            ('small','Ours: d denotes payload width and p pool size; fixed character pools and event depths. '
             'Each result records its validation interval and credit horizon. References: width-512 LSTM or four width-256 Transformer layers, '
             '256-position fitting chunks and 999,999 aligned official test targets. Ours uses representative '
             'operator traces including counterfactual credit, backward, clipping and Adam; neural references '
             'use shape formulas and backward ≈ twice forward. RNG, indexing, memory traffic and evaluation '
             'passes are additional. Same-quality and iso-FLOP conclusions await comparable completed runs.')])
    points=language_work_points(tasks,ev)
    pages.append([
        ('h1','Appendix B (continued). Ours and neural controls: accuracy versus FLOPs'),
        ('p','Each point is a completed model, not a projected scaling law. Left: ours on cold '
         'development characters, with integrated models and earlier carrier controls labelled separately. '
         'The new 2K screens score 2,047 development targets; the earlier ladders score 8,191. '
         'Right: saved neural test results INCLUDING the later segment-batched native width, depth, pool '
         'and credit models (purple diamonds, NB labels). Lower bpc means better prediction; lower fitting work means '
         'fewer estimated operations. No curve is drawn between different model families or scoring splits.'),
        ('figure',('language_quality_vs_work',174)),
        ('table',(['Model type','Fitting budget','bpc / split ↓','Whole fit GFLOPs ↓','Fitting MFLOPs / target ↓'],[
            [r['model'],f"{r['fit']:,} / {r['passes']:g} passes",f"{r['bpc']:.3f} / {r['split']}",
             f"{r['total']/1e9:,.3f}",f"{r['total']/r['targets']/1e6:.3f}"]
            for family in ('integrated','native_batched','carrier','lstm','tf')
            for r in [max([p for p in points if p['family']==family],key=lambda p:(p['fit'],-p['bpc']))]],
            [43,33,25,33,40])),
        ('small','The table selects the largest fitting budget currently completed for each family; '
         'the best score breaks ties. Point numbers refer to the following variant ledger, which lists all plotted '
         'variants. Variant labels: I = ours integrated payload/pool/data; IKV adds per-position race memory '
         '(S uses the content index); '
         'NB = later batched native, T128/T256 evaluation shown; C = ours carrier width/data '
         '(g means content gates); L = LSTM width/data; T = Transformer '
         'width x layers/data; s denotes seed. K is 1,024 characters in ours labels; M is decimal million in neural labels.'),
        ('small','Estimates include learning, clipping and Adam, with unit-weight special functions. '
         'Ours uses representative operator traces; neural controls use shape formulas and backward '
         'approximately twice forward. Scoring splits, data, passes, capacity and credit differ; '
         'these panels are evidence inventories, not an iso-FLOP or equal-quality benchmark.')])
    recent=tasks['native_language_batched']
    recent_names=['p32/d4 + route credit','p32/d4/pool4 + route credit','p64/d4 + route credit',
                  'p64/d4/pool4 + route credit','p96/d4 + route credit']
    recent_rows=[r for label in recent_names for r in recent['native'] if r['label']==label]
    pages.append([
        ('h1','Appendix B (continued). Later native language: quality versus fitting work'),
        ('p','Focused view of the later credited width/depth/capacity models. Same completed T256 '
         'scores as the common inventory; nominal10M fitting characters, one pass and saved one-pass controls. '
         'Blue is alternative-value route credit; light blue is timing-only credit; gray is a dense control.'),
        ('figure',('latest_native_language_fitting',174)),
        ('table',(['Model','T256 test bpc','Whole fit TFLOPs est.','Fit MFLOPs / input position est.'],[
            [r['label'],f"{r['test256']:.4f}",f"{r['whole']/1e12:.2f}",f"{r['fit']/1e6:.2f}"] for r in recent_rows]+[
            [r['label'],f"{r['test']:.4f}",f"{r['whole']/1e12:.2f}",f"{r['fit']/1e6:.2f}"] for r in recent['controls']],
            [61,29,39,45])),
        ('small','Native quality comes from the compiled training evaluator, fitting work from representative '
         'full-step traces; controls use shape estimates. Every column has the same units and denominator '
         'for ours and controls. Native random-segment fitting and evaluation tail coverage differ from '
         'controls. Single seeds; trained sparse-backend rescore and modern replications remain open. '
         'p96 now slightly exceeds LSTM quality with more fitting work. All13 later native fits, '
         'including timing-only/pool1/write-credit history, remain in the common graph and following ledger.')])
    pages.append([
        ('h1','Appendix B (continued). Accuracy versus inference FLOPs'),
        ('p','Inference predicts with frozen weights: no backward pass, clipping or optimizer update. '
         'These are the same completed checkpoints, quality scores and point IDs as the fitting graph. '
         'Ours uses saved forward operator traces; the integrated models read only winning values. '
         'Later native NB points use emulator traces, charged for evaluated warm positions per scored target; '
         'winner-only estimates remain in the native appendix pending trained parity/rescore. '
         'LSTM and Transformer costs use shape estimates. Development and test evidence remain separate.'),
        ('figure',('language_quality_vs_inference',174)),
        ('table',(['Model type','bpc / split ↓','Inference MFLOPs / character ↓','Cost boundary'],[
            [r['model'],f"{r['bpc']:.3f} / {r['split']}",f"{r['inference']/1e6:.4f}",r['inference_method']]
            for family in ('integrated','native_batched','carrier','lstm','tf')
            for r in [max([p for p in points if p['family']==family],key=lambda p:(p['fit'],-p['bpc']))]],
            [43,27,42,62])),
        ('small','Solid Transformer points estimate its saved 256-position scorer: full windows advanced '
         'by 128 positions, approximately two forward positions per scored character (boundary/tail overhead omitted). '
         'Hollow points show a hypothetical one-step decode with cached keys/values and 256 available positions, '
         'using L(24d² + 4Td + 30d + 20T) + 54d + 135 unit-weight operations, T = 256. '
         'No cached decoder was run. Its plotted bpc belongs to the saved window scorer; learned positions reset '
         'between windows, so cache reuse has not been shown to preserve those scores.'),
        ('small','Two FLOPs per multiply-add; special functions count as one operation. Ours traces include '
         'numerical clocks and loss scoring; neural elementwise overhead is approximate. Traces are representative '
         'warm-state costs, not full-stream measurements; growing KV candidate occupancy can change work. '
         'The separate KV pages also show projected event-architecture costs that remove numerical clock simulation. '
         'RNG, indexing, memory traffic and physical race energy are additional. These are work estimates, '
         'not latency or joules, and differing data, quality and evaluation protocols prevent a supremacy conclusion.')])
    ledger_chunk=math.ceil(len(points)/math.ceil(len(points)/12))
    for start in range(0,len(points),ledger_chunk):
        pages.append([
            ('h1','Appendix B (continued). Completed language variants and work'),
            ('table',(['Variant','Params K','Fit / passes','bpc / split ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Inference MFLOPs / char ↓'],[
                [f"{i+1}. "+('Ours: ' if r['family'] in ('integrated','native_batched','carrier') else '')+r['label'],
                 f"{r['parameters']/1e3:,.1f}",f"{r['fit']:,} / {r['passes']:g}",
                 f"{r['bpc']:.3f} / {r['split']}",f"{r['total']/1e9:,.3f}",f"{r['total']/r['targets']/1e6:.3f}",f"{r['inference']/1e6:.4f}"]
                 for i,r in enumerate(points[start:start+ledger_chunk],start=start)],
                 [36,18,25,20,27,24,24])),
            ('small','Each row retains its original architecture, fitting budget and score. The selected '
             '10M LSTM/Transformer rows use the aligned 999,999-target scores; other neural rows retain '
             'their original E64 test scorers. The 90M LSTM uses its saved recurrent scoring protocol. '
             'Carrier and integrated development scores use frozen evaluation; integrated official scores '
             'appear only after their full test completes. Validation/test work, RNG and physical traffic '
             'are outside fitting totals. Sources: E64/E174, saved AWS E64 results and the completed '
             'parallel_language, episodic_language and language_batched JSON records. Later NB rows use '
             'T256 when completed (first v1 stays T128), actual fitting presentations and native window '
             'overlap charged per scored target; different tail coverage is retained. The global ledger uses emulator '
             'floating arithmetic consistently; fitting work per target divides by actual training target presentations. '
             'The separate KV page reports architectural projections. '
             'No new dense model was trained.')])
    for pair in episodic_pairs(tasks):
        kv=pair['kv'];a=kv['args'];act=kv['final']['dev']['activity']
        rows=[pair[mode] for mode in ('receiver','kv')]
        gain=rows[0]['final']['dev']['bpc']-kv['final']['dev']['bpc']
        mean_candidates=act['kv_scores']/max(1,act['kv_queries'])
        index=a.get('candidate_index','character')
        index_explanation=(f"The content index uses three random-hyperplane bits of learned keys/queries, "
            f"with up to {a['matching']} recent/full-history samples in the query bucket and its one-bit "
            f"neighbors, plus {a['recent']} recent positions. " if index=='semantic' else
            f"The index admits up to {a['matching']} recent matching-character entries plus {a['recent']} "
            'recent positions. Older entries outside these tails cannot be addressed by this index. ')
        pages.append([
            ('h1','Appendix B (continued). Ours: per-position race KV memory'),
            ('p',f"Both integrated models fit {a['fit']:,} characters for {a['epochs']} passes, with "
             f"payload {a['payload']}, {a['depth']} sparse receiver depths, seed {a['seed']} and "
             f"{a['dev']-1:,} identical cold development targets. The KV arm retains separate historical "
             'keys and values at every depth; learned queries select one value through time. '
             'Incoming content is retained and gated with the retrieved message. No dense carrier is added.'),
            ('figure',(f"episodic_language_comparison_D{a['fit']}_depth{a['depth']}_{index}",148)),
            ('table',(['Ours: memory','Dev bpc ↓','Projected fit GFLOPs ↓','CPU fit GFLOPs ↓','Projected forward MFLOPs/char ↓'],[
                [r['args']['memory'],f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['projected_event_architecture']['total_training_unit_special_flops']/1e9:,.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:,.3f}",
                 f"{(r['work']['projected_event_architecture']['inference_arithmetic_flops_per_character']+r['work']['projected_event_architecture']['inference_special_functions_per_character'])/1e6:.4f}"] for r in rows],
                [30,24,40,36,44])),
            ('p',f"Completed KV improvement over receiver memory: {gain:+.3f} bpc (positive is better). "
             +index_explanation+f"Duplicates are removed. Development averages {mean_candidates:.2f} keys scored "
             f"and one value delivered per retrieval query. All {act['kv_stored_entries']:,} entries remain "
             f"stored ({act['kv_raw_key_value_bytes']/2**20:.2f} MiB raw keys/values); the oldest selected "
             f"entry is {act['kv_winner_age_max']:,} characters old. This bounds reads, not stored history."),
            ('small','Physical clock competition replaces explicit numerical rate exponentiation and '
             'noise/rate division plus the bounded-delay simulation in the projected ledger. Query/key/value '
             'maps, scored candidates, gated content, backward, counterfactual teaching, clipping and '
             'actual Adam remain charged. Counts are representative first/mature/partial traces; '
             'special functions have unit weight here and are separate in JSON. Physical rate setting, '
             'clock circuits, index/address operations, RNG and traffic need their own implementation '
             'costs; FLOPs do not certify energy.'),
            ('small','Temporal races avoid the explicit normalizing reduction/division and deliver one '
             'value at inference; training reads all admitted values for route credit. The orange bar '
             'is an analytical same-shortlist aggregation comparison, not another trained model. '
             'Candidate coverage is approximate and does not guarantee full-bank attention equivalence. '
             'Random-hyperplane indexing is an established primitive (Charikar, STOC 2002); novelty is '
             'not claimed for this index. Historical activations '
             'are detached at the credit boundary and are not recomputed after parameter updates. '
             'One seed and a small data budget; no equal-quality Transformer or frontier claim.')])
    optimizer_pilots=[r for r in tasks['episodic_language'] if r['args'].get('heads')==2 and r['args']['fit']==2048 and r['args']['dev']==8192 and 'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and r['args'].get('chunk',16)==16]
    if len(optimizer_pilots)>1:
        optimizer_pilots.sort(key=lambda r:(r['args'].get('update_targets',16),r['args']['lr']))
        reference=next(r for r in optimizer_pilots if r['args'].get('update_targets',16)==16)
        best=min(optimizer_pilots,key=lambda r:r['final']['dev']['bpc'])
        selected=min((r for r in optimizer_pilots if r['final']['dev']['bpc']<=best['final']['dev']['bpc']+.05),key=lambda r:r['work']['cpu_emulator']['total_training_unit_special_flops'])
        saving=1-selected['work']['cpu_emulator']['total_training_unit_special_flops']/reference['work']['cpu_emulator']['total_training_unit_special_flops']
        pages.append([
            ('h1','Appendix B (continued). Ours: cheaper learning updates'),
            ('p','Credit still propagates over 16-character segments. Gradients are summed over U targets, '
             'normalized by their actual count, clipped once and used for one Adam update. '
             'All models here use two independent heads, payload 32/head, eight blocks, seed 6, four passes '
             'over 2,048 fitting characters and 8,191 frozen development targets.'),
            ('figure',('parallel_optimizer_work',174)),
            ('table',(['Ours: U / learning rate','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Adam + clip GFLOPs ↓'],[
                [f"U{r['args'].get('update_targets',16)} / {r['args']['lr']:g}",f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/r['work']['fitting_targets']/1e6:.3f}",
                 f"{sum(r['work']['cpu_emulator']['training_stages'].get(k,0) for k in ('optimizer','gradient_clipping'))/1e9:.3f}"] for r in optimizer_pilots],
                [40,25,35,37,37])),
            ('p',f"The cheapest schedule within the declared 0.05 bpc tolerance of the best pilot uses U{selected['args'].get('update_targets',16)}, lr {selected['args']['lr']:g}: {selected['final']['dev']['bpc']:.3f} bpc and {100*saving:.1f}% less whole fitting work than reference. The best quality is {best['final']['dev']['bpc']:.3f} bpc. These completed results support optimizer amortization in this configuration, not language-model supremacy."),
            ('small','Learning rates and warmup differ across configurations, so this is not an isolated optimizer-interval ablation. '
             'Selected checkpoints minimize frozen development loss over the fixed four passes; all fitting work '
             'remains charged. Each result is one seed. Candidate scoring, losing-value credit, backward and '
             'gradient accumulation/normalization remain in the operator ledger. Larger-data and repeat-seed '
             'comparisons must establish transfer of the selected schedule.'),
        ])
    repeated=[r for r in tasks['episodic_language'] if r['args'].get('arrivals',1)>1]
    arrival_fields=('heads','payload','depth','pool','matching','recent','fit','dev','epochs','chunk','update_targets','warmup_targets','seed','lr')
    for group in sorted({tuple(r['args'][k] for k in arrival_fields) for r in repeated}):
        trials=[r for r in repeated if tuple(r['args'][k] for k in arrival_fields)==group]
        a=trials[0]['args'];fit=a['fit']
        same=lambda r: all(r['args'].get(k,1 if k=='arrivals' else None)==a.get(k) for k in
            ('heads','payload','depth','pool','matching','recent','fit','dev','epochs','chunk','update_targets','warmup_targets','seed','lr'))
        reference=[r for r in tasks['episodic_language'] if 'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and same(r)]
        rows=sorted(reference+trials,key=lambda r:r['args'].get('arrivals',1))
        extra=[]
        if reference:
            extra.append(('figure',('repeated_arrival_quality_work_'+hashlib.sha256(repr(group).encode()).hexdigest()[:10],174)))
            many=max(trials,key=lambda r:r['args']['arrivals']);base=reference[0]
            cost=100*(many['work']['cpu_emulator']['total_training_unit_special_flops']/base['work']['cpu_emulator']['total_training_unit_special_flops']-1)
            extra.append(('p',f"Ours delivers {many['args']['arrivals']} times as many historical winner messages for {cost:.2f}% additional whole fitting arithmetic in this completed screen. Quality changes from {base['final']['dev']['bpc']:.3f} to {many['final']['dev']['bpc']:.3f} bpc. This supports cheap arrival multiplicity under shared matches; it does not establish language-model superiority."))
        pages.append([
            ('h1','Appendix B (continued). Ours: shared-match temporal arrivals'),
            ('p',f"{a['heads']} independent spatial heads, payload {a['payload']}/head, {a['depth']} blocks. "
             f"All rows fit {fit:,} characters for {a['epochs']} passes and score {a['dev']-1:,} cold development targets. "
             'Each query forms its candidate matches once. Multiple temporal marks reuse those rates; only the winning emitter renews its clock. '
             'The receiver and each selected message evolve until the last local read, then the messages are averaged and gated.'),
            *extra,
            ('table',(['Ours: arrivals / head','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Inference MFLOPs / char ↓'],[
                [f"m={r['args'].get('arrivals',1)}",f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/r['work']['fitting_targets']/1e6:.3f}",
                 f"{(r['work']['cpu_emulator']['inference_arithmetic_flops_per_character']+r['work']['cpu_emulator']['inference_special_functions_per_character'])/1e6:.4f}"] for r in rows],
                [32,25,35,38,44])),
            ('p','Candidate discovery and independent Q/K/V projections are retained. Multiple marks reuse one rate setting within each query; trained scores and candidate trajectories can differ across runs. '
             'More marks can retrieve the same value; they do not create extra learned spatial heads or discover absent candidates. '
             'Training reads all admitted values once and aggregates the conserved per-arrival teacher in O(Cd + md). '
             'All delivered messages, temporal transports, backward, clipping and Adam remain charged.'),
            ('small','Completed single-seed development screens; m=1 reuses the saved reference under exact nesting contracts. '
             'Unit-weight special functions are included; CPU minimum comparisons/RNG and memory traffic are separate counters. '
             'The bounded numerical time encoding is not a demonstrated homogeneous physical Poisson clock. '
             'This local counterfactual teacher is a declared surrogate, not an exact gradient through nonlinear route changes. '
             'No matched-quality dense-model, physical-energy or frontier superiority is inferred.')])
    head_rows=[r for r in tasks['episodic_language'] if r['args'].get('heads',1)>1 and 'write_credit' not in r['args'] and r['args'].get('arrivals',1)==1 and r['args'].get('chunk',16)==16]
    if head_rows:
        display=head_rows[-4:]
        pages.append([
            ('h1','Appendix B (continued). Ours: independent temporal heads'),
            ('p','Each head has its own receiver pool, historical bank and query/key/value/gate matrices. '
             'A winning content vector evolves through learned rotation and decay until its channel is read. '
             'The next block reads at the latest parallel arrival, preserves each channel and learns their mix. '
             'Heads need not arrive simultaneously. This is implemented in the CPU emulator; execution there is serial.'),
            ('figure',('parallel_temporal_heads',174)),
            ('figure',('parallel_temporal_head_pilots',160)),
            ('table',(['Ours: heads / update','Fit / passes','Dev bpc ↓','Whole fit GFLOPs ↓'],[
                [f"H{r['args']['heads']} / U{r['args'].get('update_targets',16)} / lr {r['args']['lr']:g}",
                 f"{r['args']['fit']:,} / {r['args']['epochs']}",f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}"] for r in display],
                [62,37,30,45])),
            ('small','Payload 32 per head: H2 total width 64, H4 total width 128; eight blocks. More heads also '
             'increase capacity, and source/channel dynamics differ from the old single-head model. '
             'The baseline H2 pilot selects epoch 2 and overfits later; no head-count quality benefit is established. '
             'Adam interval U is separate from 16-character credit. Training reads admitted losing values '
             'and charges gradients, clipping and optimizer work. Contracts pass for causality, independent '
             'projections, evolving channels, all-head gradients and exact next-update recovery. '
             'All completed variants remain in the ledger; this table shows the latest four records.'),
        ])
    scaled_heads=[r for r in head_rows if r['args'].get('update_targets')==128 and
        r['args']['lr']==.004 and r['args']['seed']==6 and r['args']['dev']==8192 and r['args']['fit'] in (2048,8192)]
    if any(r['args']['fit']==8192 for r in scaled_heads):
        pages.append([
            ('h1','Appendix B (continued). Ours: completed head/data scaling'),
            ('p','Fixed d32 per head, eight blocks, pool2, credit16, U128/lr.004, four fitting passes, seed6. '
             'Every point scores the same 8,191 frozen development targets. More data improves these '
             'configurations, while four heads increase both width/capacity and fitting cost.'),
            ('figure',('parallel_head_data_work',174)),
            ('table',(['Ours: heads / fit','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓'],[
                [f"H{r['args']['heads']} / {r['args']['fit']:,}",f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/r['work']['fitting_targets']/1e6:.3f}"]
                 for r in sorted(scaled_heads,key=lambda r:(r['args']['heads'],r['args']['fit']))],
                [50,30,46,48])),
            ('p','The earlier 8K single-head controls reach receiver 3.311 and indexed KV 3.357 bpc. '
             'The multihead construction also changes source/channel dynamics and total width, so this '
             'is not a pure head-count ablation. The completed parallel-head 8K results missed the declared '
             '0.10 bpc tolerance of the indexed control; the campaign stopped before 32K/131K promotion. '
             'Route-credit fidelity, recurrent/channel conditioning, candidate coverage and optimization '
             'are diagnosis targets. These results constrain this implementation rather than the whole substrate.'),
            ('small','One seed and small fitting budgets. Whole fitting includes all four passes, backward, '
             'admitted losing-value credit and optimizer work. Development selection uses the lowest full '
             'development loss over those passes. Logical FLOPs and unit-weight special functions do not '
             'measure wall time, physical traffic or energy; no language supremacy follows from these points.')])
    long_credit=[r for r in tasks['episodic_language'] if r['args'].get('chunk',16)>16]
    for r in long_credit:
        a=r['args'];fields=('heads','payload','depth','pool','matching','recent','fit','dev','epochs','update_targets','warmup_targets','seed','lr')
        matched=[v for v in tasks['episodic_language'] if 'write_credit' not in v['args'] and v['args'].get('chunk',16)==16 and v['args'].get('arrivals',1)==a.get('arrivals',1) and all(v['args'].get(k)==a.get(k) for k in fields)]
        rows=matched+[r]
        pages.append([
            ('h1','Appendix B (continued). Ours: longer temporal credit'),
            ('p',f"Independent H{a['heads']} heads, d{a['payload']}/head, {a['depth']} event blocks, pool{a['pool']}. "
             f"{a['fit']:,} fitting characters / {a['epochs']} passes; {a['dev']-1:,} frozen development targets, seed{a['seed']}. "
             f"Adam uses U{a['update_targets']} / lr{a['lr']:g}. Graphs remain live for {a['chunk']} targets before detachment. "
             'Forward stored history and the inference architecture are retained.'),
            ('table',(['Ours: credit','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Inference MFLOPs / char ↓'],[
                [str(v['args']['chunk']),f"{v['final']['dev']['bpc']:.3f}",f"{v['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{v['work']['cpu_emulator']['total_training_unit_special_flops']/v['work']['fitting_targets']/1e6:.3f}",
                 f"{(v['work']['cpu_emulator']['inference_arithmetic_flops_per_character']+v['work']['cpu_emulator']['inference_special_functions_per_character'])/1e6:.4f}"] for v in rows],
                [28,26,35,40,45])),
            ('p','Sealed historical keys and values still affect predictions, but detachment removes '
             'later loss paths to their old producers. Longer credit restores those paths for more writes '
             'inside each optimizer window; it does not backpropagate through unlimited history. '
             'Temporal races, sparse receiver commits, separate Q/K/V and losing-route credit remain active.'),
            ('small','Matched 16-credit records are shown when completed under identical settings. '
             'Additional backward/normalization/clip/Adam work is counted and peak memory is guarded. '
             'Inference traces average different representative spans; the operation definitions and '
             'inference architecture are the same. One seed, development selection, no frontier claim. '
             'Forward-partition equality and full 64-credit gradient/update contracts precede fitting.')])
    pages.append([
        ('h1','Appendix B (continued). What most reduces research uncertainty'),
        ('p','The main direction now tests native content-and-time computation directly. '
         'Episodic race attention remains a preserved comparison. Its small language improvements '
         'do not yet establish that an attention scaffold is the best use of this substrate.'),
        ('table',(['Priority','Experiment','Doubt resolved'],[
            ['1','Integrated order/time learning; refitted credit/time controls; three seeds','Can deep sparse temporal state learn useful representations?'],
            ['2','Native-core text8 adapter and small-to-larger data ladder','Does the native construction learn economically without a KV attention bank?'],
            ['3','Occupy 4, 16, 64 stream states at fixed event/query budgets','Does useful state grow without proportional per-event activity?'],
            ['4','Chronological real streams and predict-before-update adaptation','Does the advantage survive real data and online change?'],
            ['5','Timestamp-aware AWS controls, equal-budget/quality curves','Is the quality/resource advantage reproducible?'],
            ['6','Whole-system FPGA/ASIC timing, traffic and energy measurement','Does the physical substrate deliver the projected savings?']], [19,87,68])),
        ('p','The first native branch uses eight event blocks, two independent receiver heads, '
         'observed source addresses, persistent content/state, analytic temporal evolution and '
         'counterfactual learning. It has no per-position KV attention. Other sources keep independent '
         'progress; source-local causal dependencies and internal joins remain charged.'),
        ('p','The same native core receives token content for the language test. Sharing receiver maps '
         'across tokens changes capacity and parameter exposure; it is a whole-construction comparison, '
         'not an isolated attention-removal ablation. Persistent state is not a full-cache equivalence claim.'),
        ('small','Contracts/smokes precede fixed-budget pilots, conditional capacity/data scaling and '
         'independent replication. Whole fitting, per-target work, inference, occupancy, memory and '
         'confidence intervals are published from completed files. Synthetic learning, real-data '
         'Pareto advantage and physical joules are separate milestones. The executable protocol, '
         'gates and current host limitations are documented in experiments/RESEARCH_VALUE_PLAN.md.')])
    for begin in range(0,len(tasks.get('delay_language',[])),4):
        rows=tasks['delay_language'][begin:begin+4]
        reference=[r for r in tasks.get('native_language',[]) if r['args']['fit'] in {v['args']['fit'] for v in rows}]
        rows=reference[:1]+rows
        pages.append([
            ('h1','Appendix B (continued). Ours: content-gated temporal reception'),
            ('p','Native eight-block independent-head models reuse key/query matches for two/four additional '
             'scalar clock policies. Local rotating/decaying clock vectors gate content-dependent projections '
             'and compose the next message. The waiting control retains the same clocks and joins but removes '
             'temporal reception/readout. The native parent has no extra clock branch.'),
            ('figure',('delay_language_quality_work',145)),
            ('table',(['Ours','Fit chars / passes','Dev bpc ↓','CPU fit GFLOPs ↓','Fit MFLOPs / target ↓','Infer MFLOPs / char ↓'],[
                ['Native' if 'clock_features' not in r['args'] else f"R{r['args']['clock_features']} {r['args'].get('clock_allocation','uniform')}{'' if r['args']['clock_readout'] else ' wait'}",
                 f"{r['args']['fit']:,}/{r['args']['epochs']}",f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/r['work']['fitting_targets']/1e6:.3f}",
                 f"{(r['work']['cpu_emulator']['inference_arithmetic_flops_per_character']+r['work']['cpu_emulator']['inference_special_functions_per_character'])/1e6:.4f}"] for r in rows],[33,32,23,28,30,28])),
            ('table',(['Ours','Projected whole fit GFLOPs','Receivers / commits per token','Matches / clocks per token','Parameters'],[
                ['Native' if 'clock_features' not in r['args'] else f"R{r['args']['clock_features']}/{r['args'].get('clock_allocation','uniform')}/{'on' if r['args']['clock_readout'] else 'waiting'}",
                 f"{r['work']['projected_event_architecture']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['args']['depth']*r['args']['heads']*r['args']['pool']}/{r['args']['depth']*r['args']['heads']}",
                 f"{r['args']['depth']*r['args']['heads']*r['args']['pool']}/{r['args']['heads']*(r['args']['depth']+sum(r.get('protocol',{}).get('clock_counts',[r['args'].get('clock_features',0)]*r['args']['depth'])))}",f"{r['parameters']:,}"] for r in rows],[36,37,35,35,31])),
            ('small','Completed fits only; identical frozen 8,191-target development protocol, four passes, '
             'U64/lr.002/warm512, ordinary credit16. Different fitting sizes are explicitly marked. All scalar '
             'policies, projections, temporal bases, gates, counterfactual content teachers and actual Adam '
             'remain charged. Projected arithmetic removes only numeric clock simulation. Physical rate '
             'setting, clock circuits, traffic, precision and measured joules remain separate. Exploratory '
             'development results; these do not alone establish comparable-quality Transformer superiority.')])
        if reference:
            base=reference[0]
            matched=[r for r in rows if r is not base and all(r['args'].get(k)==base['args'].get(k)
                     for k in ('fit','dev','payload','epochs','depth','heads','pool','seed','chunk',
                               'update_targets','warmup_targets','lr'))]
            if matched:
                best=min(matched,key=lambda r:r['final']['dev']['bpc'])
                delta=best['final']['dev']['bpc']-base['final']['dev']['bpc']
                ratio=best['work']['cpu_emulator']['total_training_unit_special_flops']/base['work']['cpu_emulator']['total_training_unit_special_flops']
                pages[-1].append(('small',f'Current matched-fit interpretation: best completed added-clock row is '
                    f'{abs(delta):.4f} bpc {"worse" if delta>=0 else "better"} than native, with '
                    f'{100*abs(ratio-1):.2f}% {"more" if ratio>=1 else "less"} fitting work. '
                    'Pending allocations and waiting controls cannot establish a benefit yet.'))
    if tasks.get('count_carrying_language'):
        def work_cells(r):
            w=r['work']['cpu_emulator']
            return [f"{w['total_training_unit_special_flops']/1e9:.3f}",
                    f"{w['total_training_unit_special_flops']/r['work']['fitting_targets']/1e6:.3f}",
                    f"{(w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'])/1e6:.4f}"]
        count_rows=[];seen_sizes=set()
        def variant(r):
            g,m=r['args'].get('escape_gate'),r['args'].get('count_message')
            core='' if (r['args']['payload'],r['args']['depth'])==(16,8) else f" [minimal core p{r['args']['payload']}/d{r['args']['depth']}]"
            credit='' if r['args']['chunk']==16 else f" [credit{r['args']['chunk']}]"
            return (' + gate + message' if g and m else ' + escape gate' if g else ' + count message' if m else '')+core+credit
        ordered=sorted(tasks['count_carrying_language'],key=lambda r:(r['args']['fit'],variant(r)))
        for position,r in enumerate(ordered):
            N=r['args']['fit'];first=N not in seen_sizes;seen_sizes.add(N)
            if first:
                for ref in tasks.get('native_language',[]):
                    if ref['args']['fit']==N:
                        count_rows.append([f"Native alone {N:,}",f"{N:,}/{ref['args']['epochs']}",f"{ref['final']['dev']['bpc']:.3f}",*work_cells(ref)])
            count_rows.append([f"Count-carrying native K{r['args']['orders']}{variant(r)} {N:,}",f"{N:,}/{r['args']['epochs']}",f"{r['final']['dev']['bpc']:.3f}",*work_cells(r)])
            if not variant(r):
                count_rows.append([f"Same, untrained base {N:,}",f"{N:,}/0",f"{r['initial_dev']['bpc']:.3f}",'Not trained','Not trained',work_cells(r)[2]])
            if position+1<len(ordered) and ordered[position+1]['args']['fit']==N:
                continue  # count references follow the last row of each fitting size
            refs=[row for path in sorted((RES/'count_reference').glob('*language_*.json'))
                  for row in json.loads(path.read_text())['rows'] if row['fit']==N]
            kn=min((row for row in refs if row['method']=='kn' and not row['adaptive']),key=lambda row:row['bpc'],default=None)
            ad=min((row for row in refs if row['adaptive']),key=lambda row:row['bpc'],default=None)
            for label,row in (('KN counts, frozen',kn),('Counts, stream-adaptive',ad)):
                if row:
                    if row.get('method')=='kn_interpolated':
                        label='Calibration ceiling: adaptive interpolated KN'
                    count_rows.append([f"{label} o{row['order']}",f"{N:,}/1",f"{row['bpc']:.3f}",'Not FLOPs','Not FLOPs','Not FLOPs'])
        pages.append([
            ('h1','Appendix B (continued). Ours: count-carrying native receivers'),
            ('p','The unchanged native eight-block core supplies the base predictive; addressed context-suffix '
             'receivers of orders 1..K carry sufficient statistics and deliver by an escape-race cascade with '
             'learned discount and concentration (Theory §§376–380, 387). The escape-gate variant makes discount and '
             'concentration per-position functions of the native predictive and count evidence; the count-message '
             'variant adds the counts to the base logits (Theory §389). Fitting counts are leave-one-out; '
             'development counts are prequential persistent state with frozen weights. Count tables are '
             'capacity; each target touches K addresses.'),
            ('table',(['Model','Fit chars / passes','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Infer MFLOPs / char ↓'],
             count_rows,[44,27,20,27,28,27])),
            ('small','Same 8,191 development targets for every row; one seed. Count increments/lookups are '
             'integer table work reported in the result files, not FLOPs. The initialized-base/escape row measures '
             'whole-model fitting benefit; it does not isolate the native base. Count rows are dev-selected-order references, not neural '
             'controls. How to read these rows (Theory §§393–394): at a few thousand to tens of thousands of characters, '
             'smoothed counting is a near-optimal estimator, and no learner (Transformers included) is expected to exceed it. '
             'At 10M characters the repository\'s dense Transformer control is still .12 bpc worse than frozen counts (§381). '
             'The strongest such reference, stream-adaptive interpolated Kneser–Ney, scores 2.521 / 2.414 / 2.271 bpc at '
             '2K / 8K / 32K and 2.101 at 131K (results/count_reference/curie_adaptive_kn_language_reference_20261002T131500Z.json). It is a calibration ceiling, not a competitor. Distances to it '
             'measure remaining smoothing, and the rows here are mechanism diagnostics, not a verdict on the architecture, '
             'whose claims are tested on tasks where learning matters (§394). Exploratory development evidence.'),
            *[('small',f'At {r["args"]["fit"]:,} fitting characters, fitting the native base and escape parameters '
                f'improves {r["initial_dev"]["bpc"]-r["final"]["dev"]["bpc"]:.4f} bpc over their untrained initialization. '
                'The complete composed predictor improves over native-alone, while this smaller learning contribution '
                'is the relevant comparison for the cost of fitting the base. The integer count path remains charged separately.')
              for r in tasks['count_carrying_language']]])
    for comparison in tasks.get('count_credit_pair',[]):
        ledger=comparison['common_unit_ledger'];quality=[];activity=[]
        for r in ledger:
            name=('Ours: full core' if r['arm'].startswith('full') else 'Minimal core control')+f" /credit{r['credit_targets']}"
            quality.append([name,f"{r['bpc']:.4f}",f"{r['cpu_whole_fit_gflops']:.3f}",
                f"{r['cpu_fit_mflops_per_target']:.3f}",f"{r['cpu_inference_mflops_per_character']:.4f}"])
            activity.append([name,f"{r['parameters']:,}",str(r['core_receiver_slots']),
                f"{r['selected_core_updates_per_target']:.0f}/{r['core_key_scores_per_target']:.0f}/{r['counterfactual_values_per_target']:.0f}"])
        pages.append([('h1','Appendix B. Longer credit versus learned count smoothing'),
            ('p','Completed four-arm integrated comparison, seed6: same2K fitting characters/four passes, '
             '8,191 development targets, K4 escape gate, U64/lr.002/warmup512. Each fit processes8,188 targets '
             'and128 optimizer updates. Full H2/d16/depth8 and minimal H2/d2/depth1 each compare16 versus64 '
             'tokens of graph reach; persistent state and forward mechanisms remain unchanged within each core.'),
            ('table',(['Model/credit','Dev bpc','Whole fit GFLOPs','Fit MFLOPs/target','Infer MFLOPs/char'],quality,[54,25,30,32,32])),
            ('table',(['Model/credit','Parameters','Core state slots','Updates/scores/teacher values per target'],activity,[54,33,31,55])),
            ('p',f"Full-core longer-credit gain{comparison['full_core_credit_gain_bpc']:.4f}bpc; "
             f"minimal-core gain{comparison['minimal_core_credit_gain_bpc']:.4f}bpc. "
             f"Full versus minimal advantage at64:{comparison['full_core_advantage_bpc_at64']:.4f}bpc, "
             f"with{comparison['full_core_fitting_work_ratio']:.3f}× full-core fitting work versus16. "
             f"The predeclared follow-up gate {'passes' if comparison['followup_gate_passed'] else 'fails'}."),
            ('small','All neural fitting forward/loss/backward/normalization/clipping/Adam and admitted losing-value '
             'credit are charged in CPU emulator units; projected clockless work is a separate result ledger. '
             'Each target also looks up4 count addresses; integer counts, discovery, traffic and energy stay separate. '
             'This reuses development data and is a single-seed screen, not confirmation, semantic-feature proof or supremacy. '
             'Passing requires at least.02bpc full-core credit gain AND at least.02bpc advantage over the matched '
             'minimal64 core at no more than2× full-core fitting work. No automatic larger fit.')])
    for audit in tasks.get('deep_core_attribution',[]):
        rows=[]; credits=[]
        for r in audit['models']:
            label='Full core' if '_full_' in r['result'] else 'Minimal control'
            intact=r['interventions'][0]['composed']['bpc']
            for arm in r['interventions'][:4]:
                name=arm['intervention'].replace('erase_','Erase ').replace('_',' ')
                rows.append([label+'/'+name,f"{arm['composed']['bpc']:.4f}",
                             f"{arm['base']['bpc']:.4f}",f"{arm['composed']['bpc']-intact:+.4f}"])
            c=r['credit']; resp=c['direct_residual_responsibility']
            credits.append([label,f"{resp['mean']:.4f}",f"{resp['quantiles'][1]:.4f}",
                            f"{c['dynamic_to_standalone_gradient_ratio']:.3f}"])
        pages.append([('h1','Appendix B. Does the trained deep state contribute?'),
            ('p','Frozen saved full/minimal64-credit models from the matched2K fits above. '
             'Same32 development targets at positions128–159, same race noise and causal count vectors. '
             'Erase stored content once at the slice start, retaining arrival times and absolute count cursor; '
             'state may rebuild. The learned escape gate responds to the changed neural base.'),
            ('table',(['Model/intervention','Composed bpc','Raw base bpc','Change in composed bpc'],rows,[72,31,31,39])),
            ('p','Full-core stored content helps this slice by.0481bpc, mainly through the carried source context; '
             'erasing receiver content alone does not hurt. Useful recurrence is present, but this does not '
             'establish hierarchical semantics, generalization across slices, or a full-development lead. '
             'Per-layer receiver erasures are preserved in the diagnostic JSON.'),
            ('table',(['Model','Mean base responsibility','Median base responsibility','Dynamic/base-only gradient norm'],credits,[40,42,42,49])),
            ('p','Credit diagnostic uses64 fitting targets at positions64–127. With gate values fixed, the exact '
             'base-logit derivative is responsibility × (q−onehot); numerical errors are below5e−9. '
             'Actual dynamic-gate gradients reach every layer. Responsibility suppression is correct mixture '
             'credit, not a demonstrated autograd bug; Adam can compensate for uniform scaling. The distribution '
             'and usefulness of the residual signal, information retention and retrieval remain hypotheses.'),
            ('small','No optimizer steps or weight changes. Raw base is trained as a conditional residual, so '
             'its standalone bpc is diagnostic. Audit12.85s/522,736KiB; no FLOP or superiority claim. '
             'Source: local_deep_core_attribution_20261002T075000Z.json. The synthetic long-range queue was '
             'retired before launch after unsupported chance claims and credit/update confounding were found; '
             'repaired-driver contracts require matched optimizer windows and separately measured local controls.')])
    for audit in tasks.get('deep_memory_preflight',[]):
        rows=[];capacity=[]
        for r in audit['common_unit_ledger']:
            rows.append([r['model'],f"{r['dev_bpc']:.4f}",f"{r['cpu_whole_fit_gflops']:.5f}",
                         f"{r['cpu_fit_mflops_per_target']:.4f}",f"{r['cpu_inference_mflops_per_target']:.4f}"])
            capacity.append([r['model'],str(r['core_receiver_capacity']),
                f"{r['selected_core_updates_per_target']}/{r['scored_core_keys_per_target']}/{r['admitted_counterfactual_values_per_fit_target']}",
                f"{r['occupied_context_slots']}/{r['context_slot_capacity']}",f"{r['persistent_tensor_bytes']:,}"])
        pages.append([('h1','Appendix B. Integrated memory repair: accounting prerequisite'),
            ('p','Completed short accounting smokes, not quality pilots. Unchanged full native core versus '
             'context-addressed outcome slots: H2/d16/depth8, same192 fitting characters/one pass,191 targets, '
             'three U64 Adam updates (partial63), c16/lr.002/no warmup, seed6, same core initialization and '
             'training race RNG. Frozen cold-state development has128 targets. Both zero-repair initial scores '
             'match exactly; every actual fitting operation is traced.'),
            ('table',(['Model','Smoke dev bpc','Whole fit GFLOPs','Fit MFLOPs/target','Infer MFLOPs/target'],rows,[35,32,34,36,36])),
            ('table',(['Model','Core slots','Updates/keys/teacher values per target','Extra slots occupied/capacity','State tensor bytes'],capacity,[27,21,51,43,31])),
            ('p',f"Addressed whole-fitting work ratio{audit['whole_fit_work_ratio']:.4f}×; "
             f"observed smoke gain{audit['smoke_bpc_gain']:.4f}bpc. Three updates are insufficient to assess "
             'useful deeper features. Fixed hashed addresses test memory capacity and evidence paths; they '
             'do not implement learned context pooling or KV race attention. Native temporal races, evolving '
             'messages, receiver key/value separation and unrealized-route surrogate credit remain.'),
            ('p','Full-shape addressed/tapped contracts pass exact zero forward/parent gradients, trained '
             'causality/chunk invariance, target-weighted normalization and serialized next-Adam predictions, '
             'parameters and moments. Context-slot and tap-buffer tensors are now counted and detached. '
             'Whole-driver recovery/selection and consistent warm-state comparisons remain admission gaps '
             'before long fits; no automatic scale-up.'),
            ('small','CPU emulator, 2FLOPs/MAC plus unit-weight specials, all fitting forward/loss/backward/'
             'normalization/clipping/Adam; excludes development, RNG, hash/integer operations and traffic. '
             'Inference is one accounted target after127 warm tokens, not throughput. Activity columns are '
             'derived from fixed code/dimensions. Empty text filler-group NaN in original JSONs is undefined, '
             'not a score; preserved beside correction to null in the driver. Cold neural memory differs '
             'from fit-prefilled count references. Source: local_deep_feature_preflight_20261002T081500Z.json.')])
    for comparison in tasks.get('value_credit_pair',[]):
        labels={'native_full':'Native full p16/L8','addressed_full':'Stored projection p16/L8',
            'late_full':'Late projection p16/L8','late_shallow':'Late p16/L1','late_minimal':'Late p2/L1'}
        quality=[];work=[];capacity=[]
        for r in comparison['common_unit_ledger']:
            label=labels[r['arm']]
            quality.append([label,f"{r['cold_bpc']:.4f}",f"{r['frozen_fit_replay_bpc']:.4f}",str(r['selected_epoch'])])
            work.append([label,f"{r['cpu_whole_fit_gflops_estimate']:.3f}",f"{r['cpu_fit_mflops_per_target_estimate']:.3f}",
                f"{r['cpu_inference_mflops_per_target']:.4f}",f"{r['replay_cpu_gflops_estimate']:.3f}",
                f"{r['warm_inference_cpu_mflops_per_target']:.4f}"])
            capacity.append([label,str(r['core_capacity']),f"{r['cold_final_state'].get('context_slots',0)}/{r['context_capacity']}",
                f"{r['selected_updates_per_target']:.0f}/{r['key_scores_per_target']:.0f}/{r['teacher_values_per_fit_target']:.0f}",
                f"{r['context_reads_per_target']:.3f}/{r['context_writes_per_target']:.3f}"])
        fig=comparison['args']['tag'].replace('local_value_credit_analysis_','value_credit_learning_')
        pages.append([('h1','Appendix B. Restoring addressed value credit: completed fits'),
            ('p','Five matched integrated arms:1,024 fitting characters/four passes,4,092 targets/64 Adam '
             'updates,2,047 dev targets, H2/pool2/c16/U64/warm512/lr.002/seed6. Full native temporal core '
             'versus original projected-value slots, late-projected raw-feature slots and two shallow controls. '
             'Core initialization and race RNG are matched. Minimum cold dev bpc over fixed passes selects '
             'each checkpoint; frozen fit replay is secondary and never selects weights.'),
            ('table',(['Model','Cold dev bpc','Fit-replay dev bpc','Selected pass'],quality,[65,34,43,31])),
            ('figure',(fig,173)),
            ('p',f"Late-full gain over native:{comparison['late_vs_native_full_bpc']:+.4f}bpc; "
             f"over stored projection:{comparison['late_vs_addressed_full_bpc']:+.4f}; "
             f"over minimal:{comparison['late_full_vs_minimal_bpc']:+.4f}; "
             f"over same-width shallow:{comparison['late_full_vs_same_width_shallow_bpc']:+.4f}. "
             f"The predeclared nomination gate {'passes' if comparison['followup_gate_passed'] else 'fails'}."),
            ('small','Late projection preserves fixed-weight computation by linearity and restores fixed-feature '
             'projection credit from old detached slots. It does not restore historical core-producer credit '
             'or add linear-reader expressivity. Fixed hash addresses are not learned pooling/KV race attention. '
             'The same-width shallow arm isolates depth better than the width-changing minimal control. '
             'Reused dev and one seed: exploratory quality/learning evidence, not semantic proof or supremacy.')])
        pages.append([('h1','Appendix B. Value-credit resources and protocol boundaries'),
            ('table',(['Model','Whole fit GFLOPs est.','Fit MFLOPs/target est.','Cold infer MFLOPs/target','Replay GFLOPs est.','Warm infer MFLOPs/target'],work,[47,24,27,25,25,25])),
            ('table',(['Model','Core slots','Extra occupied/capacity','Updates/keys/teacher values per fit target','Context reads/writes per fit target'],capacity,[47,20,31,39,36])),
            ('p','Every column uses the same unit/denominator for all five models. Whole fitting costs are '
             'representative first/mature/partial-window estimates from actual forward/loss/backward/'
             'normalization/clipping/Adam audits, with admitted losing-value work charged.2FLOPs/MAC plus '
             'unit-weight specials; CPU emulator here. Projected clockless costs are separate in the '
             'completed JSON ledger. Variable retrieval occupancy and graph reach are not fully enumerated.'),
            ('p','Warm evaluation replays all1,024 fit tokens with frozen selected weights, then carries '
             'predictive state/time into dev. Hash history and previous-address are cleared at the boundary '
             'to forbid an invented cross-split outcome. Dev race noise is coupled to cold scoring. Replay '
             'cost is extra work, estimated from first/last replay chunks; it is neither fitting FLOPs nor '
             'free access to historical data. Neural slots update causally during dev with frozen parameters.'),
            ('p',f"Late/original full fitting work ratio:{comparison['late_vs_original_fitting_work_ratio_estimate']:.3f}×. "
             'All direct/core and additional memory work must earn predictive value. No automatic scale-up. '
             'The nomination gate requires at least.02bpc cold gains against all four controls and no more '
             'than2× original-full fitting work under this common estimate convention.'),
            ('small','Hash/integer bookkeeping, RNG, traffic, Python metadata and physical energy remain '
             'separate. Cold/replay scores here use2,047 dev targets and cannot be directly juxtaposed with '
             'saved8,191-target count/dense references. Original-minimal projection contrast remains open; '
             'AWS integrated capacity/exposure work remains independent. Complete work/activity/storage '
             'and exact-source checkpoint provenance are retained in local_value_credit_analysis_20261002T090800Z.json.')])
    for audit in tasks.get('value_credit_frozen',[]):
        fusion=[];sensitivity=[]
        for r in audit['models']:
            if 'fusion' in r:
                f=r['fusion'];unit=lambda t:t['arithmetic_flops']+t['special_function_evaluations']
                fusion.append([r['arm'],f"{f['unfused_cpu_unit_special_flops_per_target']/1e3:.3f}",
                    f"{f['fused_cpu_unit_special_flops_per_target']/1e3:.3f}",str(f['removed_parameters']),
                    f"{unit(f['fold_arithmetic_trace'])/1e3:.3f}",f"{f['break_even_occupied_reads']:.0f}"])
            sensitivity.append([r['arm'],*[f"{x['mean_paired_query_kl_nats']:.2e}" for x in r['prefix_information']['rows']]])
        pages.append([('h1','Appendix B. Fixed-feature credit and compiled inference'),
            ('p','Frozen-reader compilation folds A=R_vW once and removes the unused W matrix. Coupled '
             'predictions and raw-slot state preserve the unfused construction within numerical tolerance. '
             'Under2FLOPs/MAC, fold2d³ replaces2d² per occupied read, breaking even after d occupied reads. '
             'This changes deployment work, not the fitting optimizer; compiled models refuse training.'),
            ('table',(['Late model','Unfused infer KFLOPs/target','Fused infer KFLOPs/target','Removed weights','Fold KFLOPs','Break-even occupied reads'],fusion,[32,29,29,25,28,30])),
            ('p','Inference samples include16 actual targets after128 warm tokens; the saved projection '
             'work equals2d² times observed occupied reads. Matrix-fold work is charged above; constructor '
             'initialization/copy/RNG are separate. This is arithmetic/weight reduction with frozen quality, '
             'not measured latency, energy or competitive superiority.'),
            ('table',(['Text-trained model','Prefix-pair KL, gap8','Prefix-pair KL, gap32','Prefix-pair KL, gap64'],sensitivity,[56,39,39,39])),
            ('p','Each parity diagnostic balances all four input bit pairs with identical noise/query suffix. '
             'Actual causal query count vectors are identical for orders1–8, but targets are opposite across '
             'paired prefixes. A predictor restricted to those suffix/count inputs has at least1bit target '
             'logloss; the full prefix determines parity exactly. Cue-free prefixes make the count contract '
             'explicit, replacing the retired synthetic all-orders independence assertion.'),
            ('small','These frozen text checkpoints were never trained on parity. KL and top-context '
             'differences diagnose prefix dependence/retention, not parity learning or semantic abstraction. '
             'Zero optimizer steps, parameter fingerprints unchanged. A later task fit needs matched '
             'full/shallow controls, new-prefix generalization and complete prefix computation/credit '
             'accounting. Historical feature-producer credit, learned address pooling and KV races remain '
             'open. Source: local_value_credit_frozen_20261002T090800Z.json; theory63 records exact scope.')])
        headrows=[]
        for r in audit['models']:
            fit=r['head_features']['fitting'];dev=r['head_features']['development']
            headrows.append([r['arm'],f"{fit['frozen_bpc']:.4f}",f"{dev['frozen_bpc']:.4f}",
                f"{fit['mean_head_weight_gradient_norm']:.3f}",f"{dev['feature_covariance_participation_rank']:.2f}"])
        counts=[]
        for order in (1,2,3,4):
            values=[]
            for method,adaptive in (('kn',False),('wb',False),('ad',False),('wb',True),('ad',True)):
                row=next(x for x in audit['count_controls'] if x['order']==order and x['method']==method and x['adaptive']==adaptive)
                values.append(f"{row['bpc']:.4f}")
            counts.append([str(order),*values])
        pages.append([('h1','Appendix B. Frozen feature geometry and matched count calibration'),
            ('table',(['Model','Frozen fit bpc','Cold dev bpc','Fit head gradient norm','Dev feature participation rank'],headrows,[43,31,31,33,35])),
            ('p','At the selected weights, replay1,023 fitting and2,047 development targets from cold state '
             'with the primary evaluation RNG. Capture actual head inputs, then compute fixed-feature '
             'linear-head loss derivatives and covariance in float64. No head fitting, optimizer step or '
             'parameter changes. Frozen dev scores reproduce the primary results within1e−5bpc.'),
            ('p','The participation rank is (trace C)²/trace(C²), a measure of feature geometry. Finite '
             'joint training need not make the final head stationary; its gradient norm and covariance '
             'rank alone do not identify a bug or semantic abstraction. Frozen fitting loss helps '
             'separate training fit from development generalization without using online training scores.'),
            ('table',(['Count order','Frozen KN bpc','Frozen WB bpc','Frozen AD bpc','Adaptive WB bpc','Adaptive AD bpc'],counts,[28,29,29,29,29,29])),
            ('p','Same1,024 fit characters and2,047 dev targets, reused count-reference implementation, '
             'fixed discount.75, all orders/methods shown separately. Count construction uses one pass; '
             'learned models use four gradient passes. Counts are fit-prefilled; adaptive variants also '
             'update from observed dev outcomes after prediction. These state/history policies differ '
             'from cold neural scoring; frozen neural fit replay is reported separately above.'),
            ('small','KN: interpolated Kneser–Ney; WB: Witten–Bell; AD: absolute discount. Integer represented '
             'count increments/lookups and measured construction/scoring wall time are in the completed '
             'audit, without conversion to neural FLOPs. Equal scalar loss calibrates prediction quality; '
             'it does not establish identical features or a same-quality compute advantage. No per-target '
             'order selection or official-test access. Full spectra, data/checkpoint hashes and diagnostic '
             'boundaries are retained in local_value_credit_frozen_20261002T090800Z.json.')])
    for r in tasks.get('balanced_joint',[]):
        quality=[];work=[]
        for row in r['common_unit_ledger']:
            quality.append([row['arm'],f"{row['query_bits']:.4f}",f"{100*row['accuracy']:.2f}%",str(row['selected_epoch'])])
            work.append([row['arm'],f"{row['whole_fit_gflops_estimate']:.4f}",f"{row['fit_mflops_per_target_estimate']:.4f}",
                f"{row['inference_mflops_per_target']:.4f}",str(row['state_tensor_bytes'])])
        pages.append([('h1','Appendix B. Beyond query counts: balanced joint learning'),
            ('p','Each episode observes a bit followed by its complement, then another such pair, eight shared '
             'noise symbols and a unique query cue. The target is the relation between the earlier bits. '
             'All four bit combinations occur for each noise suffix. Observed symbol totals and actual '
             'order-0..8 query count vectors match; all 36 declared WB/AD controls score one bit.'),
            ('table',(['Integrated model','Dev bits/query','Dev accuracy','Selected pass'],quality,[62,38,38,35])),
            ('figure',('report/figures/'+r['args']['tag']+'_learning.png',173)),
            ('p','The native and learned-delay variants fail the predeclared 75%/.8-bit dependency gate. '
             'Eight fitting suffix groups give 32 targets/pass; 16 passes,512 target presentations,64 Adam '
             'updates,64 held-out dev targets. All arms have query-only binary supervision and full '
             '15-event credit; the target is never an input. Noise couples the quartet and variants.'),
            ('small','The one-bit bound restricts the predictor to query suffix/count inputs, including root '
             'marginals. It excludes arbitrary inspection of the full prefix or other count addresses. '
             'Counts should lead in their supported local regime; this test asks for useful nonlocal '
             'prediction. One selected seed/development set, not semantic language or supremacy.')])
        pages.append([('h1','Appendix B. Balanced-core fitting and inference work'),
            ('table',(['Model','Whole fit GFLOPs est.','Fit MFLOPs/target est.','Inference MFLOPs/query','State tensor bytes'],work,[53,29,32,32,27])),
            ('p','All rows use 512 query-target presentations as the fitting denominator. Inference includes '
             'all 15 episode inputs and query loss. Full prefix forward/backward, losing receiver values, '
             'normalization/clipping and Adam are charged. First/last complete optimizer windows supply '
             'the whole-fit estimates; routing occupancy is not exhaustively traced.2FLOPs/MAC plus '
             'unit specials, including declared delay-index rounding. Actual keys/commits/input events '
             'and wall/RSS are saved separately.'),
            ('p','Delay taps add bounded per-layer input buffers, trainable delays and projections while '
             'retaining native temporal races and private persistent receivers. Zero taps exactly nest '
             'native prediction and parent gradients. Seven contracts include actual interrupted-driver '
             'and full-shape Adam recovery. The same-width shallow control retains these mechanisms '
             'but has fewer layers; initialization also changes, so this is an architecture comparison.'),
            ('small','Memory traffic, integer/index bookkeeping, RNG, checkpoint I/O and hardware energy '
             'remain outside floating arithmetic. More retention capacity or a valid derivative is not '
             'a completed quality advantage. All unsuccessful arms and previous language evidence '
             'remain preserved; full source/data/result hashes are in the completed analysis JSON.')])
    for r in tasks.get('joint_readout',[]):
        quality=[];work=[];bits=[]
        for row in r['common_unit_ledger']:
            score=row['scores']['fresh'];name=row['arm'].replace('_',' ')+' / '+str(row['degree'])
            quality.append([name,f"{row['fitting']['fitting_bits']:.3f}",f"{score['query_bits']:.3f}",f"{100*score['accuracy']:.2f}%"])
            work.append([name,f"{row['total_fit_gflops_estimate']:.4f}",f"{row['total_fit_gflops_estimate']*1000/32:.3f}",
                f"{row['inference_mflops_per_query_estimate']:.4f}"])
        for row in r['retention']:
            bits.append([row['arm'],*[f"{100*b['scores']['fresh']['accuracy']:.2f}%" for b in row['bit_probes']]])
        pages.append([('h1','Appendix B. Frozen retention versus interaction decoding'),
            ('table',(['Encoder / degree','Fit bits/query','Fresh bits/query','Fresh accuracy'],quality,[72,33,35,33])),
            ('table',(['Frozen encoder','First-bit probe accuracy','Second-bit probe accuracy'],bits,[73,50,50])),
            ('p','Freeze the selected encoders, plus native initial weights as a reservoir control. Fit '
             'zero-initialized affine or standard degree-2 polynomial residuals on actual query features. '
             'Fitting standardization only; four race-noise views of the same32 fitting episodes. '
             'Regularized full-batch L-BFGS, fixed lambda1e-5/100 iterations; reported derivatives '
             'do not establish convergence. New seed73001 has32 paired suffix groups/128 targets '
             'and never selects readout settings or weights.'),
            ('p','Neither readout generalizes; polynomial heads become badly overconfident. Separate '
             'one-bit probes also remain near chance. They use extra bit supervision only for diagnosis; '
             'their outputs never enter the relation predictor. These failures motivate evidence-access '
             'tests rather than assuming a larger decoder solves the current model.'),
            ('small','Zero core optimizer steps and encoder fingerprints preserved. Eight predeclared '
             'readout arms on one fresh synthetic distribution, not general confirmation or a semantic '
             'claim. Failed probes do not prove all information was erased or a universal learning ceiling.')])
        pages.append([('h1','Appendix B. Complete frozen-readout pipeline resource boundary'),
            ('table',(['Encoder / degree','Total fitting GFLOPs est.','MFLOPs/distinct fit query','Inference MFLOPs/query est.'],work,[72,32,35,34])),
            ('p','Every total includes the entire earlier encoder fit, four-view feature-prefix replay and '
             'actual residual-head optimization/feature construction. Initial weights pay zero encoder '
             'fitting. The common denominator here is32 DISTINCT fitting queries for every row, rather '
             'than differing L-BFGS closure counts. This denominator differs explicitly from the '
             'presentation-based table above. Closure target evaluations and exact head work are saved '
             'separately. The same feature replay is charged once to each hypothetical standalone arm.'),
            ('p','Every inference estimate includes processing the complete prefix plus the chosen '
             'query head. Encoder replay/inference estimates use coupled first-quartet operation audits; '
             'head construction, loss, backward and optimizer operations are traced directly. Separate '
             'retention-probe supervision/work is diagnostic overhead, not free parity training. '
             'Raw candidate/traffic/metadata/RNG costs and measured energy remain separate.'),
            ('small','Local vector products are supporting primitives, not substitutes for the native '
             'temporal core or evidence of learned KV routing. Numerical contracts verify polynomial '
             'adjoints, regularized fixed-feature gradients, zero nesting, causal integrated prediction '
             'and restored head outputs. Preserve these negative results beside the subsequent '
             'protected-memory hypothesis; no favorable cell is inferred from unfinished training.')])
    for r in tasks.get('joint_outcome',[]):
        quality=[];work=[];activity=[]
        for row in r['common_unit_ledger']:
            d=row['development'];f=row['fresh']
            quality.append([row['arm'],f"{d['query_bits']:.3f}",f"{f['query_bits']:.3f}",f"{100*f['accuracy']:.2f}%",str(row['selected_epoch'])])
            work.append([row['arm'],f"{row['whole_fit_gflops_estimate']:.4f}",f"{row['fit_mflops_per_target_estimate']:.4f}",f"{row['inference_mflops_per_target']:.4f}"])
            activity.append([row['arm'],f"{row['native_available_receivers']} / 27",f"{row['mean_occupied_outcome_addresses']:.2f}",
                f"{row['native_selected_updates_per_target']} / {row['raw_outcome_writes_per_target']}",
                f"{row['terminal_keys_per_target']:.2f} / 2",f"{row['terminal_counterfactual_loss_pairs_per_training_target']:.2f}"])
        pages.append([('h1','Appendix B. Protected outcomes and joint terminal races'),
            ('p','Each observed token writes its observed successor at its predecessor address, never a '
             'target-derived or hand-selected bit address. All occupied addresses are candidates. '
             'Two learned query/key exponential races deliver small interpreted symbol values to a '
             'generic bilinear query decoder. Native temporal computation, sparse vector receivers '
             'and earlier counterfactual learning remain. No XOR extraction enters prediction.'),
            ('table',(['Integrated arm','Selected dev bits','Reserved bits','Reserved accuracy','Selected pass'],quality,[48,31,31,34,29])),
            ('figure',('report/figures/'+r['args']['tag']+'_learning.png',173)),
            ('p',f"Reserved seed74001/32 groups/128 queries scores fixed selected weights after all fits; "
             f"no retuning. Joint-minus-local loss improvement:{r['joint_vs_local_fresh_bits']:+.3f} bits; "
             f"full-minus-shallow improvement:{r['full_vs_shallow_fresh_bits']:+.3f} bits. The joint-credit "
             f"gate {'passes' if r['joint_credit_gate_passed'] else 'fails'}; all arms remain visible."),
            ('small','One fitted seed and structured generator. Sixteen fit groups/64 queries,16 passes, '
             'U4/lr.01/p4/H2/pool2,fullL2/shallowL1. The one-bit bound applies to identical query '
             'suffix/count inputs including root marginals, not arbitrary inspection of other count '
             'addresses. No natural-language, dense-control or resource-supremacy claim.')])
        pages.append([('h1','Appendix B. Joint-credit work, capacity and activity'),
            ('table',(['Arm','Whole fit GFLOPs est.','Fit MFLOPs/query est.','Inference MFLOPs/query'],work,[51,40,40,42])),
            ('table',(['Arm','Core / raw address capacity','Mean raw occupied','Core commits / raw writes per query','Terminal keys / values per query','Training loss pairs per query'],activity,[34,29,22,34,28,26])),
            ('p','All fitting columns use1,024 query presentations,256 Adam updates and all15 observed '
             'prefix events per query. Whole-fit first/last-window estimates include native candidates, '
             'losing proposals, protected-state discovery, terminal decoder/loss, backward, clipping '
             'and Adam. Inference scores every occupied key and delivers only two values. Available '
             'addresses, occupied state, scored keys, commits, raw writes and value deliveries are '
             'distinct; sparse activity does not imply zero key or learning cost.'),
            ('p','Joint training enumerates C² terminal losses and differentiates their categorical '
             'expected risk, giving exact conditional terminal content-choice credit. Local training '
             'uses a sampled pair and the existing value-linearized race surrogate. Their initial '
             'forward predictions and inference policy match; learning estimators differ. Earlier '
             'native route surrogates, raw fixed-address writes and downstream timing credit remain '
             'separate limitations. No exact whole-sequence gradient is claimed.'),
            ('small',f"Joint/local whole-fitting work ratio:{r['joint_local_fitting_work_ratio_estimate']:.3f}. "
             '2FLOPs/MAC plus unit specials; traffic, raw integer state, Python objects, RNG and energy '
             'separate. Five contracts and three accounting smokes precede fits. The same-width shallow '
             'comparison changes initialization too. A successful protected-state read does not prove '
             'learned context pooling, arbitrary-distance KV retrieval or useful deep producer credit.')])
    for r in tasks.get('joint_replication',[]):
        quality=[];work=[];comparisons=[]
        for row in r['common_unit_ledger']:
            label=f"s{row['seed']} / {row['read_credit']}"
            quality.append([label,f"{row['development']['query_bits']:.3f}",f"{row['fresh']['query_bits']:.3f}",
                f"{100*row['fresh']['accuracy']:.2f}%",str(row['selected_epoch']),
                f"{100*row['zero_delivered_value']['accuracy']:.2f}%"])
            work.append([label,f"{row['whole_fit_gflops_estimate']:.4f}",f"{row['fit_mflops_per_target_estimate']:.4f}",
                f"{row['inference_mflops_per_target']:.4f}",f"{row['wall_s']:.1f}"])
        for pair in r['comparisons']:
            interval=pair['paired_suffix_group_95_interval']
            comparisons.append([str(pair['seed']),f"{pair['joint_vs_local_fresh_bits']:+.3f}",
                f"[{interval[0]:+.3f}, {interval[1]:+.3f}]",f"{pair['joint_local_fitting_work_ratio_estimate']:.3f}"])
        pages.append([('h1','Appendix B. Fixed joint-credit confirmation'),
            ('table',(['Fit seed / credit','Selected dev bits','New suffix bits','New suffix accuracy','Pass','Zero-value accuracy'],quality,[38,27,27,30,18,33])),
            ('table',(['Fit seed','Joint improvement bits/query','Paired suffix interval','Joint/local fitting work'],comparisons,[25,48,50,50])),
            ('p','Two additional fitted seeds7/8 use the unchanged full p4/L2 configuration and16 passes. '
             'All four declared fits are shown. Development seed72001 selects the minimum across fixed '
             'passes; seed75001/64 suffix groups/256 targets is reserved for fixed selected models. '
             'No parameter, learning rate, stopping rule or evaluation setting is retuned on this set.'),
            ('p',f"{sum(x['joint_credit_gate_passed'] for x in r['comparisons'])} of2 declared joint-versus-local gates pass. "
             'A seed must reach at least75%/.8bits and improve local credit by.05bits. Paired suffix '
             'bootstrap intervals are conditional on the fitted seed and synthetic generator; they '
             'are not a broad confidence interval over learning algorithms or tasks.'),
            ('p','Initial predictions, race choices and probabilities match within each joint/local pair. '
             'Inference mechanisms are identical. The differing training risk/credit objectives therefore '
             'test this terminal learning intervention, while protecting all earlier negative evidence.'),
            ('p','Zero-value evaluation removes delivered content while keeping native context, query/key '
             'races and contextual decoder. No fitting or model selection; every parameter is restored '
             'and verified afterwards. Three arms retain98.8–100% accuracy: native context has learned '
             'useful nonlocal prediction. Joint seed8 drops to75%. Value delivery is therefore not '
             'necessary in every selected fit; neither its training benefit nor a depth premium is isolated.'),
            ('small','Query-suffix/count bound only. The protected bank has fixed observed predecessor '
             'addresses and no learned writes; the standard bilinear decoder and exact terminal content '
             'credit do not establish deep core learning, clock-gradient accuracy or natural-text gains.')])
        pages.append([('h1','Appendix B. Confirmation fitting and inference resources'),
            ('table',(['Seed / credit','Whole fit GFLOPs est.','Fit MFLOPs/query est.','Inference MFLOPs/query','Fit wall seconds'],work,[45,34,34,34,26])),
            ('p','Every row fits64 distinct queries for16 passes:1,024 target presentations and256 Adam '
             'updates, all15 prefix events per query. Whole-fit and per-target columns share that '
             'denominator. First/last complete optimizer-window estimates include prefix/native '
             'races, all terminal candidate values, joint pair losses where applicable, backward, '
             'normalization/clipping and Adam. Measured wall and RSS are separate from FLOPs.'),
            ('p','All models have8 native receivers plus27 raw outcome addresses. Each query causes '
             '60 native state commits and14 raw outcome writes. Terminal inference scores2C keys '
             'and delivers2 values. Joint training enumerates C² losses; both training arms inspect '
             'C candidate values. C depends on observed occupancy and is recorded in activity traces.'),
            ('p','Additional counterfactual arithmetic buys prediction improvement only where completed '
             'results support it. A quality advantage at this budget is not an iso-quality compute '
             'advantage; shorter or better-controlled local fits have not been optimized. No measured '
             'energy, candidate-discovery latency or hardware throughput claim is inferred.'),
            ('small','One guarded one-thread CPU job at a time, watchdog active,8GiB available-memory '
             'floor. Unique queues, checkpoints, source/data hashes and failed results preserved. '
             'This confirmation adds seed evidence for a small structured nonlocal relation, not '
             'architectural supremacy or superiority to an unrestricted count-memory algorithm.')])
    for r in tasks.get('uniform_address',[]):
        checks=[['Actual outcome bank','Uniform two-value law identical across second-bit alternatives'],
            ['Fixed zero-context risk',f"{r['fixed_context_stationary_risk_bits']:.6f}bits"],
            ['Maximum decoder gradient',f"{r['maximum_decoder_gradient']:.3e}"],
            ['Maximum policy gradient',f"{r['maximum_policy_gradient']:.3e}"],
            ['Analytic marker-policy witness',f"{r['analytic_marker_witness_max_expected_bits']:.3e}expected bits"]]
        pages.append([('h1','Appendix B. A terminal symmetry, not a learning impossibility'),
            ('table',(['Numerical contract','Completed result'],checks,[65,108])),
            ('p','At query time the four distinguished outcome addresses contain b1,b2,1−b2,n0. '
             'Their multiset is b1,0,1,n0. Remaining entries depend only on the shared noise suffix. '
             'Uniform independent value reads therefore have the same full joint distribution for '
             'both second-bit alternatives. If additional context also lacks that bit, no decoder '
             'of those values can predict the balanced relation better than one bit.'),
            ('p','At zero residual, every candidate pair has the parent logit. Exact terminal key '
             'credit is initially zero. With fixed zero-logit uninformative context, uniform policies '
             'and balanced examples, decoder gradients are zero too. The actual bank and decoder '
             'contracts verify this conditional stationary point in float64 on four suffix groups. '
             'This is not a stationary-point proof for the whole trainable native core.'),
            ('p','An explicit target-independent existence witness selects marker addresses24/25 '
             'and interprets symbols0/1 as signed values. A bilinear interaction can predict their '
             'relation with very small expected loss. The witness is analytic and never enters the '
             'fitted model or benchmark score. Finite score clamps leave a nonzero routing error '
             'floor, so unlimited decoder confidence need not reduce unconditional expected loss.'),
            ('p','The useful solution and terminal symmetry coexist. Random key contrast may break '
             'the alias, which is compatible with late fitted transitions but does not prove their '
             'cause. A new controlled conditioning study must retain all seeds and mechanisms; '
             'these fixed confirmation fits are neither extended nor tuned after observation.'),
            ('small','Theory70 states the information conditions and proof. Numerical result records '
             'source hashes, targets, wall and RSS. The obstruction concerns addressed value pooling '
             'and terminal initialization; it does not imply that all counting, temporal computation '
             'or architectural adaptation faces a mathematical ceiling.')])
    for r in tasks.get('joint_stateful_table',[]):
        scores=[]
        for split,score in r['final'].items():
            scores.append([split,str(score['targets']),f"{100*score['accuracy']:.3f}%",f"{score['nll']:.6f}",
                f"{100*(1-score['accuracy']):.3f}pp"])
        pages.append([('h1','Appendix B. Practical headroom: a stronger joint-event table'),
            ('table',(['Fixed table evaluation','Queries','Accuracy','NLL','Maximum accuracy headroom'],scores,[47,24,34,34,34])),
            ('p','This control consumes observed timestamps as well as text. It retains the complete '
             'question string and the latest observed time for each of four marks. A generic fit learns '
             'one mark-age split per question: which mark, threshold, split direction and both leaf '
             'probabilities. The generator word/mark mapping and recency threshold are not supplied.'),
            ('p',f"Same512 distinct fitting episodes/seed1301,one fitting pass,20 question strings; "
             f"fit wall{r['fitting_wall_s']:.6f}s,maxRSS{r['max_rss_kb']:,}KiB. "
             'Fixed learner; dev2301/256 and confirmation3301/1,024 episodes do not tune it. '
             'Contracts verify label mutation, appended future observations and common clock shifts '
             'cannot change causal prediction features.'),
            ('p','Less than half a percentage point of accuracy headroom remains. The proposed20-point '
             'learned-control advantage gate is impossible against this reference on these episodes, '
             'even for a perfect learner. A gain over the time-blind table would therefore measure '
             'extra supplied information rather than demonstrate the requested practical advantage.'),
            ('p','The joint recency task remains a capability and mechanism diagnostic. Likelihood may '
             'still improve; neither saturated accuracy nor tiny fitting cost supports general '
             'supremacy. Strong real-stream calibration with causal common inputs takes priority. '
             'Old successful mechanism evidence and negative results are preserved.'),
            ('small','Logical content inspections, timestamp writes, age subtraction, sorting comparisons, '
             'threshold candidates, count/probability work and wall are saved. They are not converted '
             'to neural FLOPs. Full question lookup is a task-specific conventional control, not a '
             'general language model or the integrated research architecture. Theory72 records scope.')])
    for r in tasks.get('dvs_practical_controls',[]):
        rows=[[x['arm'].replace('time_binned_naive_bayes','Calibrated counts'),
            f"{100*x['final']['accuracy']:.2f}",f"{x['final']['nll']:.4f}",
            f"{x['fitting_wall_s']:.3f}",f"{x['inference_wall_s']:.3f}"] for x in r['rows']]
        pages.append([('h1','Appendix B. Real gesture practical calibration'),
            ('p','A meaningful practical region:984 first-second fitting gestures from users1–19 and192 '
             'development gestures from users20–23. The common causal representation has4×4 spatial '
             'cells, two polarities and20 observed50ms count closures. No official test was opened.'),
            ('table',(['Control','Dev accuracy %','Dev NLL','Fit seconds','Dev inference seconds'],rows,[58,28,27,28,32])),
            ('p',f"The lowest development-NLL control is {r['selected_by_dev_nll']}; its accuracy exceeds "
             f"calibrated counts by{100*r['selected_control_accuracy_gain_over_counts']:.2f}points. Maximum "
             'grid accuracy is74.48%, so this observation protocol has headroom rather than the99.5% '
             'saturation of the cheap recency table. This admits an integrated learning test, not a '
             'native advantage claim. The older weak dense gesture screens are not the practical ceiling.'),
            ('p','All nine fixed learned cells are preserved. Linear/RBF controls use fitting-only '
             'log-count centering/scaling. Counts use all640 time bins and3-fold fitting-only '
             'alpha/temperature calibration; this avoids treating correlated camera events as '
             'independent label evidence. SVM probability calibration is included in measured fitting.'),
            ('small',f"Common raw preprocessing costs{r['preprocessing_wall_s']:.3f}s. The complete control "
             f"campaign takes{r['wall_s']:.3f}s with{r['max_rss_kb']/1024:.1f}MiB peak process RSS. "
             'Table inference uses the complete192-example batch. Third-party solver FLOPs are '
             'unmeasured, not zero; CPU wall, arithmetic, storage and energy are distinct. Development '
             'selection and one fitted seed are exploratory evidence, not independent confirmation.')])
    for r in tasks.get('dvs_practical_native',[]):
        def fmtwork(x):return 'Unmeasured' if x is None else f'{x:.3f}'
        rows=[[x['arm'].replace('time_binned_naive_bayes','Calibrated counts').replace('native_temporal_p16_L2_H2_pool2_seed6','Ours p16/L2/H2'),
            f"{100*x['development_accuracy']:.2f}",f"{x['development_nll']:.4f}",
            fmtwork(x['whole_fit_gflops_estimate']),fmtwork(x['fit_mflops_per_presentation_estimate']),
            fmtwork(x['inference_mflops_per_target_estimate'])] for x in r['common_unit_ledger']]
        native=r['common_unit_ledger'][-1]; workflow=r['workflow_comparison']
        pages.append([('h1','Appendix B. Integrated real-packet quality and work'),
            ('table',(['Model','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[48,23,23,26,27,26])),
            ('p','Same984 distinct fitting gestures and192 subject-disjoint development targets. '
             'Ours fits eight fixed passes:7,872 target presentations and496 Adam updates,U16, '
             'lr.003, seed6. Conventional solvers have their own convergence/calibration policies; '
             'equal pass or fitting-work protocols are not claimed. Every learned baseline is retained.'),
            ('p',f"Selected native pass{r['selected_epoch']}: accuracy difference versus minimum-NLL "
             f"control{100*r['native_accuracy_gain_over_selected_control']:+.2f}points, NLL improvement "
             f"{r['native_nll_gain_over_selected_control']:+.4f}. Development quality dominance flag: "
             f"{r['native_dominates_selected_control_in_development_quality']}. Independent practical "
             'advantage remains unproved; no pending, best-seed or official-test prediction fills this table.'),
            ('small','Native estimates include complete prefix/query forward/backward, all scored keys, '
             'candidate values, counterfactual value credit, normalization/clipping and Adam. '
             'First/last windows are sampled by actual16/8 target size;2FLOPs/MAC and unit-weight '
             'special functions. Solver arithmetic is unmeasured. Preprocessing, validation, traffic '
             'and measured CPU latency must be charged separately; isolated inference estimates '
             'cannot establish total resource or energy advantage.')])
        pages.append([('h1','Appendix B. Real-packet learning, capacity and total workflow'),
            ('figure',(r['args']['tag']+'_learning',145)),
            ('table',(['Completed native quantity','Value'],[
                ['Parameters / available receivers',f"{native['parameters']:,} / {native['available_receivers']}"],
                ['Persistent state tensor bytes',str(native['state_tensor_bytes'])],
                ['Keys / selected updates / counterfactual values per fitting prefix',
                 f"{native['key_scores_per_fit_presentation']:.1f} / {native['selected_updates_per_fit_presentation']:.1f} / {native['counterfactual_values_per_fit_presentation']:.1f}"],
                ['Fit plus nonfitting forward GFLOPs estimate',f"{r['native_fitting_plus_nonfitting_forward_gflops_estimate']:.3f}"],
                ['Native workflow / whole control grid seconds',f"{workflow['native_wall_s']:.3f} / {workflow['control_grid_wall_s']:.3f}"],
                ['Common raw preprocessing seconds',f"{workflow['raw_preprocessing_wall_s']:.3f}"],
                ['Native / grid workflow plus preprocessing seconds',f"{workflow['native_workflow_plus_preprocessing_wall_s']:.3f} / {workflow['control_grid_plus_preprocessing_wall_s']:.3f}"],
                ['Native peak process RSS MiB',f"{native['max_rss_kb']/1024:.1f}"]],[100,73])),
            ('p','The unchanged native core computes through temporal races, separate keys/values, '
             'sparse persistent receiver updates and local mixing. Nonempty observed packets arrive '
             'at their physical closure and an observed query at1s; state resets per gesture. '
             'Race noise depends on pass/fitting seed, never clip identity, index or label. '
             'Counts are observed camera content rather than fitted statistical prediction experts.'),
            ('p','Workflow wall includes fitting, validation, checkpoints and operation profiling; '
             'the complete control-grid wall includes all fixed solvers, calibration, scoring and '
             'serialization. Nonfitting neural forward work is estimated using completed prefix '
             'samples; raw preprocessing is common and added once to each pipeline. Small tensor '
             'state does not mean small process RSS, zero scoring cost, useful extra depth or '
             'zero optimizer work. Earlier numerical admission/research costs remain separately saved.')])
    for r in tasks.get('dvs_practical_inference',[]):
        rows=[[x['arm'].replace('time_binned_naive_bayes','Calibrated counts'),
            f"{100*x['quality']['accuracy']:.2f}",f"{x['quality']['nll']:.4f}",
            f"{x['median_wall_ms_per_prefix']:.3f}",f"{x['uncompressed_joblib_bytes']/1024:.1f}"]
            for x in r['common_sequential_inference_ledger']]
        pages.append([('h1','Appendix B. Frozen practical inference and storage'),
            ('table',(['Model','Dev accuracy %','Dev NLL','Sequential ms / prefix','Saved model KiB'],rows,[55,26,25,34,33])),
            ('p','All models consume the same saved observed first-second packet counts, one prefix '
             'at a time on one CPU thread. Fit-only feature transformation and native race simulation '
             'are timed. Three deterministic repeats measure execution variation; they are not '
             'independent fits. Warmup, loading and common raw event coalescing are reported separately.'),
            ('p','Every frozen probability matches the completed development result and every repeated '
             'prediction is identical. Native weights are the fixed minimum-NLL selected checkpoint; '
             'no model is refitted, temperature-adjusted or selected during this audit. All conventional '
             'cells remain visible, including faster or better alternatives.'),
            ('p','Storage is uncompressed joblib serialization of each fitted model plus its necessary '
             'fitting-only transform. It is not resident process memory or memory traffic. Native '
             'parameter tensor bytes and persistent state bytes are separate quantities. CPU '
             'emulator latency cannot be relabelled as event-hardware latency or measured energy.'),
            ('small','Subject-disjoint development evidence only. A quality/resource tradeoff here '
             'requires frozen independent confirmation before promotion. No test leakage, broad '
             'supremacy, useful-depth premium or dormant-capacity advantage is inferred from this audit.')])
    for r in tasks.get('dvs_frozen_features',[]):
        rows=[[x['encoder'],f"{100*x['original_development']['accuracy']:.2f}",
            f"{100*x['frozen_decoder_development']['accuracy']:.2f}",f"{x['frozen_decoder_development']['nll']:.4f}",
            x['fit_only_selected_decoder']['configuration']['kind'],
            f"{x['replay_wall_s']:.3f}",f"{x['decoder_grid_wall_s']:.3f}"] for x in r['rows']]
        pages.append([('h1','Appendix B. Real learned features versus the initial reservoir'),
            ('table',(['Frozen encoder','Original accuracy %','Fresh-head accuracy %','Fresh-head NLL','Head','Replay seconds','Head grid seconds'],rows,[27,27,27,22,18,26,26])),
            ('p','Both encoders expose the same32-dimensional query feature and use the same984 '
             'fitting/192 development gestures and race draws. Every original prediction is reconstructed '
             'exactly by its saved head; replay and new head fitting preserve every encoder parameter. '
             'The initial encoder is the exact pre-fitting reservoir, not a separate tuned control.'),
            ('p','Three linear C values and six RBF C/gamma cells are evaluated by3-fold fitting-only '
             'decoder NLL. The selected head is then fitted once on all fitting features. No head '
             'hyperparameter is selected on development. Scaling is fitting-fold only. Conditional '
             'decoder CV is not unbiased end-to-end validation because the selected encoder already '
             'saw all fitting labels during its original fit.'),
            ('p','The fitted representation gives a better selected readout than the initial reservoir: '
             '67.19% versus57.81%. This supports useful feature learning under this probe protocol. '
             'Replacing the fitted head increases accuracy only2.08points and worsens NLL relative '
             'to the original65.10%/.963161. The final decoder alone does not close the strong '
             'raw-input control gap; failed finite heads also do not prove information is absent.'),
            ('p','This is a frozen diagnostic, not an architectural substitution or practical advantage. '
             'The selected encoder still pays its original20.075GFLOPs/1557.590s workflow, plus full '
             'replay and every readout fit shown here. Solver arithmetic remains unmeasured. Useful '
             'extra depth, semantic language features, whole-route gradient accuracy and independent '
             'confirmation are separate questions. No official test was opened.'),
            ('small','Theory75 states admission and scope. Selected initial/fitted readout artifacts, '
             'all fitting-CV cells, native/checkpoint hashes, probability arrays, wall and RSS are saved.')])
    for r in tasks.get('dvs_noise_covariance',[]):
        rows=[]
        for x in r['models']:
            g=x['groups'];rows.append([str(x['seed']),*[f"{g[k]['common_over_independent_ratio']:.5f}" for k in
                ('all_parameters','route_maps','time_maps','message_maps')],
                'Pass' if x['admission_supports_independent_rows'] else 'FAIL'])
        pages.append([('h1','Appendix B. Shared fitting noise: covariance hypothesis fails its gate'),
            ('table',(['Saved seed','All variance ratio','Route ratio','Time ratio','Message ratio','Gate'],rows,[23,36,29,29,32,24])),
            ('p','Four fixed fitting prefixes per saved native seed6/7, 32 independent whole-history '
             'draws and zero optimizer updates. Each clip gradient is recorded under the same stream. '
             'Shared-batch trace covariance is measured; independent-coupling covariance is estimated '
             'from those same marginal samples. Their difference equals the cross-clip covariance sum.'),
            ('p','Both route-map ratios miss the declared 1.20 admission gate. Proposed fresh_shared/ '
             'fresh_independent fits are stopped. Neither model shows a large cross-clip covariance '
             'penalty on these prefixes. This is not a corpus-wide result or a test of repeated-noise '
             'adaptation across optimizer updates. Small-group variance ratios retain their finite-sample scope.'),
            ('p','Theory93 derives the effective-batch covariance law and separates per-window freshness '
             'from cross-row decorrelation. Independent noise need not help when cross-covariance is '
             'negative. Flipout is a primary-paper analogy about shared perturbation correlation, '
             'not an implementation or transferred quality guarantee for these event races.'),
            ('small',f"Two algebra contracts pass. Audit {r['wall_s']:.3f}s/{r['max_rss_kb']/1024:.1f}MiB; "
             f"{r['whole_audit_unit_special_flops_estimate']/1e9:.6f} known diagnostic GFLOPs est. "
             'Complete first-draw per-clip forward/backward traces times draws; reporting reductions separate. '
             'Frozen weights preserved. No training, official test or superiority claim.')])
    for r in tasks.get('dvs_persistent_state_probes',[]):
        quality=[];work=[]
        for x in r['rows']:
            for feature,key,dimension in [('Query','query_probe_development',32),('Query+state','augmented_probe_development',176)]:
                q=x[key];quality.append([x['encoder'],feature,str(dimension),f"{100*q['accuracy']:.2f}",f"{q['nll']:.4f}"])
            original=x['original_encoder_fit_gflops_estimate'];replay=x['replay_whole_gflops_estimate']
            work.append([x['encoder'],f'{original:.6f}',f'{original*1e3/7872:.6f}',
                f'{replay:.6f}',f'{replay*1e3/1176:.6f}','Unknown'])
        pages.append([('h1','Appendix B. Causal persistent-state access: positive probe, qualified cause'),
            ('table',(['Frozen encoder','Feature access','Dimensions','Dev accuracy %','Dev NLL'],quality,[30,46,26,36,35])),
            ('table',(['Encoder','Original fit GF est.','Fit MF / presentation','Core replay GF est.','Replay MF / prefix','Decoder/grid GF'],work,[23,31,31,31,31,26])),
            ('p','Same 984 fitting/192 development gestures. Augmentation adds all eight pre-query '
             'receiver memories, their ages at source query admission and occupancy flags to the actual '
             '32-dimensional query feature. No future or target-dependent feature. Three contracts '
             'reproduce serial state, original probabilities and saved query-probe probabilities while '
             'keeping every encoder parameter bitwise fixed. Nine-cell three-fold fitting-only head selection.'),
            ('p','Initial augmentation improves NLL by .244819; trained augmentation by .140774. Both '
             'pass the declared diagnostic access signal. The trained augmented probe reaches 71.35%/ '
             '.859084, versus original native 65.10%/.963161, compact prototype 66.67%/.902951 and '
             'strong full RBF 73.44%/.706478. Decoder/grid costs are unknown, so these are quality '
             'comparisons and diagnostic signals, not practical advantage.'),
            ('p','Important qualification from the completed partition controls: stronger query-only '
             'regularization alone reaches .889842; adding full state at that same C contributes '
             'only .030759 further NLL improvement. At the weaker C, full-state NLL worsens to '
             '1.278955. Most apparent trained-state gain is therefore compatible with regularization '
             'rather than uniquely missing information. Early/later state partitions retain smaller gains.'),
            ('small','Theory94/95. Dense all-state probing is a diagnostic, not sparse inference or a '
             'new main architecture. Selected encoder saw all fitting labels and was dev-selected; '
             'decoder CV is not unbiased end-to-end validation. Known original fit uses 7,872 '
             'presentations; current two-pass core replay uses 1,176 prefixes. Solver/materialization/ '
             'traffic/energy and total probe fit/inference remain unmeasured rather than zero.')])
    for r in tasks.get('dvs_state_partition_probes',[]):
        labels={'payloads':'Payloads','clocks':'Clocks','layer0':'Layer0','layer1':'Layer1',
            'query_with_state_setting':'Query@state setting','state_with_query_setting':'State@query setting'}
        rows=[]
        for x in r['rows']:
            q=x['development'];c=x.get('fixed_configuration',x.get('fit_only_selected_decoder',{}).get('configuration',{}))
            rows.append([x['encoder'],labels[x['partition']],str(x['feature_dimension']),f"{100*q['accuracy']:.2f}",
                f"{q['nll']:.4f}",f"{c['kind']}/{c['C']}"])
        pages.append([('h1','Appendix B. Retained payloads, clocks, layers and regularization'),
            ('figure',('report/figures/'+r['args']['tag']+'_learning.png',95)),
            ('table',(['Encoder','Added state/control','Dims','Accuracy %','NLL','Kind/C'],rows,[23,49,19,27,25,30])),
            ('p','Four partitions retain the query: both payload layers, ages/occupancy only, '
             'layer0 state or layer1 state. Same nine-cell fitting-only decoder selection per '
             'partition, both initial and trained encoders. Two fixed configuration swaps per '
             'encoder test regularization without another grid or development selection.'),
            ('p','Trained payload .860321, layer0 .838315 and layer1 .840329; clocks-only 1.017809. '
             'The query C.1 control .889842 explains most full-state gain. These finite probe '
             'differences do not isolate a causal depth failure or population conditional information. '
             'All outcomes remain visible; core producers/routes were never retrained.'),
            ('small','Theory95. Cached features add zero CORE replay, not zero fitting work. '
             'Prior encoder fit/core replay retained; additional solver arithmetic unknown. '
             'Conditional folds share the label-trained encoder. Producer-held fitting examples '
             'are the next decoder-selection check. No official test or supremacy claim.')])
    for clock in tasks.get('dvs_clock_completed',[]):
        comparison=next(r for r in tasks['dvs_practical_native']
                        if clock['args']['controls'] in r['inputs'])
        rows=[[x['arm'].replace('time_binned_naive_bayes','Calibrated counts').replace('native_temporal_p16_L2_H2_pool2_seed6','Ours original'),
               f"{100*x['development_accuracy']:.2f}",f"{x['development_nll']:.4f}",
               *['Unmeasured' if x[k] is None else f'{x[k]:.3f}' for k in
                 ('whole_fit_gflops_estimate','fit_mflops_per_presentation_estimate','inference_mflops_per_target_estimate')]]
              for x in comparison['common_unit_ledger']]
        w=clock['work']
        rows.append(['Ours packet-scale clock',f"{100*clock['final']['accuracy']:.2f}",f"{clock['final']['nll']:.4f}",
                     f"{w['whole_fit_unit_special_flops_estimate']/1e9:.3f}",
                     f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.3f}",
                     f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.3f}"])
        for compact in tasks['dvs_compact_controls']:
            q=compact['selected_compact_development_quality']
            rows.append(['Compact prototype33',f"{100*q['accuracy']:.2f}",f"{q['nll']:.4f}",
                         'Unmeasured','Unmeasured','Unmeasured'])
        pages.append([('h1','Appendix B. Completed packet-scale clock comparison'),
            ('table',(['Model','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[48,23,23,26,27,26])),
            ('p','Both native fits use984 gestures/eight fixed passes,7,872 presentations and496 updates; '
             '192 subject-disjoint dev targets select pass8 by minimum NLL. Only initial decay rates and '
             'rotation frequencies are scaled to the observed50ms packets. Temporal races, key/value '
             'separation, sparse updates and counterfactual learning remain. Parameters15,523; '
             'available receivers8; per-prefix168 key scores,84 selected updates,168 candidate values.'),
            ('p',f"Clock fit wall{clock['wall_s']:.3f}s; peak RSS{clock['max_rss_kb']/1024:.1f}MiB. "
             'Accuracy rises1.04points versus original while NLL worsens. Both trail the selected full '
             'kernel and compact prototype in both quality measures. This initialization change does '
             'not establish advantage; the original lower-NLL result and all control cells are retained.'),
            ('small','Same units and native target denominators in every work column. Solver FLOPs remain '
             'unmeasured; unequal solver policies and development tuning preclude iso-FLOP superiority. '
             'Raw preprocessing, validation, checkpoints, traffic and energy remain separate. One seed; '
             'no official-test access. Source: '+clock['args']['tag']+'.json; theory76.')])
    for r in tasks.get('dvs_batched_smoke_admission',[]):
        rows=[[x['credit'],f"{x['initial_fit_nll']:.4f}",f"{x['final_fit_nll']:.4f}",
               f"{x['whole_fit_gflops']:.6f}",f"{x['fit_mflops_per_target']:.4f}",
               f"{x['inference_mflops_per_prefix']:.4f}"] for x in r['rows']]
        resources=[[x['credit'],f"{x['wall_s']:.3f}",f"{x['max_rss_kb']/1024:.1f}",
                    str(x['window_size_counts']['16']),str(x['window_size_counts']['8'])] for x in r['rows']]
        pages.append([('h1','Appendix B. Full and partial-window batched learning admission'),
            ('table',(['Credit','Initial fit NLL','Selected fit NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[25,29,29,30,30,30])),
            ('table',(['Credit','Workflow seconds','Peak RSS MiB','U16 updates','U8 updates'],resources,[35,40,38,30,30])),
            ('p','Each arm uses24 fitting gestures/two fixed passes and8 development targets:48 target '
             'presentations/four Adam updates. Both select pass2 and score25% dev accuracy; dev NLL '
             '2.3115 local and2.3070 pairs. These tiny fixed smokes verify learning and resource readiness, '
             'not prediction advantage. All optimizer stages have complete operator coverage. '
             'Accounting includes candidate values, backward, normalization/clipping and Adam.'),
            ('p','Same p16/L2/H2/pool2 integrated architecture:15,523 parameters,8 available receivers, '
             '21 events per prefix,168 scored keys/84 selected state updates/168 candidate values, '
             '720 persistent-state tensor bytes. Independent-clip batching passes forward/state and '
             'every-parameter gradient equality plus actual interrupted model/Adam/cursor recovery.'),
            ('p','Optional pair credit enumerates the actual final-query two-head outcome losses. '
             'Its conditional risk and all derivatives match explicit enumeration; earlier routes '
             'retain local surrogate credit. Inference still delivers two hard winners per layer. '
             'No exact whole-core gradient or useful-depth claim follows. The bounded matched pilot '
             'is next; stronger full-data controls are reported above and are not comparable tiny-fit controls.'),
            ('small','One guarded one-thread job at a time; RSS watchdog and8GiB available-memory floor. '
             'Theory77; completed admission record local_dvs_batched_smoke_admission_20261002T163600Z.json. '
             'All negative full-fit results remain visible; no pending score fills an evidence table.')])
    for r in tasks.get('dvs_credit_comparisons',[]):
        rows=[[x['arm'].replace('_',' '),f"{100*x['development_accuracy']:.2f}",f"{x['development_nll']:.4f}",
            f"{x['whole_fit_gflops_estimate']:.6f}",f"{x['fit_mflops_per_presentation_estimate']:.4f}",
            f"{x['inference_mflops_per_target_estimate']:.4f}"] for x in r['common_unit_ledger']]
        resources=[[x['arm'].replace('_',' '),f"{x['wall_s']:.3f}",f"{x['max_rss_kb']/1024:.1f}",
            str(int(x['key_scores_per_fit_presentation'])),str(int(x['selected_updates_per_fit_presentation'])),
            str(int(x['counterfactual_values_per_fit_presentation']))] for x in r['common_unit_ledger']]
        pages.append([('h1','Appendix B. Completed matched credit pilot: '+r['args']['kind'].replace('_',' ')+f" / seed{r['args'].get('seed',6)}"),
            ('figure',('report/figures/'+r['args']['tag']+'_learning.png',173)),
            ('table',(['Credit','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[29,25,24,31,32,32])),
            ('table',(['Credit','Workflow seconds','Peak RSS MiB','Keys / fit target','Commits / fit target','Values / fit target'],resources,[29,29,26,29,30,30])),
            ('p',f"Fixed256 fit/192 development gestures, four passes/1,024 target presentations/64 Adam updates; "
             'same initialization, causal packets, draws and minimum-devNLL selection. Unchanged p16/L2/H2/pool2 '
             'architecture:15,523 parameters/eight available receivers; inference168 scored keys/84 commits/168 '
             'candidate values per21-event prefix. Fitting activity includes shadow replay where present.'),
            ('p',f"Promotion gate {'PASSES' if r['promotion_gate_passed'] else 'FAILS'}: NLL improvement "
             f"{r['nll_improvement']:.6f}, accuracy decline{r['accuracy_decline_percentage_points']:.3f}pp, "
             f"whole-fit work ratio{r['whole_fit_work_ratio']:.4f}. Required gain>=.02, decline<=1pp, "
             f"work ratio<={r['gate']['maximum_work_ratio']:.2f}, peak RSS<900,000KiB. "
             'A failed gate stops unchanged confirmation/scale-up; all passes and negative findings retained.'),
            ('small','All forward/replay, backward, normalization/clipping and Adam paid;2FLOPs/MAC plus unit '
             'specials. Strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 references have unequal fitting '
             'data; their solver FLOPs remain unmeasured. See common-unit full-fit tables above. One seed, '
             'no official test or superiority claim. '+r['args']['tag']+'.json; theory77/80–84.')])
    choice_repeats=[r for r in tasks.get('dvs_credit_comparisons',[]) if r['args']['kind']=='state_choice']
    if len(choice_repeats)>=2:
        rows=[[str(r['args']['seed']),f"{r['nll_improvement']:+.6f}",
            f"{-r['accuracy_decline_percentage_points']:+.4f}",f"{r['whole_fit_work_ratio']:.5f}",
            'Pass' if r['promotion_gate_passed'] else 'FAIL'] for r in choice_repeats]
        pages.append([('h1','Appendix B. Actual-write choice: independent-seed gate fails'),
            ('table',(['Seed','NLL improvement','Accuracy gain pp','Fit work ratio','Declared gate'],rows,[20,46,40,35,32])),
            ('p','Same256 fitting/192 development gestures, four fixed passes and1,024 presentations per arm/seed. '
             'Seed6 improves54.17%/1.305937 to58.85%/1.194006. Seed7 gives55.21%/1.306508 local versus '
             '55.21%/1.336201 corrected credit. Every pass and both completed comparisons are retained.'),
            ('p','The unchanged method fails its independent confirmation gate. No full984-fit campaign, '
             'extra pass extension or best-seed promotion is admitted. Numerical credit correctness remains '
             'established for the isolated categorical component; repeatable predictive benefit is not. '
             'Averaging seeds does not override the predeclared gate.'),
            ('p','Next distinguish information access, branch/content credit exposure, common-clock noise '
             'and private-map dilution on fitting-only frozen probes. Alternative no-grad forwards teach '
             'choices without directly teaching losing content on that realization. Winner-sampled branch '
             'derivatives can nevertheless be correct in expectation; rare exposure and other approximate '
             'teachers require separate tests. A larger pool is not an automatic repair.'),
            ('small','Theory87/88 prioritize one diagnosed repair, numerical contracts, a tiny integrated '
             'fit and a prespecified matched-work comparison before independent confirmation. Joint versus '
             'alternating route/message updates is a conditional schedule experiment, not a change to the '
             'coupled temporal forward computation or a currently demonstrated advantage.')])
    for audit in tasks.get('dvs_suffix_credit_audits',[]):
        rows=[[Path(x['native']).stem.replace('local_dvs_',''),str(x['opposed_teacher_directions'])+'/'+str(x['nonzero_comparable_policy_gradients']),
            f"{x['mean_absolute_value_only_effect']:.5f}",f"{x['mean_absolute_write_only_effect']:.5f}",
            f"{x['mean_absolute_value_write_interaction']:.5f}"] for x in audit['models']]
        pages.append([('h1','Appendix B. Why actual write and joint-clock credit need a test'),
            ('table',(['Frozen model','Opposed directions','Mean |value effect|','Mean |write effect|','Mean |interaction|'],rows,[61,26,29,29,28])),
            ('p','Replay32 fixed event/head sites on the first four previously used development clips/model. '
             'Both legal winner choices change delivered content and actual persistent writes, with full '
             'suffix replay at unchanged current first time and future random draws. Two additional '
             'value/write hybrids are diagnostic, not legal routes. Nonzero-comparable direction counts '
             'exclude tiny/zero gradients. Absolute effects are not additive attribution percentages.'),
            ('p','In the original full model, one alternative improves value-only loss by.01016 but worsens '
             'write-only loss by.10514; its legal complete-route effect is+.09193 NLL. The averaged local '
             'value teacher favors that harmful alternative. These four original-model clips are correctly '
             'classified: this is a conditional credit defect, not attribution of the overall error rate.'),
            ('p','Theory78 independently proves a convex affine cross-entropy example where the two-route '
             'local teacher reverses exact expected-loss descent. Actual reference/backward contracts pass. '
             'The full-state correction replaces one earlier score derivative by pi_i(F_i-b) minus '
             'lambda_i*T times the probability-weighted centered loss. It retains sampled content derivatives '
             'and includes common-clock credit through later timing jumps. The isolated-node joint likelihood '
             'component is exact in expectation; other route teachers remain local, and variance is unresolved.'),
            ('p','Routes, messages and time parameters do learn: frozen original/256-fit local/pair models '
             'change35.8%/26.2%/25.2% of16,128 audited choices versus initialization, with nonzero updates '
             'in every parameter group. Removing routing score-gradient paths leaves the factual forward '
             'unchanged but removes incoming-content/message gradients. This does not prove their fitted '
             'utility or diagnose the full quality gap.'),
            ('p','Learned-window gradients were tested as smooth primitives; fixed-count repeated-arrival '
             'fits are a separate mechanism. New silence-burst/popcorn contracts cover scheduled deadlines, '
             'causal EOF, fixed-partition gradients and exact finite timeout-bank risk. No integrated learned '
             'window or popcorn fit is established. Hard merge/split credit and an irregular adapter remain '
             'prerequisites; fixed50ms packet schedules do not test natural silence.'),
            ('small','Sources: completed frozen route-content audit171500Z, suffix audit172000Z, curvature '
             'contracts165900Z, silence-burst contracts170500Z and joint-clock contracts173000Z; '
             'theory78–81. Numerical diagnostics are not held-out superiority evidence.')])
    for audit in tasks.get('dvs_clock_variance_audits',[]):
        rows=[[('Local' if 'batched_local' in x['native'] else 'Joint-clock treatment'),
            f"{x['mean_choice_rms']:.6f}",f"{x['mean_common_clock_rms']:.6f}",
            f"{x['mean_common_clock_rms']/x['mean_choice_rms']:.0f}",
            f"{x['summed_prefix_baseline_variance']:.6f}"] for x in audit['models']]
        pages.append([('h1','Appendix B. Frozen variance audit after the failed clock pilot'),
            ('table',(['Selected checkpoint','Choice RMS','Common-clock RMS','Clock / choice RMS','Summed variance'],rows,[47,30,32,32,32])),
            ('p','First four previously used fitting prefixes/model, event9/both layers/heads. Replay both '
             'legal delivered values and actual persistent writes through the complete suffix at8/16 '
             'exponential-time quadrature nodes, with fixed future draws. Each model pays192 complete '
             'legal shadow forwards. No optimizer or new quality score. Magnitudes and summed variances '
             'are in score space with the same per-clip/probe convention for both rows.'),
            ('p','Decompose the conditional-winner joint score into choice credit pi_i(F_i-R) and '
             'common-clock credit pi_i(1-Lambda*T)(R-b). If branch losses are independent of T, the '
             'common-clock mean is zero and baseline mismatch alone produces variance '
             '||pi||^2(R-b)^2. Choosing b=R removes it exactly; the constant-outcome numerical '
             'contract passes. Time-dependent later route jumps can make the common-clock mean useful, '
             'so deleting it universally is not justified.'),
            ('p','The current intermediate decoder baseline differs substantially from actual suffix '
             'risk. Across these limited probes the common-clock RMS is900–1,300 times the choice '
             'RMS; an oracle baseline computed with all quadrature replays removes more than99.9999% '
             'of estimated variance. That oracle is a diagnostic ceiling with all replay cost paid, '
             'not a deployable cheap baseline, optimizer-noise measurement or fitted improvement.'),
            ('p','Eight-versus16-node mean-gradient L2 differences range4.3e-8 to7.3e-7. This agrees '
             'numerically on these probes but does not establish convergence across discontinuous '
             'histories. Per-pass shared draws also prevent assuming ordinary minibatch variance '
             'reduction; actual batch covariance was not measured. Neither audit proves the cause '
             'of the held-out regression or the primary bottleneck on other benchmarks.'),
            ('p','Decision: stop the failed joint-clock campaign. Next isolate exact conditional '
             'actual-write choice credit while retaining the native pathwise clock derivative and '
             'all temporal/sparse mechanisms. This retains a known approximation for downstream '
             'timing jumps; numerical contracts and one small smoke precede any matched fit. A '
             'later independent-noise/control-variate comparison needs fresh matched local controls '
             'and full recovery/work accounting. No automatic wider capacity or pass extension.'),
            ('small',f"Completed diagnostic{audit['args']['tag']}.json;{audit['wall_s']:.3f}s/"
             f"{audit['max_rss_kb']/1024:.1f}MiB peak RSS, one guarded CPU job. Theory82/83 and "
             'COUNTERFACTUAL_CREDIT_PLAN.md preserve proof, negative fit and limited diagnostic scope.')])
    for smoke in tasks.get('dvs_state_choice_smokes',[]):
        w=smoke['work'];act=smoke['activity'];n=act['targets']
        rows=[['Write-choice + native timing',f"{100*smoke['final']['accuracy']:.2f}",f"{smoke['final']['nll']:.4f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.6f}",f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"]]
        pages.append([('h1','Appendix B. Actual-write choice credit: integrated admission'),
            ('table',(['Credit','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[46,25,22,28,27,25])),
            ('p',f"Fixed24 fit/eight dev/two passes,48 presentations/four Adam updates: two U16 and two "
             f"partial U8 windows. Fitting NLL{smoke['initial_fit']['nll']:.4f} to"
             f"{smoke['final_fit_diagnostic']['nll']:.4f}; workflow{smoke['wall_s']:.3f}s, "
             f"peak RSS{smoke['max_rss_kb']/1024:.1f}MiB. This is readiness, not benchmark advantage."),
            ('p',f"Unchanged native p16/L2/H2/pool2:15,523 parameters/eight available receivers. "
             f"Fitting pays{act['key_scores']/n:.0f} key scores/{act['selected_updates']/n:.0f} state commits/"
             f"{act['counterfactual_values']/n:.0f} candidate values per target including the complete shadow. "
             'Inference retains168 keys/84 commits/168 values over21 events;720 persistent-state tensor bytes.'),
            ('p','At one event9 race head/window, substitute the exact conditional expected-loss derivative '
             'for both legal delivered values and actual persistent writes. Retain the separate native '
             'winner-delay derivative, winning content/state credit and all other local teachers. '
             'The obsolete intermediate decoder baseline is unnecessary for categorical enumeration.'),
            ('p','Contracts independently enumerate the choice risk, differentiate smooth raw clocks, '
             'preserve winning payload and other-head credit, reproduce factual state/logits and actual '
             'alternative commits, and verify every parameter-gradient change equals the upstream VJP '
             'of the replaced score residual. Interrupted Adam/cursor/RNG recovery and complete '
             'operator accounting pass. Downstream timing jumps and other local teachers remain approximate.'),
            ('small','Theory84; contracts191100Z and smoke191400Z. All losses summed before gradient '
             'normalization, global clipping and Adam; all shadow work charged. No official test or '
             'exact whole-model gradient claim. Fixed matched pilot and second-seed gate follow.')])
    for audit in tasks.get('dvs_credit_balance_audits',[]):
        rows=[]
        groups=[]
        for model in audit['models']:
            label='Full984' if 'native_full' in model['native'] else 'Pilot256'
            for x in model['credit_results']:
                rows.append([label+' / '+x['credit'].replace('state_',''),f"{x['raw_parameter_gradient_norm']:.4f}",
                    f"{x['native_global_clip_scale']:.4f}",f"{x['cosine_to_local_parameter_gradient']:.4f}",
                    f"{x['replacement_upstream_gradient_norm']:.4f}"])
                g=x['parameter_group_gradient_norms']
                groups.append([label+' / '+x['credit'].replace('state_',''),
                    *[f'{g[name]:.4f}' for name in ('incoming_context_maps','message_maps','route_maps','time_maps','decoder')]])
        pages.append([('h1','Appendix B. Calibration: noisy routing changes global gradient balance'),
            ('table',(['Frozen weights / credit','Total norm','Clip scale','Cosine to local','Replacement norm'],rows,[57,28,28,30,30])),
            ('table',(['Frozen weights / credit','Context norm','Message norm','Route norm','Time norm','Decoder norm'],groups,[52,26,25,24,23,23])),
            ('p','Same selected weights/first16 fitting prefixes/draw314159/event9 layer0 head0 '
             'for each local, actual-write choice+native timing and joint-clock intervention. Factual '
             'logits and losses are identical. All gradients summed then divided by16 before computing '
             'the native norm1 clip multiplier. No optimizer, fitting selection or new quality score.'),
            ('p','For Full984, the noisy joint-clock correction raises total gradient norm3.3901 to '
             '6.4002, rotates its direction to cosine.3940 and reduces the global clip multiplier '
             '.2950 to.1562. The decoder gradient is unchanged before clipping, so the global multiplier '
             'also attenuates its useful supervised update. This directly measures cross-path balance '
             'on one draw; it is not attribution of the held-out regression.'),
            ('p','Write-choice credit keeps total norm3.3915, cosine.99998 and clip multiplier.2949 '
             'while correcting actual legal memory utility. Positive scaling alone cannot repair a '
             'wrong direction or manufacture useful information. Per-sample norm equalization can '
             'bias a zero-mean signal; a prefix-only clock baseline or independent calibration requires '
             'a separate matched protocol. Shared weights retain both message and timing credit.'),
            ('small',f"Completed audit{audit['args']['tag']}.json;{audit['wall_s']:.3f}s/"
             f"{audit['max_rss_kb']/1024:.1f}MiB. Every intervention's forward/replay, backward and "
             'normalization operator coverage passes. No measured expected variance, batch covariance, '
             'Adam trajectory or fitted benefit from gradient normalization follows.')])
    if tasks.get('dvs_calibration_contracts'):
        contract=tasks['dvs_calibration_contracts'][-1]
        pages.append([('h1','Appendix B. Credit calibration: coordinates and limits'),
            ('p',f"Completed calibration contracts: {contract['contracts_passed']} passed. Common log-rate c "
             'and relative logits u define rates exp(c)*softmax(u). The joint winner/time score pulls back '
             'to common-clock credit(1-Lambda*T)*(R-b), and categorical credit pi_i*(F_i-R). The choice '
             'component is independent of common baseline error; the clock component remains useful and noisy.'),
            ('p','Recomposing c=logsumexp(old scores) and u=old scores preserves BOTH the old forward and '
             'the old gradient. A coordinate identity does not remove clock noise. Separately learned '
             'output maps require an architectural comparison; shared incoming content can still receive '
             'both derivatives. Eligible-key normalization and discovery remain paid.'),
            ('p','A positive score-space rescaling can oppose parameter descent through coupled Jacobians. '
             'The contract gives true parameter gradient[-1,-1] versus rescaled[8,-1], dot product-7. '
             'Per-sample unit normalization also biases an explicitly zero-mean scalar estimator. '
             'RMS equalization is therefore not a universal calibration rule.'),
            ('p','Primary-paper analogues suggest variance-trained legal control variates(RELAX), conditional '
             'averaging, phasic updates with constrained route drift(PPG), exposure accounting(MoE) and '
             'event derivative jumps(EventProp). These are distinct repairs. MoE common-logit z-loss '
             'would regularize this race clock itself; higher-order optionality needs correct stochastic '
             'derivatives(DiCE), not repeated differentiation of detached first-order teachers.'),
            ('small','Theory85 mathematical contracts and theory88 linked primary papers; no fitted benefit '
             'from these normalization/alternation proposals. Parameter-space covariance and branch exposure '
             'on independent-noise fitting probes determine the next single repair. Keep coupled forward '
             'messages/races/timing, sparse addressed writes and all existing negative evidence.')])
    for smoke in tasks.get('dvs_paired_choice_smokes',[]):
        w=smoke['work'];act=smoke['activity'];n=act['targets']
        rows=[]
        for label,x in [('Enumerated2 / readiness',tasks['dvs_state_choice_smokes'][-1]),
                        ('Paired8 / readiness',smoke)]:
            a=x['work']
            rows.append([label,f"{100*x['final']['accuracy']:.2f}",f"{x['final']['nll']:.4f}",
                f"{a['whole_fit_unit_special_flops_estimate']/1e9:.6f}",
                f"{a['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
                f"{a['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
        pages.append([('h1','Appendix B. Larger alternative pools: paired-credit readiness'),
            ('table',(['Credit / pool','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[42,25,22,28,28,28])),
            ('p',f"Both24 fit/eight dev/two passes/48 presentations/four Adam updates,U16+partial U8. "
             f"Paired8 fitting NLL{smoke['initial_fit']['nll']:.4f} to{smoke['final_fit_diagnostic']['nll']:.4f}; "
             f"{smoke['wall_s']:.3f}s/{smoke['max_rss_kb']/1024:.1f}MiB peak RSS. These tiny fits establish "
             'readiness, not quality rankings, larger-pool headroom or practical superiority.'),
            ('p',f"Paired8 has{smoke['parameters']:,} parameters/32 available receivers versus15,523/eight "
             'for pool2. Per21-event inference prefix:672 scored keys/84 selected commits/672 candidate '
             'values; persistent state2,376 bytes. Fitting includes the full shadow: '
             f"{act['key_scores']/n:.0f} keys/{act['selected_updates']/n:.0f} commits/{act['counterfactual_values']/n:.0f} "
             'values per target. More state does not mean free key discovery or training.'),
            ('p','One full alternative write/suffix forward per fitting window supplies loss-difference '
             'credit with recorded proposal propensity. Epsilon.1 gives importance multiplier at most '
             '1/.9; eight auxiliary proposal exponentials/target are recorded separately from arithmetic. '
             'Finite2/3/8/64-candidate contracts match independent expected-risk derivatives; pool2 nests '
             'enumerated credit exactly. Pool8 legal writes, every-parameter pool2 nesting and actual '
             'Adam/cursor/RNG recovery with full/partial windows pass; all operators are covered.'),
            ('small','Contracts195800Z, smoke201100Z; theory86/87. Inference and native timing remain '
             'unchanged at each shape. No-grad alternatives do not directly teach losing payload maps. '
             'Other local teachers and future timing jumps remain approximate. The failed initial195200Z '
             'operator-accounting attempt is preserved; covered equivalent subtraction passes a fresh run.')])
    for audit in tasks.get('dvs_branch_exposure_audits',[]):
        rows=[]
        for model in audit['models']:
            seed='6' if '_s6_' in model['native'] else '7';cells=model['cells']
            msg=[c['groups']['message_maps']['cosine'] for c in cells]
            rows.append([seed,str(model['choice_score_opposed'])+'/'+str(model['choice_score_comparable']),
                str(model['global_route_clock_opposed'])+'/'+str(len(cells)),
                f"{sum(msg)/len(msg):.4f}",f"{min(c['sampled_vs_enumerated_branch_cosine'] for c in cells):.6f}"])
        pages.append([('h1','Appendix B. Fitting-only branch exposure and parameter interference'),
            ('table',(['Saved seed','Opposed selected choice','Opposed global difference','Mean message cosine','Min sampled/averaged cosine'],rows,[25,37,39,35,37])),
            ('p','First two fitting prefixes of each saved native local seed6/7 model; four independent '
             'whole-history draws, events9/19, both layers, head0. Actual branch writes and full suffixes '
             'preserve current first time. Native gradients are compared with an explicit diagnostic '
             'blocking ALL race-score paths, including raw-clock score sensitivities. No optimizer or '
             'development-label selection; these prefixes do not represent all gestures.'),
            ('p','Route/clock-path differences mildly oppose message gradients in both saved models. '
             'Averaging both legal message branches leaves the full parameter-gradient direction almost '
             'unchanged on these prefixes. Negative inner product alone does not prove harmful interference '
             'or justify deleting a chain-rule term. Finite whole-history covariance estimates include '
             'changing entering states and clocks; they are not current-node conditional variance.'),
            ('p','Independent ordinary-autograd gather contracts reproduce all parameters and full state '
             'for six score-blocked factual/legal branches. Detached-probability weighting separates branch '
             'derivatives from categorical derivatives. All captured native/branch forwards, backwards and '
             'residual VJPs have complete operation coverage; reporting reductions are outside that ledger.'),
            ('small',f"Completed audit203600Z: {audit['wall_s']:.3f}s/{audit['max_rss_kb']/1024:.1f}MiB. "
             'Contracts203400Z. No causal explanation of seed7 failure, global exact-gradient claim or '
             'benchmark advantage. User-proposed phase offsets have a separate coupled-time hypothesis.')])
    for contract in tasks.get('race_teacher_expectation_contracts',[]):
        witness=contract['convex_quadratic_winner_dependent_error']
        rows=[['True hard-outcome risk','+.018750','-.018750'],
              ['Original teacher expectation','+.018750','-.018750'],
              ['Exact-pi teacher expectation','-.028125','+.028125']]
        pages.append([('h1','Appendix B. Exact probabilities do not remove winner-dependent credit bias'),
            ('table',(['Conditional gradient','Candidate0 score','Candidate1 score'],rows,[87,43,43])),
            ('p','Four completed contracts use actual backward implementations and independent autograd. '
             'A convex quadratic loss.5*(value-.6)^2 with values0/1 and race rates1/3 gives probabilities '
             '.25/.75 and legal losses.18/.08. The original local teacher happens to equal the true '
             'categorical derivative here; the exact-probability replacement reverses its direction.'),
            ('p','Replacing a random rate coefficient by known pi preserves the old expectation when '
             'downstream error is fixed. In a nonlinear model that error depends on the winner, its write '
             'and subsequent routes. The full gradient is therefore not generally unchanged-expectation '
             'or zero variance. In the quadratic witness, replacement optimizes loss at the MEAN value; '
             'hard-delivery expected loss also contains a value-variance derivative.'),
            ('p','Protocol correction beside theory59 section400: the earlier curie192000Z fidelity audit '
             'forces each candidate with its individual arrival time, changing identity AND timing. '
             'Conditional first time has mean.25 for both winners; unconditioned individual times have '
             'means1/.333333. Its original sign/magnitude numbers remain combined-intervention evidence, '
             'not an exact fixed-time route-credit diagnosis. Existing models/results are preserved.'),
            ('small','Theory89; contracts204000Z. Exact-pi remains an empirical candidate separately owned '
             'by the other host. Neither this counterexample nor the earlier diagnostic establishes '
             'general superiority of either local teacher. Correct counterfactuals match actual sparse '
             'delivery, legal writes, time law and downstream utility.')])
    if tasks.get('dvs_evolution_offset_smokes'):
        rows=[]
        for x in tasks['dvs_evolution_offset_smokes']:
            w=x['work'];rows.append([x['args']['update_schedule'],f"{100*x['final']['accuracy']:.2f}",
                f"{x['final']['nll']:.4f}",f"{w['whole_fit_unit_special_flops_estimate']/1e9:.6f}",
                f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
                f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
        pages.append([('h1','Appendix B. Evolution offset preserves signal-time coupling'),
            ('table',(['Schedule','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[28,27,24,33,33,28])),
            ('p','Phase=frequency*physical_age+beta, beta=pi*tanh(raw_offset). Physical-age damping, '
             'emission delays, receiver readiness and stored timestamps remain. The32 offsets calibrate '
             'reception phase; they do not create a separate signal clock. Time still drives representation '
             'evolution and receives timing derivatives. Initial zero reproduces the original model.'),
            ('p','Both24fit/eightdev/two passes/48 presentations/four updates,U16+partialU8; p16/L2/H2/'
             'pool2 has15,555 parameters/eight receivers,168 inference keys/84 commits/168 candidate '
             'values over21 events,720 persistent-state bytes. Joint fitting NLL2.64095 to2.09915;'
             'alternating to2.14262. Both smokes take21.0 seconds. Tiny development scores establish '
             'readiness, not quality ranking or advantage against strong controls.'),
            ('p','Alternating message/route windows train the offset only with message parameters. '
             'Inactive parameters and Adam momentum/steps remain fixed. Shared content/context maps '
             'receive the full derivative every window; their updates can still change routes. No '
             'content detach or changed credit estimator. Private blocks receive fewer updates and '
             'active-gradient clipping differs at the same total presentation count.'),
            ('small','Four contracts205000Z: zero-offset state/logit/original-gradient identity; nonzero '
             'serial/batched identity; physical-age/offset finite differences and independent directions '
             'with damping; phase ownership and actual interrupted recovery/accounting. Theory90. '
             'Pure rotation can make offset/time locally redundant; constant phase is a known calibration '
             'operator, not a universal capacity or convergence theorem.')])
    for r in tasks.get('dvs_evolution_offset_comparisons',[]):
        rows=[[x['arm'],f"{100*x['development_accuracy']:.2f}",f"{x['development_nll']:.4f}",
            f"{x['whole_fit_gflops_estimate']:.6f}",f"{x['fit_mflops_per_presentation_estimate']:.6f}",
            f"{x['inference_mflops_per_target_estimate']:.6f}"] for x in r['common_unit_ledger']]
        gates=[[g['schedule'],f"{g['nll_improvement']:+.6f}",f"{-g['accuracy_decline_percentage_points']:+.4f}",
            f"{g['whole_fit_work_ratio']:.6f}",'Pass' if g['promotion_gate_passed'] else 'FAIL'] for g in r['schedule_gates']]
        pages.append([('h1',f"Appendix B. Completed coupled-time offset pilot: seed{r['args']['seed']}"),
            ('figure',('report/figures/'+r['args']['tag']+'_learning.png',173)),
            ('table',(['Schedule','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[28,27,24,33,33,28])),
            ('table',(['Offset schedule','NLL improvement','Accuracy gain pp','Fit work ratio','Gate'],gates,[36,40,35,35,27])),
            ('p','Fixed256 fit/192 dev/four passes/1,024 presentations/64 Adam updates. Same native '
             'initial predictions, data, draws and minimum-devNLL selection. Native15,523 parameters '
             'versus15,555 with offsets; eight available receivers and identical selected activity. '
             'All offset operators, backward, clipping and active Adam work are charged.'),
            ('p',f"Candidate gate requires>=.02NLL improvement,<=1pp accuracy decline,<=1.50 fitting "
             f"work ratio and<900,000KiB RSS. Selected schedule: {r['selected_schedule'] or 'none'}. "
             'Seed7 joint confirmation fails:1.327347 versus native1.306508NLL and2.0833pp '
             'accuracy decline. No unchanged full-data stage or extra epochs is admitted. '
             'Two seed6 schedules were declared before either result; all failures remain.'),
            ('small','Preserved strong984-fit RBF73.44%/.7065 and compact66.67%/.9030 controls have '
             'unequal fitting data; solver FLOPs unmeasured. Alternation also changes private-block '
             'update counts/clipping, so this is an algorithm comparison, not isolated interference '
             'attribution. Physical time still drives content; no official test or supremacy claim.')])
    for contract in tasks.get('local_credit_geometry_contracts',[]):
        pages.append([('h1','Appendix B. When added representation freedom can help'),
            ('p','Theory91 derives a local, weighted score/content fitting criterion. With existing '
             'Jacobian A, added offset Jacobian B and desired change b, first fit existing coordinates; '
             'let r be the remaining residual. Eliminating those coordinates gives offset sensitivity '
             'h=B-transpose*r and positive definite Schur matrix S. The exact best local improvement '
             'is one half h-transpose*S-inverse*h. It is strict precisely when h is nonzero.'),
            ('p','Additional coordinates can address a missing useful direction, or lower a movement '
             'penalty along an already available direction. The second case improves conditioning '
             'without proving increased representational rank. Shared offsets act at many receptions, '
             'whose demands may cancel. A local post-race offset preserves that already selected race '
             'while changing future state and routes; the full sequence remains coupled.'),
            ('p','For a damped rotating carrier z, the age/phase Jacobian determinant is minus damping '
             'rate times squared message norm. The raw bounded offset adds pi*sech(raw-offset)^2. '
             'Nonzero damping gives independent local age and phase directions; weak damping, '
             'decayed messages or saturated offsets can still make them poorly conditioned. '
             'Physical-time derivatives and actual timestamps remain intact.'),
            ('p','Four completed independent matrix contracts check 16 direct-versus-eliminated solves, '
             'zero-projection/no-benefit, redundant coordinates with lower penalized movement cost, '
             'and the rotor determinant. These are mathematical identities, not a demonstrated '
             'learning cure. Exact gradients of one loss can have opposed route/content terms and '
             'still yield descent when added; opposition alone does not justify deleting credit.'),
            ('small','Contracts211000Z; theory91. Joint offset seed6 gains .047767 NLL at 1.000910 '
             'fitting-work ratio; unchanged seed7 loses .020839 and 2.0833 accuracy points. '
             'No full-data stage is admitted. The local criterion does not guarantee validation '
             'generalization, convergence of surrogate race credit or practical superiority.')])
    for contract in tasks.get('race_replay_law_contracts',[]):
        pages.append([('h1','Appendix B. Counterfactual replay must match the first-time law'),
            ('table',(['Time-only loss, rates 1/3','First score gradient','Second score gradient'],[
                ['True expected loss gradient','-.062500','-.187500'],
                ['Incorrect replay choice addition','+.125000','-.125000'],
                ['Factual pathwise + wrong replay','+.062500','-.312500'],
                ['Correct joint-score expectation','-.062500','-.187500']],[87,43,43])),
            ('p','The first arrival conditioned on either winner has Exp(sum rates) law. Forcing a '
             'losing alternative at its individual Exp(its rate) arrival changes both identity and '
             'clock. Detached softmax weighting does not turn those combined interventions into '
             'exact conditional choice credit. The table uses loss equal to raw arrival time; '
             'the first score direction reverses under the incorrect combined estimator.'),
            ('p','Four completed contracts include independent analytic expectations and the actual '
             'external local-expectation replay wrapper with one time-recording event at four seeds. '
             'Forcing the factual winner reproduces the factual delay, while every losing alternative '
             'uses its later individual arrival. Existing softmax-algebra and factual-winner tests '
             'cannot establish the expected-risk claim. External model, queue and empirical results '
             'are preserved; this is a credit-protocol correction, not a fitted quality result.'),
            ('p','Correct conditional choice replays retain the factual first time and actual selected '
             'memory write. Clock credit needs its own consistent estimator. Factorized winner/first '
             'time sampling or the joint likelihood can supply coherent derivations, with all future '
             'discrete credit and deterministic branch derivatives included. Do not double-count '
             'one sampled time through both likelihood and reparameterized derivatives.'),
            ('p','A critic residual correction is unbiased for a declared replay sum when the critic '
             'is fixed before its correction subset is drawn. Training on that same subset first '
             'can introduce bias: a two-site counterexample gives -1 for a true zero target. '
             'Even a sampling-correct critic preserves the base target; it cannot cure wrong replay '
             'time laws. Independent conditioning and critic work require separate contracts.'),
            ('small','Theory92; contracts212000Z; qualifications beside theory59 sections402/403. '
             'Four contracts pass; audited external source versions are recorded. No integrated '
             'repair fit, whole-model exactness, guaranteed variance reduction or superiority claim.')])
    for audit in tasks.get('dvs_prefix_replay_savings',[]):
        rows=[[x['variant'],f"{100*x['accuracy']:.2f}",f"{x['nll']:.4f}",
            f"{x['whole_fit_gflops']:.6f}",f"{x['fit_mflops_per_target']:.6f}",
            f"{x['inference_mflops_per_target']:.6f}"] for x in audit['rows']]
        pages.append([('h1','Appendix B. Exact prefix reuse reduces integrated fitting work'),
            ('table',(['Replay','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[26,28,25,33,33,28])),
            ('p',f"Detached causal-prefix reuse saves{100*(1-audit['fit_work_ratio']):.3f}% counted fitting "
             'work with identical entire learning curves, predictions, model weights, Adam, cursor and '
             'RNG recovery. Same24-fit/eight-dev/two-pass/48-target native p16/L2/H2/pool2 protocol. '
             'Both rows use the same full-fit and presentation denominators.'),
            ('p','Reuse skips repeated alternative-prefix computation while preserving the exact '
             'actual-write estimator. This is completed implementation advantage against its full-replay '
             'reference, not quality advantage against the strong gesture controls. Snapshot/copy memory '
             'traffic and energy remain unmeasured; all retained suffix and optimizer work is charged.'),
            ('small','Other-host evidence retained from AWS_PREFIX_REPLAY_FINDINGS_20261002.md and '
             'aws_prefix_replay_smokes_20261002T201100Z_analysis.json, with verified parent hashes. '
             'Observed13.609 versus12.653 seconds is one concurrent-host observation, not a general '
             'latency claim. Exact reuse may support future credit comparisons after matching contracts.')])
    if tasks.get('dvs_external_capacity_completed'):
        reference=tasks['dvs_practical_native'][0];clock=tasks['dvs_clock_completed'][0]
        selected_references={'time_binned_naive_bayes',reference['selected_control'],
                             'native_temporal_p16_L2_H2_pool2_seed6'}
        rows=[[x['arm'].replace('time_binned_naive_bayes','Calibrated counts').replace('native_temporal_p16_L2_H2_pool2_seed6','Ours original'),
            f"{100*x['development_accuracy']:.2f}",f"{x['development_nll']:.4f}",
            *['Unmeasured' if x[k] is None else f'{x[k]:.3f}' for k in
            ('whole_fit_gflops_estimate','fit_mflops_per_presentation_estimate','inference_mflops_per_target_estimate')]]
            for x in reference['common_unit_ledger'] if x['arm'] in selected_references]
        for compact in tasks['dvs_compact_controls']:
            q=compact['selected_compact_development_quality']
            rows.append(['Compact prototype33',f"{100*q['accuracy']:.2f}",f"{q['nll']:.4f}",
                'Unmeasured','Unmeasured','Unmeasured'])
        cells=[clock]+tasks['dvs_external_capacity_completed']
        capacity=[]
        for x in cells:
            w=x['work'];a=x['args'];label=f"Native p{a['payload']}/L{a['depth']}/pool{a['pool']}"
            rows.append([label,f"{100*x['final']['accuracy']:.2f}",f"{x['final']['nll']:.4f}",
                f"{w['whole_fit_unit_special_flops_estimate']/1e9:.3f}",f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.3f}",
                f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.3f}"])
            capacity.append([label,str(x['parameters']),str(w['native_available_receivers']),
                str(int(x['activity']['key_scores']/x['activity']['targets'])),
                str(int(x['activity']['selected_updates']/x['activity']['targets'])),
                str(x['final']['max_state_tensor_bytes'])])
        pages.append([('h1','Appendix B. Other-host completed native capacity comparisons'),
            ('table',(['Model','Dev accuracy %','Dev NLL','Whole fit GFLOPs est.','Fit MFLOPs / target est.','Infer MFLOPs / prefix est.'],rows,[49,23,22,27,26,26])),
            ('table',(['Native capacity','Parameters','Available receivers','Keys / prefix','Commits / prefix','State bytes'],capacity,[47,27,28,24,25,22])),
            ('p','These completed stages use the same984-fit/192-dev, eight fixed passes/7,872 '
             'presentations/496 updates and seed6 as the packet-clock reference. Wider eligible '
             'pool and extra depth retain native timing/races/persistent addressed state/key-value '
             'separation/counterfactual credit. All scored keys and losing values remain charged.'),
            ('p','Pool8 scores62.50%/1.0880NLL versus pool2 reference66.15%/1.0420, at57.2985 '
             'versus20.0747 whole fitting GFLOPs. Depth4 scores57.29%/1.2216 at37.8686GFLOPs. '
             'Neither completed variant improves the baseline. Extra private capacity also adds '
             'untied trainable maps; this is not isolated counterfactual-support or useful-capacity '
             'evidence. Other-host tied-map/replication work remains pending and separate.'),
            ('small','Completed curie_dvs_clock_p16d2pool8 and p16d4pool2 results172500Z; '
             'theory59§§396/398. Conventional solver work is unmeasured, not zero. Raw controls '
             'retained in the full comparison above; selected references here use the same column '
             'units. No official-test or supremacy claim.')])
    for r in tasks.get('dvs_compact_controls',[]):
        groups={}
        for x in r['rows']:
            key=(x['arm'].split('_m')[0],x['components'])
            if key not in groups or x['final']['nll']<groups[key]['final']['nll']:groups[key]=x
        rows=[[x['arm'],f"{100*x['final']['accuracy']:.2f}",f"{x['final']['nll']:.4f}",
            f"{x['uncompressed_joblib_bytes']/1024:.1f}",str(x['within_native_storage_budget']),
            f"{x['sequential_ms_per_prefix']:.3f}"] for x in groups.values()]
        selected=next(x for x in r['rows'] if x['arm']==r['selected_within_budget_by_dev_nll'])
        pages.append([('h1','Appendix B. Strong compact controls rule out an easy storage claim'),
            ('table',(['Lowest-NLL cell / family size','Dev accuracy %','Dev NLL','Model KiB','In budget','CPU ms / prefix'],rows,[59,24,23,24,21,22])),
            ('p',f"Native uncompressed saved-model budget{r['native_storage_budget_bytes']/1024:.1f}KiB. "
             'The fixed72-cell grid uses random8/16/24/32 Nyström landmarks or1/2/3/4 learned '
             'prototypes per class, three gamma scales and three logistic C values. Prototypes, '
             'landmarks and normalization use fitting data only; each model includes its transform '
             'in the same uncompressed joblib serialization. Every cell remains in the completed JSON.'),
            ('p',f"Selected in-budget control {r['selected_within_budget_by_dev_nll']}: "
             f"{100*selected['final']['accuracy']:.2f}%/{selected['final']['nll']:.4f}NLL, "
             f"{selected['uncompressed_joblib_bytes']/1024:.1f}KiB, "
             f"{selected['sequential_ms_per_prefix']:.3f}ms/prefix. This exceeds the original native "
             '65.10%/.963161 while using less storage and far less CPU time. Thus the original '
             'native result does not establish advantage even in this bounded-storage region.'),
            ('p',f"The whole compact grid costs{r['wall_s']:.3f}s/{r['max_rss_kb']/1024:.1f}MiB peak RSS; "
             f"common raw preprocessing still costs{r['preprocessing_wall_s']:.3f}s. Shared prototype "
             'construction, kernel features, all fitting/tuning and scoring are paid in workflow wall. '
             'Solver FLOPs are unknown, not zero. The table summarizes each family/size by its '
             'minimum development NLL, with all gamma/C settings and probabilities retained.'),
            ('small','Exploratory development calibration, not independent model confirmation. '
             'Single timing passes are not repeated latency claims. Class prototypes are supervised '
             'fitting controls, not target-derived inference inputs. Neither the full kernel nor '
             'these compact alternatives is omitted when evaluating the next native initialization.')])
    composed=[r for path in sorted((RES/'count_composed_carrier').glob('*Z.json'))
              if (r:=read(str(path.relative_to(RES)))).get('status')=='completed' and 'final' in r
              and r['args']['fit']>=2048 and r['args']['dev']==8192]
    if composed:
        carriers={}
        for path in [*(RES/'parallel_language').glob('curie_carrier_selective_w*_D131072_*Z.json'),
                     RES/'parallel_language/local_selective_w128_D131072_selective_20260930T161050Z.json',
                     RES/'parallel_language/local_staged_language_capacity_w256_D131072_20260930T163234Z.json']:
            if path.exists() and (r:=read(str(path.relative_to(RES)))).get('status','completed')=='completed':
                carriers.setdefault(r['args']['width'],r)
        def cwork(r):
            w=r['work']
            return [f"{w['total_training_unit_special_flops']/1e9:.1f}",
                    f"{w['total_training_unit_special_flops']/w['fitting_targets']/1e6:.3f}",
                    f"{(w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'])/1e6:.4f}"]
        crow=[]
        shown=set()
        for r in sorted(composed,key=lambda r:(r['args']['width'],bool(r['args'].get('escape_gate')),bool(r['args'].get('count_message')))):
            W,N=r['args']['width'],r['args']['fit']
            if W in carriers and W not in shown:
                shown.add(W)
                c=carriers[W]
                crow.append([f"Carrier alone w{W}",f"{N:,}/{c['args']['epochs']}",f"{c['final']['dev']['bpc']:.3f}",*cwork(c)])
            g,m=r['args'].get('escape_gate'),r['args'].get('count_message')
            tag=' + gate + message' if g and m else ' + escape gate' if g else ' + count message' if m else ''
            label=f"Minimal base w{W}/d{r['args']['depth']}" if W<8 else f"Carrier w{W}"
            crow.append([f"{label} + counts K{r['args']['orders']}{tag}",f"{N:,}/{r['args']['epochs']}",f"{r['final']['dev']['bpc']:.3f}",*cwork(r)])
            if not tag:
                crow.append([f"Same, untrained base w{W}",f"{N:,}/0",f"{r['initial_dev']['bpc']:.3f}",'Not trained','Not trained',cwork(r)[2]])
        refs=[row for path in sorted((RES/'count_reference').glob('*language_*.json'))
              for row in json.loads(path.read_text())['rows'] if row['fit']==131072]
        for label,row in (('KN counts, frozen',min((x for x in refs if x['method']=='kn' and not x['adaptive']),key=lambda x:x['bpc'],default=None)),
                          ('Counts, stream-adaptive',min((x for x in refs if x['adaptive']),key=lambda x:x['bpc'],default=None))):
            if row:
                if row.get('method')=='kn_interpolated':
                    label='Calibration ceiling: adaptive interpolated KN'
                crow.append([f"{label} o{row['order']}","131,072/1",f"{row['bpc']:.3f}",'Not FLOPs','Not FLOPs','Not FLOPs'])
        pages.append([
            ('h1','Appendix B (continued). Diagnostic: count receivers over the temporal carrier'),
            ('p','Labelled diagnostic, not the integrated native architecture. The input-gated temporal carrier '
             'supplies the base predictive to the same escape-race count cascade (Theory §§386–388). It tests '
             'whether sufficient-statistic receivers remove the memorization tax: if counts hold the exact '
             'local statistics, a small learned base should lose far less than the carrier alone does.'),
            ('table',(['Model','Fit chars / passes','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Infer MFLOPs / char ↓'],
             crow,[44,27,20,27,28,27])),
            ('small','Same 8,191 development targets; seed 6, one seed per row; same depth, chunk, learning rate and '
             'passes per width. Predeclared: P1 composed w128 < 2.326; P2 composed w32−w256 gap < half the carrier '
             'gap; both hold formally for the scalar cascade, but the trained bases alone score 8.17 (w32) / 11.34 (w128) bpc, worse than uniform: the '
             'standalone base is trained as a conditional residual, so this alone cannot establish an inert base (Theory §389.1). '
             'With the escape gate (+ count message) every width reaches 2.12–2.14, and a minimal 2-wide, one-layer base matches '
             'w128 (2.124 vs 2.129) at 1/50 of the work: the gain is learned count smoothing, not the temporal carrier. '
             'Count references are near-optimal estimators at this size. The rows above exceed the frozen-KN and Witten–Bell '
             'references, and the stronger stream-adaptive interpolated Kneser–Ney is the calibration ceiling here (§393). '
             'This is a mechanism diagnostic in a regime where counting is expected to be near-optimal for any learner, '
             'not an architecture verdict (§394). '
             'Count increments/lookups (5 per target) are integer table work outside FLOPs. Exploratory '
             'development evidence; no comparable-quality Transformer claim.')])
    for begin in range(0,len(tasks.get('native_language',[])),4):
        rows=tasks['native_language'][begin:begin+4]
        pages.append([
            ('h1','Appendix B (continued). Ours: native temporal-core language'),
            ('p','27-character text8 content enters one conversation address in the native event core. '
             'Independent receiver heads, persistent state, learned delays, evolving channels and '
             'losing-route credit remain; no historical KV attention or dense carrier is added. '
             'The receiver maps now learn across all token marks.'),
            ('figure',('native_language_quality_work',145)),
            ('table',(['Ours','Fit chars / passes','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Infer MFLOPs / char ↓'],[
                [f"d{r['args']['payload']}/p{r['args']['pool']}/s{r['args']['seed']}",
                 f"{r['args']['fit']:,}/{r['args']['epochs']}",f"{r['final']['dev']['bpc']:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['cpu_emulator']['total_training_unit_special_flops']/r['work']['fitting_targets']/1e6:.3f}",
                 f"{(r['work']['cpu_emulator']['inference_arithmetic_flops_per_character']+r['work']['cpu_emulator']['inference_special_functions_per_character'])/1e6:.4f}"] for r in rows], [31,32,23,29,30,29])),
            ('small','Fixed passes and frozen 8,191-target development selection; official test untouched. '
             'Existing stronger receiver and dense-reference evidence is preserved in the common ledger. '
             'Architecture, capacity and weight sharing differ from the KV variants. Work uses audited '
             'representative optimizer windows including backward, teachers and Adam; special functions '
             'have unit weight here, with separate JSON counts. These are exploratory scaling points, '
             'not a frontier score or a measured energy advantage. With H2/depth8/pool2, '
             'available receivers are 32, selected commits 16/token, scored keys and training proposals 32/token. '
             'Parameter counts and occupied state are retained in completed records.')])
    for begin in range(0,len(tasks.get('native_event',[])),4):
        rows=tasks['native_event'][begin:begin+4]
        pages.append([
            ('h1','Appendix B (continued). Ours: native event/state capability'),
            ('p','Each source writes three signed marks then receives an explicit causal query. '
             'Fits use 512 queries / 2,048 input events per pass, eight passes; development has 256 queries. '
             'All source populations are occupied. The order task predicts the last two signs; '
             'the timing task predicts a two-mode temporal trace. All events, timestamps and query '
             'flags are observed; target labels never address a module.'),
            ('figure',('native_event_quality_work',145)),
            ('table',(['Ours: task / sources / seed','Credit / time','Dev accuracy ↑','Whole fit GFLOPs ↓','Fit MFLOPs / query ↓','Infer MFLOPs / event ↓'],[
                [f"{r['args']['task']}/S{r['args']['sources']}/s{r['args']['seed']}",
                 ('CF' if r['args']['credit']=='counterfactual' else 'pathwise')+'/'+r['args']['time_input'],
                 f"{r['final']['dev']['accuracy']:.1%}",f"{r['work']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['total_training_unit_special_flops']/r['work']['fitting_query_targets']/1e6:.3f}",
                 f"{(r['work']['inference_arithmetic_flops_per_event']+r['work']['inference_special_functions_per_event'])/1e6:.4f}"] for r in rows], [40,34,25,28,24,23])),
            ('small','Equal declared query/event budgets; available receivers scale with source population '
             'while per-event key scores and selected commits are logged explicitly. All fitting arithmetic '
             'is summed from executed operators, including complete producer graphs, losing proposals, '
             'clipping and Adam; CPU numerical clocks and unit-weight specials are included. '
             'Bootstrap intervals by population and raw state-clearing/time/long-gap probes remain in '
             'the completed JSON. More addresses change training exposure. Synthetic capability and '
             'bounded activity do not establish superiority over timestamp-aware recurrent/attention '
             'controls, real-stream generalization or clockless energy savings.')])
    for begin in range(0,len(tasks.get('native_event',[])),6):
        rows=tasks['native_event'][begin:begin+6]
        pages.append([
            ('h1','Appendix B (continued). Ours: occupied state and activity'),
            ('p','Every source stores three content events and receives a query. These are useful-state '
             'tests rather than padded unused capacity. Event budgets, active depth, head count and '
             'candidate pool are fixed. More addresses reduce observations per local parameter; '
             'addressing is input information that strong controls must also receive.'),
            ('table',(['Ours: task / sources / seed','Available / occupied receivers','Commits / scores per event','Persistent state KiB','Parameters'],[
                [f"{r['args']['task']}/S{r['args']['sources']}/s{r['args']['seed']} "
                 +('CF' if r['args']['credit']=='counterfactual' else 'PW')+'/'+r['args']['time_input'],
                 f"{r['work']['available_receivers']}/{r['work']['inference_storage']['occupied_receivers']}",
                 f"{r['work']['selected_updates_per_event']}/{r['work']['key_scores_per_event']}",
                 f"{r['work']['inference_storage']['persistent_tensor_bytes']/1024:.2f}",f"{r['parameters']:,}"] for r in rows], [49,35,35,27,28])),
            ('table',(['Ours: task / sources / seed','Whole fit GFLOPs','Fit MFLOPs / input event','Infer MFLOPs / query'],[
                [f"{r['args']['task']}/S{r['args']['sources']}/s{r['args']['seed']} "
                 +('CF' if r['args']['credit']=='counterfactual' else 'PW')+'/'+r['args']['time_input'],
                 f"{r['work']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{r['work']['total_training_unit_special_flops']/r['work']['fitting_events']/1e6:.3f}",
                 f"{(r['work']['inference_arithmetic_flops_per_query']+r['work']['inference_special_functions_per_query'])/1e6:.3f}"] for r in rows], [57,36,43,38])),
            ('table',(['Ours: task / sources / seed','Selected dev accuracy','Clear-history accuracy','Independent confirmation'],[
                [f"{r['args']['task']}/S{r['args']['sources']}/s{r['args']['seed']} "
                 +('CF' if r['args']['credit']=='counterfactual' else 'PW')+'/'+r['args']['time_input'],
                 f"{r['final']['dev']['accuracy']:.1%}",f"{r['final']['cleared_state']['accuracy']:.1%}",
                 f"{r['final']['confirmation']['accuracy']:.1%}" if 'confirmation' in r['final'] else 'Not read'] for r in rows], [57,36,38,43])),
            ('small','Occupied receivers and stored float bytes come from the audited inference population, '
             'not parameter storage, Python metadata, graphs or host RSS. Inference/query includes all four '
             'producer/query events. Full fitting includes every event and optimizer operation. '
             'Arithmetic plus unit-weight specials; traffic/RNG/energy separate. Development selects '
             'checkpoints; independent confirmation is read once only on fixed promoted seed protocols. '
             'Population bootstrap intervals in the quality plot condition on the selected model and '
             'do not correct development selection or substitute for independent seed uncertainty.')])
    if tasks['historical_write_contracts']:
        pages.append([
            ('h1','Appendix B (continued). Ours: compact delayed learning'),
            ('p','Available historical content is not automatically learned historical content. '
             'After graph detachment, a retrieved old key/value can affect the prediction while '
             'its write map receives no later loss credit. Increasing the ordinary credit span '
             'from 16 to 64 did not improve the completed matched 2K pilot.'),
            ('figure',('historical_write_credit_flow',160)),
            ('p','The tested integrated construction saves the normalized feature φ of each historical '
             'write. Later queries send a compact producer teacher to sealed key/value maps while '
             'preserving independent heads, temporal races, incoming content and persistent addressed state.'),
            ('table',(['Ours: added teacher','Factorization','Extra work / query'],[
                ['Old keys','α q (Σ eᵢ φᵢ)ᵀ','O(C_old d + d²)'],
                ['Old winning value','α a φ_wᵀ','O(d²)'],
                ['Saved eligibility','One detached d-vector / write','50% extra K/V float storage']], [37,82,55])),
            ('p','A shared offset of all race scores leaves winner probabilities unchanged but '
             'rescales arrival time. Content contrasts can teach which value wins; a separate '
             'clock signal can teach when it arrives and how persistent state evolves. '
             'The new write teacher retains both signals.'),
            ('p','The shared query factors all admitted old key teachers into one matrix update. '
             'Live writes keep ordinary gradients without duplicate credit. Six read-only numerical '
             'checks and guarded full eight-block update/recovery contracts passed; disabling '
             'the new teacher exactly reproduces parent outputs, RNG, gradients and two Adam windows.'),
            ('small','This is exact for a hypothetical common perturbation of historical write maps '
             'with saved features fixed. Transport to current maps is a delayed local surrogate: '
             'old weight versions and omitted representation paths remain limitations. Additional '
             'learning arithmetic and eligibility traffic are real costs. Existing counterfactual '
             'score credit is retained. Fitting work uses representative audited windows, with exact '
             'credit-coverage counters logged separately. Matched quality/work results require completed fits; '
             'improved language accuracy and native clockless learning are not established by these contracts.')])
    for figure_name,rows in historical_write_groups(tasks):
        a=rows[-1]['args']
        pages.append([
            ('h1','Appendix B (continued). Ours: historical write credit'),
            ('p',f"H{a['heads']}, d{a['payload']}/head, {a['depth']} event blocks, pool{a['pool']}; "
             f"{a['fit']:,} fitting characters / {a['epochs']} passes; {a['dev']-1:,} frozen development targets, seed{a['seed']}. "
             f"Credit{a['chunk']}, Adam U{a['update_targets']}, lr{a['lr']:g}, warmup{a['warmup_targets']}. "
             'Forward races/history are preserved; sealed writes gain a compact local producer teacher.'),
            ('figure',(figure_name,140)),
            ('table',(['Ours: credit','Dev bpc ↓','Whole fit GFLOPs ↓','Fit MFLOPs / target ↓','Inference MFLOPs / char ↓'],[
                ['parent' if 'write_credit' not in v['args'] else f"α={v['args']['write_credit']:g}",
                 f"{v['final']['dev']['bpc']:.3f}",f"{v['work']['cpu_emulator']['total_training_unit_special_flops']/1e9:.3f}",
                 f"{v['work']['cpu_emulator']['total_training_unit_special_flops']/v['work']['fitting_targets']/1e6:.3f}",
                 f"{(v['work']['cpu_emulator']['inference_arithmetic_flops_per_character']+v['work']['cpu_emulator']['inference_special_functions_per_character'])/1e6:.4f}"] for v in rows],
                [28,26,35,40,45])),
            ('p','A historical normalized write feature φ is retained alongside each cached key/value. '
             'The old key-map teacher factors as α q (Σ eᵢ φᵢ)ᵀ: one weighted feature sum and one '
             'matrix outer product per query. The old winning value map receives α a φᵀ. '
             'Live writes keep ordinary credit without duplication; no full-history graph is reopened.'),
            ('small','One extra feature vector raises K/V float storage by 50%. The update is exact for '
             'a common hypothetical perturbation of historical maps at fixed saved features; transporting '
             'it to current maps is a delayed local surrogate with stale weight versions and omitted '
             'representation paths. Additional teacher/optimizer arithmetic is counted; feature traffic '
             'and physical energy are separate. Parent reuse requires exact forward/RNG/gradient/Adam '
             'nesting at α=0. One-seed exploratory development selection, not comparable-quality supremacy.')])
    if tasks['head_diagnosis']:
        diagnosis=tasks['head_diagnosis'][-1]
        pages.append([
            ('h1','Appendix B (continued). Ours: frozen information-flow diagnosis'),
            ('p',f"Saved selected H2/H4 eight-block 8K checkpoints; {diagnosis['args']['n']-1} targets from the development prefix. "
             'Weights remain fixed. Each intervention removes one path only during this short evaluation; '
             'these are diagnostic probes rather than refitted architecture comparisons.'),
            ('table',(['Ours','Frozen intervention','Window bpc ↓','Context RMS mean'],[
                [f"H{row['heads']}",v['mode'].replace('_',' '),f"{v['bpc']:.3f}",f"{v['context_rms_mean']:.3f}"]
                for row in diagnosis['rows'] for v in row['variants']], [24,70,36,44])),
            ('p','Removing source-message carry and learned channel mixing worsens both saved checkpoints in this window. These content/state paths are useful here; short-window removal probes do not warrant discarding them or demonstrate a refitted architecture advantage.'),
            ('p','A route-credit audit replays each admitted historical value at three depths/head0 '
             'for the two-head model. It holds one realized race time and continuation seed fixed, '
             'then compares the centered local content teacher with the conditional categorical loss gradient.'),
            ('table',(['Ours: local audit','Probes','Mean cosine','Opposed directions','Mean oracle gap (nats)'],[
                [f"H{row['heads']} / head0",str(row['route_credit']['summary']['probes']),
                 f"{row['route_credit']['summary']['mean_cosine']:.3f}" if row['route_credit']['summary']['mean_cosine'] is not None else 'undefined',
                 f"{row['route_credit']['summary']['negative_cosines']} / {row['route_credit']['summary']['defined_cosines']}",
                 f"{row['route_credit']['summary']['mean_oracle_gap_nats']:.3f}"] for row in diagnosis['rows'] if row['route_credit']], [42,22,30,40,40])),
            ('small','Higher cosine means better alignment in this narrow replay. The oracle gap is realized '
             'loss minus the best candidate replay, not achieved improvement. Different routing can change '
             'subsequent candidate/RNG paths. The audit conditions away time derivatives and does not '
             'estimate the full expected gradient, long-history utility or discovery coverage. '
             'Interventions have no refitting and no confidence intervals; do not use these window scores '
             'as promotion metrics or dense-model superiority evidence.')])
    if tasks['integrated_online_language']:
        row=tasks['integrated_online_language'][-1];a=row['args']
        pages.append([
            ('h1','Appendix B (continued). Ours: separate online neural learning'),
            ('p','Both arms start from the same selected integrated-model checkpoint and maintain '
             'persistent event memory on the same new development stream. The frozen arm retains '
             'its parameters. The online arm updates the complete neural backbone after making '
             f"the causal predictions in each {a['chunk']}-character block, with no replay."),
            ('figure',('integrated_online_language',174)),
            ('table',(['Ours: mode','Stream bpc ↓','Whole stream MFLOPs ↓','Parameter change L2','Updates'],[
                [f"Ours: {r['arm']}",f"{r['bpc']:.3f}",f"{row['work'][r['arm']]['unit_special_flops']/1e6:,.3f}",
                 f"{r['parameter_change_l2']:.3f}",str(r['updates'])] for r in row['rows']],
                [37,26,44,40,27])),
            ('p',f"The {a['n']-1:,} targets come from [{a['offset']:,}, {a['offset']+a['n']:,}). "
             f"Learning rate {a['lr']:g} was fixed before this stream; Adam starts with fresh moments. "
             'Both arms use paired block race noise. Parameter updates precede only future blocks; '
             'the current target cannot change its own prediction. Their contexts can diverge after learning.'),
            ('small','This measures block-delayed online adaptation, not instant per-character updates '
             'or a frozen official-test score. One inherited model/window/rate. All neural parameters '
             'may learn, including content, keys, clocks and retention; this differs from the earlier '
             'statistical expert-mixing-only ablation. Inherited fitting is additional and identical '
             'in both arms. Work estimates observe the actual first/middle/last blocks and include '
             'counterfactual learning, backward, clipping and Adam; RNG, indexing and physical '
             'traffic remain additional. Lower online loss, if observed, is evidence only for this protocol.')])
    language_rows=[
        ["Ours: separate statistical count/copy baseline",f"{ev['native10']:.3f}","10M count fitting + three 1M mixing-rate trials",compact_work(lm_costs['native_without_word']['total_training_flops'])+" + integer count construction"],
        ["Ours: count/copy plus causal word context",f"{ev['native_word10']:.3f}","10M count fitting + three 1M mixing-rate trials",compact_work(lm_costs['native_with_causal_word']['total_training_flops'])+" + integer count construction"],
    ]
    pages.append([
        ("h1","Appendix B (continued). Separate statistical language baseline"),
        ("p","This count/copy predictor does not use the learned Sleeping Machines event backbone. "
         "It is a separate statistical system: order-0 through order-6 counts, backoff probabilities and "
         "a bounded causal copy cache, combined by learned mixing weights. Its quality/work results "
         "must not be attributed to the event architecture."),
        ("p","The statistical predictor and saved neural references score the same 999,999 character targets, "
         "starting from a cold context. Parameters are frozen during testing. Earlier observed test characters "
         "can supply causal context, including the mixture's bounded 256-character copy cache. Lower bits per "
         "character means better prediction."),
        ("table",(["Predictor","Test bpc ↓","Fitting and selection budget","Estimated training FLOPs ↓"],language_rows,[51,21,59,43])),
        ("small","The neural totals charge all original optimizer steps that produced the inherited E174 "
         "checkpoints: forward, estimated 2×-forward backward, clipping and Adam. Alignment evaluation is excluded. "
         "The statistical floating estimate charges expert probability preparation and all three mixing-rate trials, "
         "including local gradients and weight updates. Exp/log/root evaluations count as one operation in these "
         "language estimates. Count construction additionally uses about 80M integer count presentations, "
         "plus sorting, lookup and hashing; that work is not quantified as FLOPs. "
         "The floating totals alone cannot establish total compute, runtime or energy savings."),
        ("p",f"The count/copy mixture improves on LSTM by {ev['lstm10']-ev['native10']:.3f} bpc "
         f"and Transformer by {ev['tf10']-ev['native10']:.3f} bpc. These results establish useful specialized prediction; "
         "generic learned representations are assessed in the separate language screen."),
        ("p","The statistical baseline's count arrays occupy 66.55 MB. "
         "Vocabulary, capacities, optimization and fitting budgets differ from the neural references. "
         "The comparison does not measure total training energy or a matched-capacity advantage."),
        ("p","All three use the same historical 27-character alphabet. Modern shared subword tokenization "
         "is a separate comparison gate for the learned event architecture, described later in this appendix."),
        ("small","One exploratory seed. Text8 offsets: count fitting [0,10M), mixing-weight validation "
         "[90M,91M), test [95M,96M); test index zero is excluded for all three predictors. The mixture "
         "selects its update rate on validation. Saved neural weights are unchanged. "
         "Results: E173/E174; stream contract: E175. "+language_90m_reference_text(ev))])

    generic = tasks["generic_language_audit"]["rows"]
    pages.append([
        ("h1","Appendix B (continued). Learned language and depth"),
        ("p","A bounded screen trains the common event backbone to predict the next character. "
         "Both configurations use width 32, the same 8,192 training characters, "
         "four passes, 32-character contexts and 1,024 validation predictions. All eight layers' value, route "
         "and memory-time parameters update. Lower bits per character means better prediction."),
        ("figure",("e133_generic_language",174)),
        ("table",(["Ours: depth","Learned parameters","Validation bpc: lower is better","Total CPU wall time"],[
         [str(depth),f"{generic[str(depth)]['parameters']:,}",f"{generic[str(depth)]['final_dev_bpc']:.3f}",f"{generic[str(depth)]['wall_s']:.1f} s"]
         for depth in (1,8)],[24,40,60,50])),
        ("p","Eight layers improve validation loss from 4.752 to 3.395 bpc, versus 3.464 with one layer. "
         "Shuffling preceding characters while preserving the last character, count and timestamps increases "
         "the deeper model's loss to 3.805; replacing preceding context raises it to 3.777. These frozen input "
         "probes show context sensitivity, not a retrained baseline comparison."),
        ("p","The deeper model has more parameters and takes more CPU time. The quality/time panel includes "
         "fitting and evaluation, with backpropagation and optimizer updates executed during fitting. These "
         "bounded-query models replay preceding context. The following persistent implementation consumes "
         "each character once. Physical memory traffic and joules remain unmeasured."),
        ("small","One seed; different parameter counts. This establishes a generic learned-language foothold "
         "and a small depth gain, not competitive large-scale representation, a matched tuned dense-model "
         "advantage or a scaling law. The statistical 10M-character mixture remains a separate result. Official test "
         "data are untouched. E133 preserves commands, source/data hashes, layer diagnostics and work coverage.")])

    pages.append([
        ("h1","Appendix B (continued). Persistent learned language"),
        ("p","The event-state language model consumes each character once and retains local modal "
         "memories and its delayed-message queue. Chunk boundaries "
         "truncate learning credit without discarding the observed history. Only actual event arrivals evaluate "
         "layers; text time is measured in token intervals."),
        ("figure",("e176_stream_language_learning",174)),
        ("table",(["Ours: representation","Parameters","Fitting bpc ↓","Validation bpc ↓","Deliveries/pass"],[
            ["Characters",f"{tasks['stream_training']['parameters']:,}",
             f"{tasks['stream_training']['final']['fit']['bpc']:.3f}",
             f"{tasks['stream_training']['final']['dev']['bpc']:.3f}",
             f"{tasks['stream_training']['final']['training_event_deliveries']:,}"],
            ["Causal prefix tokens",f"{tasks['token_training']['parameters']:,}",
             f"{tasks['token_training']['final']['fit']['bpc']:.3f}",
             f"{tasks['token_training']['final']['dev']['bpc']:.3f}",
             f"{tasks['token_training']['final']['training_layer_deliveries']:,}"],
        ],[46,31,30,31,36])),
        ("p",f"Validation loss falls from {tasks['stream_training']['initial']['bpc']:.3f} to "
         f"{tasks['stream_training']['final']['dev']['bpc']:.3f} bpc. All eight layer teachers are nonzero in "
         "every fitting pass. Each pass consumes 8,223 characters including warmup and makes 65,784 block "
         "deliveries. The result establishes learning with persistent causal state and no prefix replay."),
        ("p","A train-only 131-token prefix dictionary reduces layer deliveries by 30.4% and observed CPU time "
         "by 26.2%, while validation bpc is 3.517. Exact partial-token marginalization scores identical raw "
         "targets. Its larger vocabulary fits better but generalizes less well in this small screen: compression "
         "alone does not explain or resolve the quality gap."),
        ("small",f"One seed; 28,403 parameters, width 32, sixteen temporal modes per block. Four passes, "
         "128 Adam steps/pass, credit truncated every 64 characters, 31 warm characters and 1,024 validation "
         "targets. Target offsets match the bounded E133 screen; topology, capacity, history and update counts "
         "differ, so this is not a matched intervention. Total CPU wall time "
         f"{tasks['stream_training']['wall_s']:.1f} s including fitting/evaluation; peak RSS "
         f"{tasks['stream_training']['max_rss_kb']/1024:.1f} MiB. These are event counts and observed resources, "
         "not total arithmetic, physical memory traffic or energy. No official test or large-corpus claim. E176.")])

    event_work=tasks["event_language_work"]
    pages.append([
        ("h1","Appendix B (continued). Learned event language: training and inference work"),
        ("p","These counts belong to the learned eight-layer persistent Sleeping Machines event model "
         "that reaches 3.351 validation bpc. It has no count/copy/word experts. Primary counts describe "
         "the logical event algorithm; simulator dispatch/allocation is excluded."),
        ("table",(["Ours: fitting stage","Estimated arithmetic FLOPs"],[
          [stage.replace('_',' ').capitalize(),compact_work(value)]
          for stage,value in event_work['training_stages'].items()
        ]+[["Total",compact_work(event_work['total_training_arithmetic_flops'])]],[95,79])),
        ("p",f"The entire completed budget includes 32,768 fitting targets, 512 Adam/clipping steps and "
         f"four stream warmups. Special functions add {event_work['training_special_function_evaluations']/1e6:.3f}M "
         "evaluations, reported separately from arithmetic FLOPs."),
        ("table",(["Ours: inference boundary","Per character"],[
          ["Prediction plus NLL-scoring arithmetic",f"{event_work['inference_arithmetic_flops_per_character']:,.0f} FLOPs"],
          ["Additional special functions",f"{event_work['inference_special_functions_per_character']:,.0f} evaluations"],
        ],[112,62])),
        ("p","A 64-character saved-checkpoint trace measures forward/loss, backward, clipping and warm Adam; "
         "the fitting ledger scales those stages by the original 512 steps and separately charges warmup. "
         "This is a representative arithmetic estimate, not a whole-run trace. The traced chunk has complete "
         "floating-operator formula coverage. Index/queue work, comparisons, memory traffic and evaluation "
         "passes are outside the fitting arithmetic boundary."),
        ("p","The 10M-character LSTM/Transformer checkpoints have different data, capacity and achieved "
         "quality. Dividing their full fitting budgets by this small run would not establish a fair training "
         "advantage. Full learned-event runs must complete before a larger aligned quality/work comparison."),
        ("small","Evidence: event_language_work/local_event_language_work_20260930T124646Z.json; "
         "quality: E176. Two arithmetic FLOPs per multiply-add. Additional unit-weight special functions "
         "would give a different logical-operation total; no event-device runtime or energy has been measured.")])

    online=tasks["online_language"]
    pages.append([
        ("h1","Appendix B (continued). Statistical online-learning pilot"),
        ("p","The official language comparison freezes parameters during evaluation. A new development-only "
         "pilot asks whether causal local learning improves prediction: score each character first, reveal it, "
         "then update only the expert mixing weights. Count experts stay frozen and both arms use identical "
         "causal copy-cache behavior."),
        ("table",(["Ours: statistical expert set","Frozen bpc ↓","Online bpc ↓","Extra update FLOPs"],[
          ["Without word" if r['arm']=='without_word' else "With causal word",
           f"{r['frozen_bpc']:.3f}",f"{r['online_bpc']:.3f}",compact_work(r['extra_update_arithmetic_flops'])]
          for r in online['rows']],[56,37,37,44])),
        ("p","This pilot reuses a 100,000-character count checkpoint and its learning rate selected on an "
         "earlier 2,048-character validation window. It scores 8,191 targets in a fresh 8,192-character "
         "development window [90,032,768,90,040,960), excluding the first target. Each adaptive arm performs "
         "8,192 updates, including first-position warmup. Official test data are untouched."),
        ("small","One checkpoint and one window; a specialized adaptive readout, not deep learned TTT or a "
         "frontier result. The extra arithmetic column counts only local gradient/weight updates, on top of "
         "shared prediction and inherited fitting costs. Total pilot CPU wall time is 0.587 s. "
         "Evidence: online_language/local_online_language_20260930T121741Z.json."),
        ("h2","Compute allocation is the next architectural hypothesis"),
        ("p","Independent budgets for active width/depth, dormant capacity, temporal memory, retrieval, "
         "credit and adaptation may let extra work buy more prediction quality. The new theory derives "
         "conditional marginal-value allocation and retrieval-error bounds, and identifies exposure, "
         "routing overhead and hardware utilization as possible limits. A small-scale win does not prove "
         "a better scaling exponent or a widening frontier advantage."),
        ("p","Next learned-language comparisons need shared modern subword tokenization, Unicode/byte "
         "coverage, suitable rotary/relative position, packed optimized Transformer kernels, sparse MoE "
         "and modern recurrent/attention-hybrid controls. Existing event memory already uses relative-time "
         "rotations. Token coordinates must remain distinct from learned scheduling delays."),
        ("p","Known sequences can be fitted with input-known affine scans and then generated sequentially "
         "with persistent state. Sequence-parallel fitting also permits Transformer adaptation between "
         "tokens/chunks. Compare frozen and online arms at identical observations and adaptation budgets; "
         "charge cache consistency and update work."),
        ("small",'<a href="experiments/theory/43_compute_allocation_and_frontier_scaling.md">Theory §§280–286</a> '
         'and the <a href="experiments/FRONTIER_COMPUTE_PROTOCOL.md">frontier compute protocol</a> '
         "specify these tests. Modern architecture and larger-scale comparison arms remain proposed, not completed.")])

    historical_market=[]
    for path,label in (("e57/regime_m5.json","Ours: event hazard + rate/flow state"),
                       ("e57/regime_m5_pt.json","Ours: event hazard + per-type state"),
                       ("e57/regime_m5_fine_pt.json","Ours: event hazard + per-type state; finer gap bank")):
        old=max(json.loads((RES/path).read_text()),key=lambda r:r['val_day5'])
        historical_market.append([label,f"{old['test_day6']:.3f}",f"{old['test_day7']:.3f}"])
    old_tf=read("e52/thp_test_d64_L32_f0_e11.json")['epochs'][-1]
    historical_market.append(["Saved Transformer Hawkes reference",f"{old_tf['day6']:.3f}",f"{old_tf['day7']:.3f}"])
    pages.append([
        ("h1","Appendix C. Preserved historical results and revisions"),
        ("p","Earlier result files remain part of the research record. The following numbers explain "
         "older report headlines and why their interpretation changed. They are preserved here with the "
         "identified protocol errors; they are not current valid benchmark comparisons."),
        ("h2","Earlier statistical language results"),
        ("p","The target-dependent partial-word mixtures have been removed from the active results tree and "
         "all numerical comparisons. Raw records are quarantined for audit only. The corrected E17310M "
         "results are1.727 without word context and1.719 with causal word context; "
         "a corrected90M mixture comparison remains open."),
        ("h2","Earlier event world-model results"),
        ("table",(["Historical predictor","Day 6 log-likelihood ↑","Day 7 log-likelihood ↑"],
                  historical_market,[94,40,40])),
        ("p","The event hazard models use sparse conditional memories and local rate/flow state. Their "
         "frozen test parameters and causal state updates are useful mechanisms. However, the event-size "
         "threshold was fitted across all seven pilot days, including the evaluation days, for both the "
         "event and neural references. Day resets and warmup exclusions also differ. The recorded gap "
         "needs fitting-only preprocessing and aligned rescoring before it supports a held-day advantage."),
        ("small",'<a href="experiments/EXPERIMENTAL_REVIEW.md">Source and numerical review</a>; '
         '<a href="experiments/results/e57/">E57 world-model records</a>. '
         "Preserved timing, composition, retrieval and modular results remain in the main report. "
         "New learned-model benchmarks add evidence; they do not erase these earlier runs.")])

    pages.append([
        ("h1","Appendix D. Evidence and metric definitions"),
        ("table",(["Metric","Interpretation"],[
         ["Bits per character","Held-out negative log probability in base two; lower is better next-character prediction."],
         ["Accuracy","Fraction of correct class decisions on the declared development or test protocol."],
         ["Event likelihood","Scores both the next event type and waiting time, including the observed silence."],
         ["Arithmetic FLOPs","Multiply-add counts as two operations. Tables state whether special functions are separate or assigned unit cost; these conventions must be aligned before forming ratios."],
         ["Logical operations","The structured-task ledger assigns 2 units per MAC and 1 per other scalar arithmetic/nonlinear operation or estimated sort comparison."],
         ["Resource boundary","Event-target arithmetic includes required active algorithm work. Memory traffic, queues, indexing and simulator overhead are reported separately where measured. Activity counts alone do not determine total FLOPs."],
         ["Energy","Measured total joules over an explicit boundary. Operation estimates and CPU timings support work comparisons, but are not joule measurements."],
        ],[45,129])),
        ("p","The evidence is preserved in versioned result summaries with configurations, split identities, "
         "learning curves and source hashes. E173/E174 support the language comparison; E61 supports retrieval; "
         "E34/E53/E54 support native composition; E41 supports the original periodic computation. E121/E124 "
         "establish consolidated arithmetic and its certificate; E123 supplies the new dense controls and E124 "
         "the operation ledger. E118/E119/E122/E125/E126 support deep speech, readout and causal-context comparisons; "
         "E127–E131 audit credit geometry, hard race boundaries and separate key/value learning; E132 checks "
         "a joint race-credit formalism, E133 supplies the language/depth screen, and E134–E135 test "
         "whole-value credit and content-selective temporal memory. E136 audits reversible augmented transport "
         "and its supervised memory boundary, including twelve-layer query/learning interventions. E137 tests "
         "compact memory queries and class-visible credit geometry. E138–E141 examine richer source messages "
         "and trainable signed temporal memory, with exact local teacher and initial-nesting contracts. "
         "E142 establishes signed-state and first-coalescing identities; E143 tests a larger nonlinear temporal "
         "residual learner, and E144 audits simultaneous state/query pooling. E171 reproduces the consolidated "
         "screens and selected speech answers, and checks causal input boundaries. E172 records complete "
         "training-step arithmetic; E175 checks the generic persistent language stream."),
        ("p","The project theory index contains formal assumptions and proofs. Research findings retain detailed "
         "analyses and the full experimental record. The model documentation describes reproducible configurations "
         "and operational procedures. This report presents the project, its evidence and its potential.")])
    generality_page=[
        ('h1','A general architecture for content, time and selective activity'),
        ('p','Language tokens, irregular observations and action requests can be expressed as '
         'content-bearing events with timestamps and source identities. Sleeping Machines aim to '
         'learn through this common interface: local state evolves between arrivals, delays perform '
         'computation, and only recruited modules act. This broader design is the central research target.'),
        ('figure',('general_temporal_interface',174)),
        ('h1','Why this could matter across domains'),
        ('bullets',[
         '<b>Ordering as computation.</b> Earlier events change the state/routes encountered by later ones; elapsed time changes that state. The structured order-learning results support this prior for sequence-sensitive signals.',
         '<b>Asynchronous sensing.</b> Updates can follow observations and required deadlines rather than a periodic sweep of all modules. Silence remains informative when the objective depends on waiting time.',
         '<b>Instruction-conditioned control.</b> Language can guide event routing and memory; observations can ground language and update a world state that informs timed actions.',
         '<b>Useful dormant capacity.</b> Stored modules need not all execute for each input. The gain depends on economical discovery and credit, and is judged at a fixed total work budget.',
         '<b>Distributed hardware.</b> Local event-triggered state and communication can reduce global coordination. Globally clockless ASICs are a target; conventional FPGA prototypes retain clocks. Hardware joule savings remain to be measured.'
        ]),
        ('h1','What is established, and what is next'),
        ('table',(['Ours: family evidence','Completed result / scope','Next generality test'],[
         ['Temporal reasoning','99.73–99.93% event-order accuracy; five runs, declared structured task','Unseen delays, gaps and concurrent streams'],
         ['Deep learned context','3.121 development bpc; sparse six-block / 32K fit; eight-block models also train','Matched-quality work and capacity scaling'],
         ['Auditory events','79.69% on 512 private development utterances; selected temporal encoder','Aligned official-test real-stream comparison'],
        ],[45,77,52])),
        ('small','Joint multimodal learning and robot reliability remain research targets. Existing results use '
         'separately trained variants; native language uses 27 character pools. Sparse routing motivates tabular '
         'prediction: paired delays can represent feature thresholds (theory §323). Preserve feature IDs and '
         'avoid invented row order. Trees and tabular Transformers remain controls. The banknote screen leads '
         'the original trees; broader superiority requires the stronger-control confirmation.'),
    ]
    investment_page=[
        ('h1','The research upside: five routes to useful advantage'),
        ('p','The investment thesis is a trainable substrate with a known useful workload and broader '
         'capability beyond it. Reproducing relevant Transformer quality and convergence with lower '
         'whole-system energy would already be valuable. Temporal expressivity, selective capacity '
         'and cross-modal integration offer additional, independently testable upside.'),
        ('table',(['Strength','Potential benefit','Evidence needed'],[
         ['Temporal races and local state','Less digital normalization and value aggregation; a path to globally clockless hardware','Matched-quality full learning/inference, precision, throughput and measured system joules'],
         ['Evolving state and reused matches','More useful transformations per expensive match; potentially smaller models','Width/depth/data sweeps at fixed quality and complete work'],
         ['Capacity beyond activity','More specialized stored representations without executing all modules per observation','Improve quality as capacity grows; keep discovery, teaching and execution budgets economical'],
         ['Common content/time interface','Language-guided sensing and event-grounded reasoning/control in a shared model','Joint held-out modality combinations, task success and causal deadline tests'],
         ['Native learning during use','Adapt delays/routes/content locally on an event ASIC; reduce external trainer traffic','CPU online: 3.191→3.096 bpc; asynchronous on-chip credit/updates remain untested'],
        ],[43,64,67])),
        ('h1','Quantitative scenarios, with conditions'),
        ('bullets',[
         '<b>Compression.</b> At equal quality, half the width would give one-quarter projection arithmetic and roughly half the matching/state work. This compression is unproven.',
         '<b>Energy scenario.</b> Assume baseline shares of 40% compute, 40% memory, 10% clock, 10% fixed. Halving compute/memory energy, removing the clock and adding 5% control gives 45% savings (1.82×). These are assumptions, not a chip forecast.',
         '<b>Dormant units.</b> H2 stores 864 receivers and selects 16 updates/character (54× capacity/activity). Teaching evaluates 32 receiver alternatives and admitted historical values; shared maps execute. This is not a 54× resource saving.'
        ]),
        ('h1','Why this is a research program worth testing'),
        ('p','Multi-run structured learning/generalization, temporal algebra and deep trainable event representations '
         'provide starting evidence. The current language fits have not established a matched-quality '
         'resource advantage. Optimizer, width/head/data scaling and repeatability tests address that gap. '
         'Theory §320 proposes repeated Poisson arrivals that reuse matched keys without resetting all losing clocks.'),
        ('small','Current autograd, global clipping and block-window Adam do not demonstrate fully asynchronous learning. '
         'That needs dependency/version-aware credit and tested updates. On-chip learning has precedents '
         '(<a href="https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf">Intel Loihi 2</a>); the proposed contribution is the complete temporal/sparse-credit construction. '
         'Compare competent synchronous learning ASICs and charge gradient/optimizer traffic. '
         'See HARDWARE_VALUE_PROPOSITION.md and EVENT_STREAM_ADVANTAGE_PROTOCOL.md.'),
    ]
    architectural_pages=[generality_page,investment_page,[
        ('h1','The hypothesis: more capability per unit of active work'),
        ('p','Sleeping Machines combine trainable delays, temporal races, evolving local state and '
         'counterfactual credit. The hypothesis is that these mechanisms can approximate useful attention '
         'with less selected arithmetic and value movement, then use richer temporal computation and '
         'dormant capacity to reach comparable quality with smaller models or less fitting. '
         'A common content-and-time event interface can support tokens and irregular sensor streams, '
         'with task-specific adapters and losses. Existing cross-task models train separately; '
         'the integration target is language-guided event routing and shared state: '
         'events ground language and both inform actions. Shared-weight multimodal learning '
         'remains a further milestone.'),
        ('p','A softmax race samples exactly from its distribution. One winner does not equal its weighted '
         'average. Averaging m independent winners has mean-square error variance/m; approximate '
         'Transformer containment also requires historical coverage and stable propagation through depth. '
         'Temporal state permits additional computations beyond this attention analogue.'),
        ('h1','Matched attention work: retain all query/key matches'),
        ('p','Let d be total width, L depth, N historical keys, r the feed-forward expansion and '
         'B = (8 + 4r)d² the shared projection/content work per layer. '
         'S is extra evolving-state work per layer. Two FLOPs per multiply-add:'),
        ('table',(['Per token / target','Transformer','Ours: race substitution'],[
         ['Inference','L(B + 4Nd)','L(B + 2Nd + S)'],
         ['Training, approximate','3LB + 12LNd + 19P/U','3L(B + S) + 10LNd + 20P/U'],
         ['Logical value reads (FP32)','4LNd bytes','4Ld bytes'],
         ['Logical key + value reads','8LNd bytes','4L(N + 1)d bytes'],
        ],[48,60,66])),
        ('p','P is updated parameter count and U targets per Adam update. The race training term '
         'includes admitted losing-value credit; it is not winner-only training. Our normalization of '
         'accumulated gradients adds one operation per updated parameter. Shared embeddings/output '
         'and lower-order operations are added in the plotted scenario.'),
        ('h1','What would make the case decisive?'),
        ('bullets',[
         '<b>Matched-quality efficiency.</b> Repeated completed comparisons of full fitting work and inference work.',
         '<b>Capacity beyond activity.</b> More useful stored modules with nearly fixed routing and execution budgets.',
         '<b>Temporal expressivity.</b> Reuse expensive matches for distinct cheap races and evolving-state responses; test whether this reduces required width or depth.',
         '<b>Common event interface.</b> Tokens, irregular sensors and instruction-conditioned control can use content-and-time events. Real-stream, joint-reasoning/control and hardware-energy advantages require their own benchmarks.'
        ]),
        ('small','This is a research hypothesis and an architectural comparison, not a frontier-language or measured-energy claim. '
         'Current deep sparse learning and structured-task results establish meaningful mechanisms; compression and broad language advantage need further evidence.'),
    ],[
        ('h1','Expected architectural work and access scaling'),
        ('figure',('full_bank_temporal_scaling',174)),
        ('p','Scenario: d = 256, four heads, r = 4, U = 128 and S = 128d per layer. '
         'Context plots use L = 8; the depth plot scores N = 4,096 keys. '
         'All keys are scored in both models. Both retain linear context and depth terms, '
         'and quadratic width terms. At fixed width, the attention-only arithmetic limit is about 2× at inference '
         'and 1.2× during counterfactual training; common projection work lowers these total-work ratios. '
         'If richer temporal computation reaches the same quality at width αd and depth βL, '
         'projection work scales by βα² and context work by βα. Those additional savings require '
         'matched-quality evidence.'),
        ('figure',('full_bank_temporal_traffic',174)),
        ('p','Winner-only retrieval reduces logical value reads by N in this one-sample scenario. '
         'Including the key reads, total K/V access improves by at most about 2×. '
         'These counts are logical accesses, not measured off-chip transfers, cache behavior or joules. '
         'Multiple winners increase value reads. Explicit digital probability normalization is avoided '
         'in a physical race, but clock circuitry and rate setting still have costs.'),
        ('small','Shared content/projection structure isolates the attention substitution; this is not a quality-matched '
         'fit of our current receiver model. Additional receiver-alternative teaching, indexing and scheduling must '
         'be charged when present. Bounded candidate search is a separate coverage hypothesis. '
         'See theory note 48, §§313–323; measured quality/work curves remain in the appendix.'),
    ]]
    # Put new completed evidence immediately after the cover, before hypotheses.
    opening_index=next((i+1 for i,p in enumerate(pages) if p[0][1]=='Native strengths: useful time and private state'),
                       2 if len(pages)>1 and pages[1][0][1]=='New evidence: quality and complete work' else 1)
    pages[opening_index:opening_index]=architectural_pages
    pages.extend(runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))['pages'](tasks['native_language_batched']))
    pages.extend(runpy.run_path(str(ROOT/'report/current_dvs_appendix.py'))['pages'](tasks['current_dvs_appendix']))
    pages.extend(runpy.run_path(str(ROOT/'report/reception_evidence.py'))['pages'](tasks['reception_evidence']))
    pages.extend(runpy.run_path(str(ROOT/'report/route_calibration_state_scope.py'))['pages'](tasks['route_calibration_evidence']))
    pages.extend(runpy.run_path(str(ROOT/'report/score_bound_evidence.py'))['pages'](tasks['score_bound_evidence']))
    pages.extend(runpy.run_path(str(ROOT/'report/trained_route_coverage_evidence.py'))['pages'](tasks['bridge_training_evidence']))
    return pages


def markdown(pages):
    def convert(text):
        text = re.sub(r"<b>(.*?)</b>", r"**\1**", text)
        text = re.sub(r"<i>(.*?)</i>", r"*\1*", text)
        text = re.sub(r'<a href="([^"]+)">(.*?)</a>', r"[\2](\1)", text)
        return html.unescape(text)
    out = []
    for page in pages:
        for kind, value in page:
            if kind in ("title","h1","h2"):
                out.append({"title":"# ","h1":"## ","h2":"### "}[kind]+convert(value))
            elif kind == "bullets":
                out.append("\n".join("- "+convert(t) for t in value))
            elif kind == "table":
                header, rows, widths = value
                out.append("\n".join(["| "+" | ".join(header)+" |", "| "+" | ".join("---" for _ in header)+" |"]+
                                    ["| "+" | ".join(convert(t) for t in row)+" |" for row in rows]))
            elif kind == "figure":
                name, width = value
                target=name if str(name).endswith(".png") else f"report/figures/{name}.png"
                out.append(f"![{Path(name).stem.replace('_', ' ')}]({target})")
            else:
                out.append(convert(value))
    # Editorial blocks can end with a space; generated Markdown must stay clean
    # for git diff --check.
    rendered = "\n\n".join(out)
    return "\n".join(line.rstrip(" \t") for line in rendered.splitlines())+"\n"


def build(M):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle
    tasks, ev = results(), evidence(M)
    figures(M, tasks, ev)
    pages = blocks(M, tasks, ev)
    archive = ROOT/"report/archive/20260929_before_shared_model.md"
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        archive.write_text((ROOT/"REPORT.md").read_text())
    (ROOT/"REPORT.md").write_text(markdown(pages))
    st = M["styles"]()
    st["body"].fontSize = 9.7; st["body"].leading = 14.1; st["body"].spaceAfter = 7
    st["bullet"].fontSize = 9.5; st["bullet"].leading = 13.4; st["bullet"].spaceAfter = 7
    st["small"].fontSize = 8.0; st["small"].leading = 11.2
    st["h1"].fontSize = 14; st["h1"].leading = 19
    st["cell"].fontSize = 8.7; st["cell"].leading = 12.1
    flow = []
    for index, page in enumerate(pages):
        if index:
            flow.append(PageBreak())
        for kind, value in page:
            if kind == "figure":
                name, width = value
                flow.append(M["png"](name, width))
            elif kind == "bullets":
                flow.extend(M["bullets"](value, st))
            elif kind == "table":
                header, rows, widths = value
                # Escape data text while preserving our explicit table line breaks.
                data = [[Paragraph(html.escape(t).replace("&lt;br/&gt;", "<br/>"), st["cell"])
                         for t in row] for row in [header]+rows]
                table = Table(data,colWidths=[w*mm for w in widths],repeatRows=1,hAlign="LEFT")
                table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf3fb")),
                    ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor(M["BLUE"])),
                    ("BOTTOMPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),
                    ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#dbe1e8"))]))
                flow.append(table)
            else:
                flow.append(Paragraph(value,st["body" if kind == "p" else kind]))
    pdf = ROOT/"report/sleeping_machines_status.pdf"
    temporary = pdf.with_suffix(".building.pdf")
    doc = SimpleDocTemplate(str(temporary),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,
                            bottomMargin=16*mm,title="Sleeping Machines — accomplishments and the shared model",
                            author="Sleeping Machines project")
    doc.build(flow,onFirstPage=M["footer"],onLaterPages=M["footer"])
    # Publish only a complete PDF at the project's single canonical path.
    temporary.replace(pdf)
    print("wrote",pdf,"and REPORT.md")
