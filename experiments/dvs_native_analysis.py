"""Completed-only real-packet native/strong-control comparison; no refitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed(name):
    path = ROOT / name
    result = json.loads(path.read_text())
    if result.get('status') != 'completed':
        raise ValueError('Completed result required: ' + name)
    for source, digest in result.get('source_sha256', {}).items():
        if sha(ROOT / source) != digest:
            raise ValueError('Changed source: ' + source)
    return result


def validate_score(score, targets):
    probabilities = score['probabilities']
    if len(probabilities) != targets or score['targets'] != targets:
        raise ValueError('Mismatched development targets')
    for values in probabilities:
        if len(values) != 11 or any(not math.isfinite(p) or p < 0 for p in values):
            raise ValueError('Invalid class probabilities')
        if abs(sum(values) - 1) > 1e-5:
            raise ValueError('Unnormalized class probabilities')
    if not math.isfinite(score['nll']) or not 0 <= score['accuracy'] <= 1:
        raise ValueError('Invalid quality')


def analyze(native_path, controls_path):
    native = completed(native_path)
    controls = completed(controls_path)
    if native['data']['controls_sha256'] != sha(ROOT / controls_path):
        raise ValueError('Changed comparator')
    if native['data']['data_result_sha256'] != controls['data_result_sha256']:
        raise ValueError('Different data')
    if native['data']['protocol'] != controls['protocol'] or controls['protocol']['official_test_read']:
        raise ValueError('Different or opened-test protocol')
    args = native['args']
    if (args['epochs'], args['update_targets'], args['seed'], args['fit'], args['dev']) != (8, 16, 6, None, None):
        raise ValueError('Fixed full-stage protocol required')
    if len(native['curve']) != 8 or [x['epoch'] for x in native['curve']] != list(range(1, 9)):
        raise ValueError('All fixed passes required')
    selected = min(native['curve'], key=lambda x: x['dev']['nll'])
    if selected['epoch'] != native['selected_epoch'] or selected['dev'] != native['final']:
        raise ValueError('Changed selection rule or selected result')
    fit = controls['fitting_targets']; dev = controls['development_targets']
    work = native['work']; activity = native['activity']
    if fit != 984 or dev != 192 or work['fitting_targets'] != 8 * fit or activity['targets'] != 8 * fit:
        raise ValueError('Mismatched fitting exposure')
    if work['optimizer_updates'] != 8 * math.ceil(fit / 16) or native['window_size_counts'] != {'16': 488, '8': 8}:
        raise ValueError('Mismatched optimizer budget')
    for sample in native['work_samples']:
        for stage in sample['stages'].values():
            if not stage['formula_coverage_complete'] or stage['unsupported_floating_operators']:
                raise ValueError('Incomplete numerical accounting')
    if abs(work['whole_fit_unit_special_flops_estimate'] / work['fitting_targets'] - work['fit_unit_special_flops_per_target_estimate']) > 1e-6:
        raise ValueError('Different work denominators')
    rows = []
    for arm in controls['rows']:
        validate_score(arm['final'], dev)
        rows.append(dict(arm=arm['arm'], development_accuracy=arm['final']['accuracy'],
            development_nll=arm['final']['nll'], distinct_fit_targets=fit,
            fit_presentations=None, whole_fit_gflops_estimate=None,
            fit_mflops_per_presentation_estimate=None, inference_mflops_per_target_estimate=None,
            fitting_wall_s=arm['fitting_wall_s'], development_inference_wall_s=arm['inference_wall_s'],
            preprocessing_wall_s=controls['preprocessing_wall_s'],
            work_scope='Third-party solver/count calibration FLOPs unmeasured, not zero. Fitting includes internal calibration where applicable; preprocessing separate.',
            support_vectors=arm.get('support_vectors')))
    validate_score(native['final'], dev)
    rows.append(dict(arm='native_temporal_p16_L2_H2_pool2_seed6',
        development_accuracy=native['final']['accuracy'], development_nll=native['final']['nll'],
        distinct_fit_targets=fit, fit_presentations=work['fitting_targets'],
        whole_fit_gflops_estimate=work['whole_fit_unit_special_flops_estimate'] / 1e9,
        fit_mflops_per_presentation_estimate=work['fit_unit_special_flops_per_target_estimate'] / 1e6,
        inference_mflops_per_target_estimate=work['inference_unit_special_flops_per_target_estimate'] / 1e6,
        fitting_wall_s=None, development_inference_wall_s=None,
        workflow_wall_s=native['wall_s'], max_rss_kb=native['max_rss_kb'],
        preprocessing_wall_s=controls['preprocessing_wall_s'],
        parameters=native['parameters'], available_receivers=work['native_available_receivers'],
        state_tensor_bytes=native['final']['max_state_tensor_bytes'],
        key_scores_per_fit_presentation=activity['key_scores'] / activity['targets'],
        selected_updates_per_fit_presentation=activity['selected_updates'] / activity['targets'],
        counterfactual_values_per_fit_presentation=activity['counterfactual_values'] / activity['targets'],
        work_scope='Full prefix/query training forward/backward, key/value candidates, normalization/clipping/Adam; first/last window samples by actual size. Validation/recovery/profiling included in workflow wall; validation forward FLOPs separate estimate.'))
    selected_control = next(x for x in controls['rows'] if x['arm'] == controls['selected_by_dev_nll'])
    best_accuracy = max(x['final']['accuracy'] for x in controls['rows'])
    score = native['final']; reference = selected_control['final']
    quality_frontier = score['accuracy'] >= reference['accuracy'] and score['nll'] < reference['nll']
    evaluation_targets = dev * (1 + args['epochs'] + 1) + 2 * min(32, fit) + min(11, dev)
    return dict(common_unit_ledger=rows, selected_control=selected_control['arm'],
        native_accuracy_gain_over_selected_control=score['accuracy'] - reference['accuracy'],
        native_nll_gain_over_selected_control=reference['nll'] - score['nll'],
        native_accuracy_gain_over_max_grid_accuracy=score['accuracy'] - best_accuracy,
        native_dominates_selected_control_in_development_quality=quality_frontier,
        independent_advantage_proven=False,
        native_nonfitting_prefix_calls=evaluation_targets,
        native_nonfitting_forward_gflops_estimate=evaluation_targets * work['inference_unit_special_flops_per_target_estimate'] / 1e9,
        native_fitting_plus_nonfitting_forward_gflops_estimate=(work['whole_fit_unit_special_flops_estimate'] + evaluation_targets * work['inference_unit_special_flops_per_target_estimate']) / 1e9,
        workflow_comparison=dict(native_wall_s=native['wall_s'], control_grid_wall_s=controls['wall_s'],
            raw_preprocessing_wall_s=controls['preprocessing_wall_s'],
            native_workflow_plus_preprocessing_wall_s=native['wall_s'] + controls['preprocessing_wall_s'],
            control_grid_plus_preprocessing_wall_s=controls['wall_s'] + controls['preprocessing_wall_s']),
        selected_epoch=native['selected_epoch'], learning_passed=native['small_fit_learning_passed'],
        native_initial_development_nll=native['initial_dev']['nll'],
        native_first32_initial_fit_nll=native['initial_fit']['nll'],
        native_first32_selected_fit_nll=native['final_fit_diagnostic']['nll'],
        curve=[dict(epoch=x['epoch'], accuracy=x['dev']['accuracy'], nll=x['dev']['nll']) for x in native['curve']],
        protocol=controls['protocol'],
        inputs={p: sha(ROOT / p) for p in [native_path, controls_path]},
        scope='Completed subject-disjoint DEVELOPMENT comparison, one native seed and fixed conventional grid. No independent test, repeated-native seed, equal-pass/resource proof, measured energy, useful depth or dormant-capacity advantage. Solver FLOPs unknown; native window/inference samples are estimates. Numerical checks and exploratory predecessor fits are retained separately as research cost.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--native', required=True); p.add_argument('--controls', required=True)
    a = p.parse_args(); started = time.perf_counter()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unused plain tag required')
    result = analyze(a.native, a.controls)
    result.update(status='completed', args=vars(a), wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={'experiments/dvs_native_analysis.py': sha(__file__)})
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
