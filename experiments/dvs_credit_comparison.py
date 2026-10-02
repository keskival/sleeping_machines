"""Validate completed fixed-budget credit comparisons and preserve common units."""
import argparse
import json
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    import hashlib
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def completed(path):
    r = json.loads((ROOT/path).read_text())
    if r['status'] != 'completed':raise ValueError('Completed evidence required')
    for source,digest in r.get('source_sha256', {}).items():
        if sha(source) != digest:raise ValueError('Changed frozen source: '+source)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    p.add_argument('--reference', required=True); p.add_argument('--treatment', required=True)
    p.add_argument('--kind', choices=['terminal_pairs','state_clock'], required=True)
    p.add_argument('--maximum-work-ratio', type=float, required=True)
    a = p.parse_args(); started = time.perf_counter()
    out = ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name != a.tag or out.exists():raise ValueError('Unused tag required')
    reference, treatment = completed(a.reference), completed(a.treatment)
    for key in ('data','parameters','initial_dev','initial_fit','window_size_counts'):
        if reference[key] != treatment[key]:raise ValueError('Unmatched '+key)
    for key in ('payload','depth','heads','pool','seed','epochs','fit','dev','lr','update_targets'):
        if reference['args'][key] != treatment['args'][key]:raise ValueError('Unmatched setting '+key)
    settings = reference['args']
    if (settings['fit'],settings['dev'],settings['epochs'],settings['update_targets']) != (256,192,4,16):
        raise ValueError('Fixed pilot protocol required')
    if reference['args'].get('terminal_risk') != 'local':raise ValueError('Saved local reference required')
    if a.kind == 'terminal_pairs' and treatment['args'].get('terminal_risk') != 'pairs':raise ValueError('Pair treatment required')
    if a.kind == 'state_clock' and not treatment.get('credit_protocol',{}).get('inference_architecture_unchanged'):
        raise ValueError('State-clock treatment required')
    rows = []
    for label,name,r in [('local',a.reference,reference),(a.kind,a.treatment,treatment)]:
        if len(r['curve']) != 4 or r['selected_epoch'] != min(r['curve'],key=lambda x:x['dev']['nll'])['epoch']:
            raise ValueError('Fixed pass selection changed')
        if not all(s['formula_coverage_complete'] for sample in r['work_samples'] for s in sample['stages'].values()):
            raise ValueError('Incomplete work')
        w = r['work']; n = w['fitting_targets']
        if n != 1024 or w['optimizer_updates'] != 64:raise ValueError('Changed exposure')
        if abs(w['whole_fit_unit_special_flops_estimate']/n-w['fit_unit_special_flops_per_target_estimate']) > 1e-6:
            raise ValueError('Inconsistent denominator')
        rows.append(dict(arm=label,result=name,result_sha256=sha(name),distinct_fit_targets=256,
            fit_presentations=n,development_accuracy=r['final']['accuracy'],development_nll=r['final']['nll'],
            whole_fit_gflops_estimate=w['whole_fit_unit_special_flops_estimate']/1e9,
            fit_mflops_per_presentation_estimate=w['fit_unit_special_flops_per_target_estimate']/1e6,
            inference_mflops_per_target_estimate=w['inference_unit_special_flops_per_target_estimate']/1e6,
            parameters=r['parameters'],available_receivers=w['native_available_receivers'],
            key_scores_per_fit_presentation=r['activity']['key_scores']/n,
            selected_updates_per_fit_presentation=r['activity']['selected_updates']/n,
            counterfactual_values_per_fit_presentation=r['activity']['counterfactual_values']/n,
            curve=[dict(epoch=x['epoch'],nll=x['dev']['nll'],accuracy=x['dev']['accuracy']) for x in r['curve']],
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
    gain = rows[0]['development_nll']-rows[1]['development_nll']
    decline = 100*(rows[0]['development_accuracy']-rows[1]['development_accuracy'])
    ratio = rows[1]['whole_fit_gflops_estimate']/rows[0]['whole_fit_gflops_estimate']
    gate = gain >= .02 and decline <= 1 and ratio <= a.maximum_work_ratio and max(x['max_rss_kb'] for x in rows)<900000
    result = dict(status='completed',args=vars(a),common_unit_ledger=rows,
        nll_improvement=gain,accuracy_decline_percentage_points=decline,whole_fit_work_ratio=ratio,
        promotion_gate_passed=gate,gate=dict(minimum_nll_improvement=.02,maximum_accuracy_decline_pp=1,
            maximum_work_ratio=a.maximum_work_ratio,maximum_rss_kb=900000),
        source_sha256={'experiments/dvs_credit_comparison.py':sha('experiments/dvs_credit_comparison.py')},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Single-seed fixed256-fit/192-dev/four-pass screen, minimum devNLL selection. Every branch replay and optimizer charged. Full984-fit strong controls have unequal fitting data and remain separate references. No official test, confirmation or practical superiority claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':main()
