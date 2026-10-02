"""Completed-only ledger for paired integrated memory-repair accounting smokes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);a=parser.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique unused plain tag required')
    started=time.perf_counter();rows=[];references=[]
    for kind in ('native','addressed'):
        path=ROOT/'experiments/results/long_range_core'/f'local_deep_{kind}_smoke_20261002T080800Z.json'
        r=json.loads(path.read_text());assert r['status']=='completed' and r['work']['formula_coverage_complete']
        assert r['work']['fitting_targets']==r['final']['fitted_targets']==191
        assert r['work']['optimizer_steps']==r['final']['optimizer_updates']==3
        assert r['final']['dev']['targets']==128 and math.isfinite(r['final']['dev']['all_bpc'])
        assert r['max_rss_kb']<900000 and r['state_storage_by_pass'][-1]['differentiable_entries']==0
        references.append(r);w=r['work'];args=r['args'];storage=r['state_storage_by_pass'][-1]
        rows.append(dict(model=kind,result=str(path.relative_to(ROOT)),
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fit_characters=192,passes=1,fitting_targets=191,optimizer_updates=3,dev_targets=128,
            dev_bpc=r['final']['dev']['all_bpc'],parameters=r['parameters'],
            core_receiver_capacity=args['depth']*args['heads']*args['pool'],
            selected_core_updates_per_target=args['depth']*args['heads'],
            scored_core_keys_per_target=args['depth']*args['heads']*args['pool'],
            admitted_counterfactual_values_per_fit_target=args['depth']*args['heads']*args['pool'],
            activity_source='Fixed core dimensions/code-derived; not a separate measured activity trace',
            context_slot_capacity=args['buckets'] if kind=='addressed' else 0,
            occupied_context_slots=storage.get('context_slots',0),persistent_tensor_bytes=storage['persistent_tensor_bytes'],
            cpu_whole_fit_gflops=w['cpu_whole_fit_unit_special_flops']/1e9,
            cpu_fit_mflops_per_target=w['cpu_fit_unit_special_flops_per_target']/1e6,
            cpu_inference_mflops_per_target=w['cpu_inference_unit_special_flops_per_target']/1e6,
            inference_targets=1,inference_warm_tokens=127,
            wall_s=r['wall_s'],max_rss_kb=r['max_rss_kb']))
    left,right=references
    for key in ('data_sha256','source_sha256','initial_dev'):
        # Historical empty filler group is NaN; compare defined initial scores.
        if key=='initial_dev':
            for name in ('all_bpc','target_bpc','targets','target_accuracy'):assert left[key][name]==right[key][name]
        else:assert left[key]==right[key]
    for key in ('fit','dev','epochs','chunk','update_targets','payload','depth','heads','pool','seed','lr'):
        assert left['args'][key]==right['args'][key]
    dependencies=['experiments/results/diagnostics/local_deep_memory_contracts_20261002T080200Z.json',
                  'experiments/results/diagnostics/local_deep_core_attribution_20261002T075000Z.json',
                  'experiments/results/diagnostics/local_long_range_protocol_20261002T075400Z.json']
    for name in dependencies:assert json.loads((ROOT/name).read_text())['status']=='completed'
    result=dict(status='completed',common_unit_ledger=rows,
        source_sha256={'experiments/deep_feature_preflight_analysis.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        dependencies_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in dependencies},
        matched_data_optimizer_noise_and_core_initialization=True,
        whole_fit_work_ratio=rows[1]['cpu_whole_fit_gflops']/rows[0]['cpu_whole_fit_gflops'],
        smoke_bpc_gain=rows[0]['dev_bpc']-rows[1]['dev_bpc'],
        arithmetic_convention='2FLOPs/MAC plus unit-weight specials; CPU-emulator whole-smoke fitting, excludes dev/RNG/hash/traffic; no energy claim',
        historical_protocol_notes=['Original smoke JSONs have NaN for empty filler group on text, which has all positions marked targets. Defined quality/work metrics are preserved; driver now emits null and strict JSON for missing groups.',
            'Original inference trace covers one target after127 warm tokens; denominator explicitly recorded here. This is a numerical accounting sample, not an inference-throughput measurement.',
            'Cold dev state differs from fit-prefilled count references; no such quality comparison admitted.'],
        scope='Full-size integrated accounting prerequisite on192 fit characters/one pass, not a quality pilot or feature/supremacy claim. No automatic scale-up; whole-driver checkpoint recovery/selection/resource protocol still needed before long fits.',
        official_test_read=False,wall_s=time.perf_counter()-started)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=a.tag)),flush=True)


if __name__=='__main__':main()
