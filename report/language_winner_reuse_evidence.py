"""Equivalent winner-return reuse with paid production work and numeric gates."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/language_replay_driver_evidence.py'))
FILES=dict(contracts='diagnostics/local_language_winner_reuse_contracts_retry_20261003T014400Z.json',
           resource='diagnostics/local_language_winner_reuse_resource_audit_20261003T014800Z.json')


def load(read):
    data={'prior':BASE['load'](read),**{k:read(v) for k,v in FILES.items()}}
    for key,count in [('contracts',28),('resource',6)]:
        r=data[key]
        if r['status']!='completed' or r['contracts_passed']!=count:raise ValueError('Completed winner-reuse diagnostic required')
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed winner-reuse source '+name)
    if data['resource']['contract_result_sha256']!=hashlib.sha256((ROOT/'experiments/results'/FILES['contracts']).read_bytes()).hexdigest():
        raise ValueError('Changed winner-reuse admission result')
    return data


def pages(data):
    pages=BASE['pages'](data['prior']);r=data['resource'];rows=[];details=[]
    for row in r['common_unit_ledger']:
        rows.append([row['family']+'/'+row['arm'],str(row['parameters']),
            f"{row['whole_step_unit_special_flops']/1e9:.6f}",f"{row['fit_unit_special_flops_per_target']/1e6:.6f}",
            f"{row['inference_unit_special_flops_per_target']/1e6:.6f}",str(row['shadow_lanes'])])
    for f in r['families']:
        ge=f['gradient_error'];details.append(f"{f['family']}: counted fitting work "
            f"-{100*f['counted_full_fitting_work_saving_fraction']:.3f}%; gradient relative L2 "
            f"{ge['relative_l2']:.3g}, max absolute {ge['maximum_absolute']:.3g}; "
            f"{len(ge['failed_parameter_tolerances'])} parameter tensors miss the coordinate tolerance; "
            f"actual Adam-update relative difference {f['relative_actual_Adam_update_error']:.3g}. "
            f"Production numerical gate {'PASSES' if f['production_gradient_and_update_admission_passed'] else 'FAILS'}.")
    pages.append([('h1','Appendix B. Factual-winner reuse: measured work and numerical limits'),
        ('table',(['L8/p16 family/arm','Params','Whole fit GF','Fit MF/target','Infer MF/target','Shadow lanes'],rows,[34,20,29,30,34,27])),
        ('p','Every sampled factual winner already supplies its true downstream '
         'loss. Forcing that SAME receiver at the SAME first time, with the '
         'same entering state/future draws and actual write, reproduces the '
         'factual trajectory. Reuse this detached return and simulate only '
         'different outcomes. The categorical local expected-return objective '
         'and every earlier score/producer VJP are identical in exact arithmetic; '
         'native clock/content derivatives and inference stay unchanged. '
         'At two candidates this removes half the shadows; it changes neither '
         'credit horizon nor candidate support.'),
        ('p','Theory115:28private/shared double depth8 contracts for pools1/2/4 '
         'match EVERY parameter gradient against both old batched and independent '
         'sequential enumeration. ALLfactual-winner replay outcomes agree, '
         'winner-recording primal/state/RNG/pathwise gradients nest bitwise, '
         'causal token/label and exact pending-gradient/Adam recovery pass. '
         'Production rows above each pay ONE synthetic sixteen-target update '
         'from the SAME nonempty state/parameters/RNG:512→256shadow lanes and '
         '8192→4096shadow events. Actual normalization/clip/warmup/Adam and '
         'all factual/shadow backward are included, with complete operator coverage.'),
        ('p',' '.join(details)),
        ('small',f"Theory116;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         'Predeclared float32 coordinate rtol3e-4/atol3e-6, global gradient '
         'relative error<=3e-5 and Adam-update relative error<=.005 are retained. '
         'Initial strict run stopped at a key-read coordinate mismatch; the '
         'completed audit reports all gates without weakening them. '
         'Nonempty Adam/private state/pending three-target recovery and next '
         'partial update are bitwise exact. Tables use the SAME16target denominator '
         'and2FLOPs/MAC+unit-special convention for all arms. Extra recovery '
         'steps are separately paid diagnostic work. No data-fit BPC, energy, '
         'physical projection or benchmark superiority claim; existing AWS10M '
         'teacher/factorized/full-replay sources and queues remain untouched.')])
    return pages
