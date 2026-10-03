"""Completed bridge training contracts, restricted smokes and native utility scope."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={'contracts':'diagnostics/local_dvs_bridge_contracts_retry_20261003T002300Z.json',
       'audit':'diagnostics/local_dvs_bridge_utility_20261003T002700Z.json'}


def load(read):
    data={k:read(v) for k,v in FILES.items()}
    for r in data.values():
        if r['status']!='completed':raise ValueError('Completed bridge training evidence required')
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed bridge source '+name)
    for row in data['audit']['common_unit_ledger']+data['audit']['saved_references']:
        if hashlib.sha256((ROOT/row['result']).read_bytes()).hexdigest()!=row['result_sha256']:
            raise ValueError('Changed bridge parent')
    return data


def pages(data):
    audit=data['audit'];ledger=audit['common_unit_ledger'];rows=[]
    for r in ledger:
        rows.append(['Hard' if r['alpha']==0 else 'Bridge .1',f"{100*r['dev_accuracy']:.2f}",f"{r['dev_nll']:.6f}",
            f"{r['whole_fit_gflops']:.6f}",f"{r['fit_mflops_per_presentation']:.6f}",f"{r['inference_mflops_per_prefix']:.6f}",str(r['fitting_presentations'])])
    for name,r in zip(['Saved p16/L2/all','Saved p16/L4/k8'],audit['saved_references']):
        rows.append([name,f"{100*r['dev_accuracy']:.2f}",f"{r['dev_nll']:.6f}",f"{r['whole_fit_gflops']:.6f}",
            f"{r['fit_mflops_per_presentation']:.6f}",f"{r['inference_mflops_per_prefix']:.6f}",str(r['fitting_presentations'])])
    first=[('h1','Appendix B. Bounded score bridge: integrated training contracts'),
        ('p','Eight double-precision contracts pass before fitting: alpha0 logits and '
         'all parameter/content gradients exactly nest the old batched path; positive '
         'bridge matches independent sequential factual and full first-time alternative '
         'replays; real addressed writes differ; full-route gradients, normalization, '
         'clipping and Adam agree. Partial-window model/Adam/RNG/cursor continuation '
         'is bitwise exact. Full/partial fitting stages and selected-value native '
         'inference have complete operation coverage. Explicit zero gradients in batch '
         'and absent unused sequential gradients are equivalent; any nonzero mismatch fails.'),
        ('table',(['Variant','Dev acc %','Dev NLL','Whole fit GF','Fit MF/presentation','Infer MF/prefix','Presentations'],rows,[36,21,27,27,33,28,21])),
        ('p','Matched smoke arms: seed7,p8/L4/H2/pool2,32FIT/16DEV,two passes/64 '
         'presentations,U8/lr.003,eight Adam updates,all168 races and336 alternative '
         'shadow lanes per21-event prefix.7899 parameters,16available receivers, '
         '16key scores/eight actual writes per event,720live state bytes in the '
         '11audited DEV prefixes. Keys/values, temporal computation, source carry '
         'and sparse persistent writes remain; fitting computes all candidate values.'),
        ('p','Both arms learn on FIT: initial/final NLL2.829119/2.037279 hard and '
         '2.829112/2.039687 bridge. Equal31.25% small DEV accuracy; bridge NLL '
         'is worse by.00002646. Bridge adds.5967% counted fitting and.8070% inference '
         'work. Crucially, neither initial nor selected smoke model saturates a single '
         'one of5376 FIT races (max raw score2.59 initially/2.30 selected). This is '
         'integration evidence; it does not test saturation repair or nominate a larger fit.'),
        ('small','Contracts9.065s/344424KiB; first missing-versus-zero comparison '
         'failure retained. Smokes61.158/64.604s,373200/374120KiB under one-host '
         'guard. Saved references use984FIT/192DEV/eightpasses/7872presentations: '
         'different width/data/quality; raw whole-fit gaps106.498x/21.947x versus '
         'hard smoke are unsupported as efficiency advantages. Same units and target '
         'denominators within each column;2FLOPs/MAC plus unit specials, first11 DEV '
         'inference convention. Preprocessing/traffic/RNG/energy separate; no official test.')]
    rows=[]
    for r in audit['utility_rows']:
        capped=next(i for i,v in enumerate(r['raw_scores']) if abs(v)>12)
        hard,bridge=r['conditional_choice'];rows.append([str(r['fit_index']),
            '/'.join(str(x) for x in r['site']),f"{r['raw_scores'][capped]:.4f}",f"{r['utility_gap']:+.6f}",
            f"{hard['raw_score_choice_derivative'][capped]:.2e}",f"{bridge['raw_score_choice_derivative'][capped]:+.2e}"])
    second=[('h1','Appendix B. Restored native sensitivity has small, mixed route utility'),
        ('table',(['FIT','Event/layer/head','Raw capped score','L0 minus L1','Hard emitter credit','Bridge emitter credit'],rows,[17,39,32,31,36,38])),
        ('p','Five prespecified strictly saturated seed7 histories from theory104, '
         'fixed-pass4 producer, FIT258/981. For each, both actual alternatives write '
         'persistent receiver state at the factual FIRST arrival. Prefix winners/times '
         'are fixed; suffix clocks, messages and memory evolve with the original hard '
         'map. Original-winner replays recover all original logits/state/end RNG. '
         'Ten alternative plus five factual native forwards are paid;168additional raw '
         'dot products per diagnostic forward. These FIT labels are revealed for utility '
         'diagnosis and are no longer label-unused. No optimizer/DEV quality/test selection.'),
        ('p','L0-L1 is positive in one case, so its alternative improves NLL; four '
         'negative gaps mean the factual winner is better. Every previously blocked '
         'emitter has nonzero bridge choice sensitivity, confirmed by double analytic '
         'derivatives and finite-difference gradcheck. Absolute restored derivatives '
         'are only6.14e-11 to5.36e-9: alternative probabilities remain1.46e-7 to '
         '8.77e-6. Restoring rank does not by itself make those routes useful or '
         'their contribution to expected risk substantial.'),
        ('p','Conditional choice risk is sum pi_i*L_i, so dR/draw0= '
         'pi0*pi1*(L0-L1)*f\'(raw0). Utilities and first time are held fixed; '
         'this tests choice credit only. The bridge is applied to the current '
         'conditional score map in this derivative, not the suffix law. These numbers '
         'are not a full positive-bridge risk, joint time/content update or learning result.'),
        ('small','Audit18.432s/328716KiB; diagnostic whole arithmetic/traffic/energy '
         'unknown, not zero. Conditional two-score Fisher determinant is '
         'pi0*pi1*d0^2*d1^2 times the raw-gradient Gram determinant. Positive slopes '
         'preserve existing raw rank; confidence and nearly parallel raw directions '
         'can still suppress useful learning. Theory106/107 retain exact scopes. '
         'No automatic bridge scale-up; independently owned integrated deeper replay '
         'and reserved AWS comparisons remain the quality priority.')]
    return [first,second]
