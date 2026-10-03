"""Completed parameter-space race-sampling diagnosis, preserving prior evidence."""
import hashlib
import copy
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/deep_learning_diagnosis_evidence.py'))
RESULT='diagnostics/local_depth_route_sampling_variance_20261003T033000Z.json'
FUNCTION='diagnostics/local_sampled_credit_functional_forks_20261003T034000Z.json'
ROWS=[('D4 factor',7,'d4pool2_factorized_s7'),('D4 sampled8',7,'d4pool2_le8_s7'),
      ('D4 full',7,'d4pool2_leall_s7'),('D4 factor',8,'d4pool2_factorized_s8'),
      ('D4 sampled8',8,'d4pool2_le8_s8'),('D4 full',8,'d4pool2_leall_s8'),
      ('D6 factor',7,'d6pool2_factorized_s7'),('D6 sampled8',7,'d6pool2_le8_s7')]


def history_read(read,path):
    r=copy.deepcopy(BASE['history_read'](read,path));source=r.get('source_sha256',{})
    name='experiments/dvs_grow_depth_benchmark.py'
    digest='6a6ab3851e1df6d5704570c3b7c2a0d255a1e5a177b49c1de0884df1da63631b'
    if source.get(name)==digest and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        source.pop(name)
        source[f'experiments/archive/frozen_sources/{digest}/dvs_grow_depth_benchmark.py']=digest
    return r


def load(read):
    prior=BASE['load'](lambda p:history_read(read,p));r=read(RESULT);BASE['verify'](r)
    if r['contracts_passed']!=5 or [c['depth'] for c in r['cases']]!=[2,4,6]:
        raise ValueError('Complete contracted depth ladder required')
    v=r['vectors']
    bank=ROOT/v['path']
    if bank.exists() and hashlib.sha256(bank.read_bytes()).hexdigest()!=v['sha256']:
        raise ValueError('Changed parameter-vector bank')
    rows=[]
    for label,seed,name in ROWS:
        stamp='224500Z' if name.startswith('d4pool2_le8_') else '230500Z'
        path='dvs_native/curie_dvs_batched_p16'+name+'_20261002T'+stamp+'.json'
        row=history_read(read,path);BASE['verify'](row)
        if row['work']['fitting_targets']!=7872:raise ValueError('Same fitting denominator required')
        rows.append((label,seed,row))
    f=read(FUNCTION);BASE['verify'](f)
    if f['contracts_passed']!=4 or f['parent_sha256']!=hashlib.sha256((ROOT/'experiments/results'/RESULT).read_bytes()).hexdigest():
        raise ValueError('Complete source-bound actual functional forks required')
    if hashlib.sha256((ROOT/f['artifact']).read_bytes()).hexdigest()!=f['artifact_sha256']:
        raise ValueError('Changed functional predictions')
    return dict(prior=prior,sampling=r,functional=f,rows=rows,full_bank_available=bank.exists())


def pages(d):
    pages=BASE['pages'](d['prior']);rows=[]
    for label,seed,r in d['rows']:
        w=r['work']
        rows.append([label,str(seed),f"{r['final_fit_diagnostic']['nll']:.6f}",
            f"{r['final']['nll']:.6f}",f"{100*r['final']['accuracy']:.3f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.3f}",
            f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.3f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.3f}"])
    pages.append([('h1','Appendix B. Deep race sampling: preserve mixed quality and complete work'),
        ('table',(['Model/credit','Seed','FIT32 NLL','DEV NLL','DEV %','Whole fit GF','Fit MF/target','Infer MF/target'],rows,[27,10,22,22,19,24,25,25])),
        ('p','Completed native p16/H2/pool2 fine20-packet models;984 FIT/192 '
         'subject-disjoint DEV gestures, eight passes/7872 presentations/496 '
         'Adam updates, U16/lr.003/clip1. All columns use identical units and '
         'target denominators. Work includes discovery, losing-value replay, '
         'backward and optimizer; first/last-window estimates, not energy. '
         'FIT32 is the first32 fitting examples at DEV-selected weights.'),
        ('p','D6 factorized learns and improves over D4 seed7, while D6 '
         'sampled8 loses11.458 accuracy points and worsens both saved losses. '
         'Thus failure of this sampled-credit variant is not a universal '
         'depth-capacity failure. D4 full credit improves seed8 by4.6875 '
         'accuracy points and.072620 NLL, but seed7 fails. Preserve this '
         'positive result with replication limits and roughly111-fold '
         'factorized fitting cost. No practical superiority established.'),
        ('p','With21 events, D2/D4/D6 have84/168/252 races per episode. '
         'Sampling eight multiplies every chosen contribution by10.5/21/31.5. '
         'That unbiased rescaling can increase variance; these counts alone '
         'do not quantify interference after the parameter Jacobian, '
         'global clipping or Adam. Theory124 computes that conditional '
         'parameter covariance directly, without a new quality fit.'),
        ('p','Common versus independent race noise is a different question. '
         'The existing trained D2 covariance audit fails both improvement '
         'gates, with ratios near one. Earlier priority experiments also '
         'show score-variance improvements can worsen parameter variance. '
         'Neither a new noise policy nor priority-allocation fit is '
         'nominated from general variance intuition.'),
        ('small','Sparse addressed persistent state, computational clocks and '
         'hard routes remain in every row. Counterfactual losses teach route '
         'choice through detached returns; they do not directly backpropagate '
         'through losing message contents. Language credit horizons and '
         'streaming/reset-segment protocols are separate from these full '
         'DVS episode graphs. No official test or pending result used.')])
    r=d['sampling'];rows=[]
    for c in r['cases']:
        for s in c['distributions']:
            m=s['metrics'];rows.append([str(c['depth']),str(s['k']),
                f"{s['k']/c['races_per_episode'][0]:.3f}",
                f"{s['exact_combined_MSE_over_squared_combined_mean']:.4g}",
                f"{m['combined_cosine']['mean']:.5f}",
                f"{m['Adam_delta_relative_error']['mean']:.4g}",
                f"{m['Adam_delta_cosine']['mean']:.5f}"])
    pages.append([('h1','Appendix B. Conditional race variance: actual parameters and Adam updates'),
        ('table',(['Depth','k','k/R','Exact combined MSE ratio','Mean gradient cosine','Mean Adam rel error','Mean Adam cosine'],rows,[14,12,18,35,34,30,31])),
        ('p','Initialization-only double-precision audit: same represented '
         'float32-initialized p16/H2/pool2 weights promoted to double, seed7, '
         'FIT examples0/1, fixed common history noise. Factories change gain, '
         'parameter count and RNG consumption with depth; this is not an '
         'isolated causal depth perturbation. FIT-only normalization loads '
         'examples0..15 but only0/1 enter this probe; no DEV/test arrays read.'),
        ('p','For every legal race, cache its normalized full factual-score '
         'parameter VJP of sum pi times detached alternative suffix losses. '
         'Its sum matches both the all-race objective and original driver. '
         'A fixed sampled8 cached sum matches a separately differentiated '
         'subset objective using the same returns; it does not independently '
         'execute the sampled8 driver. A tiny actual native R4/k2 case '
         'enumerates all six subsets to verify unbiasedness and covariance.'),
        ('p','Exact trace covariance for uniform k without replacement is '
         'R(R-k)/(k(R-1)) times the sum of squared centered per-race parameter '
         'vectors; independent episode subset covariances add. The displayed '
         'MSE divides by squared FULL combined factual-plus-route gradient, '
         'not only a small canceling route mean. Sixty-four cached draws per '
         'row give descriptive cosine/update means, not fitted quality or '
         'Monte Carlo estimates of the exact covariance.'),
        ('p','Fresh Adam transforms include actual clip1 normalization and '
         'epsilon1e-8; full and sampled updates are independently checked '
         'against two actual discarded Adam forks per depth. Results do not '
         'describe trained moments, convergence or heldout improvement. '
         'Kernel, original parameters and caller RNG are preserved. '
         'Sparse native inference is unchanged.'),
        ('small',f"Five contract groups;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         'Accounting correction beside original artifact scope: every '
         'alternative forward bank is evaluated TWICE, once for vectors '
         'and once for full-driver equivalence; shadow_lanes/events in '
         'each case counts only the first bank. Tiny contracts add their '
         'own work. Cached draws add vector/Adam computation; total '
         'diagnostic FLOPs/traffic/energy unknown, not zero. Full per-race '
         'vectors are a228MiB generated local artifact with an immutable '
         'SHA and reproducible source/queue; report tables use the completed '
         'JSON. Local bank '+('present and hash verified.' if d['full_bank_available'] else
         'absent on this rendering host; vector-byte verification not rerun.'))])
    f=d['functional'];rows=[]
    for c in f['cases']:
        for name in ('full','factorized','k8','k32'):
            selected=[o for o in c['outcomes'] if o['arm']==name or o['arm'].startswith(name+'_')]
            def mean(seed,part,key):
                values=[next(e for e in o['evaluations'] if e['noise_seed']==seed)[part][key] for o in selected]
                return sum(values)/len(values)
            rows.append([f"D{c['depth']} "+name,str(len(selected)),
                f"{mean(171323,'same_FIT','mean_nll_change_from_initial'):+.5f}",
                f"{mean(171323,'anchor_FIT','mean_nll_change_from_initial'):+.5f}",
                f"{mean(171324,'anchor_FIT','mean_nll_change_from_initial'):+.5f}",
                f"{mean(171323,'anchor_FIT','full_to_fork_prediction_KL'):.5f}",
                f"{mean(171324,'anchor_FIT','full_to_fork_prediction_KL'):.5f}"])
    pages.append([('h1','Appendix B. Sampled credit: finite FIT predictions after actual Adam forks'),
        ('table',(['Depth/credit','Forks','Same FIT NLL delta','Anchor NLL delta','Fresh-noise anchor delta','Anchor KL vs full','Fresh-noise KL vs full'],rows,[23,13,29,27,30,26,26])),
        ('p','Same frozen initialization/parameters/gains and normalized '
         'two-example gradients as Theory124. Per depth: factual-only and '
         'full-choice controls, plus the FIRST EIGHT archived draws for k8 '
         'and k32; no selection by outcome. Every fork executes fresh '
         'clip1 Adam.003 and verifies its actual stored displacement. '
         'Parameter ordering and initial original-noise FIT NLL reproduce '
         'the source exactly.'),
        ('p','NLL deltas are relative to each depth\'s unchanged model under '
         'the SAME evaluation noise; negative means improvement. Same FIT '
         'uses0/1, anchors are disjoint unused FIT2..15 with the original '
         'FIT-only normalization. Fresh-noise columns use a fixed second '
         'whole-history draw171324; other columns use original171323. '
         'Anchor examples are not an IID or heldout split. Prediction KL '
         'is from full-credit fork to each arm under matching noise.'),
        ('p','All54 forks improve their two fitting examples under both '
         'noise draws, while ALL worsen these disjoint FIT anchors. AtD6 '
         'full-credit anchor NLL increases.46718/.46173 versus.14452/.11285 '
         'atD4. The tiny gradient batch covers classes0/1, whereas the '
         'anchors contain other classes and adjacent subject recordings. '
         'This exposes a first-step fitting-versus-anchor tradeoff, not '
         'a representative minibatch or trained generalization diagnosis.'),
        ('p','Eight-fork rows show means, not a selected best fit. All54 '
         'actual outcomes, losses, logits and subset indices remain in the '
         'artifacts. Large parameter-step differences do not automatically '
         'mean worse predictions: the finite forward is the relevant '
         'functional check. Conversely, a favorable first step cannot '
         'demonstrate convergence, quality advantage or rescue of a '
         'trained deep model.'),
        ('p','The conditional-choice/factorized-time surrogate need not '
         'descend every literal fixed-noise realized loss; two noises are '
         'a bounded diagnostic, not complete expected-risk integration. '
         'Detached losing contents, persistent addressed state, key/value '
         'separation and computational races remain unchanged. No DEV/test '
         'array, dense fit, architecture substitution or trained moment '
         'history was used.'),
        ('small',f"Theory125;four contract groups,{f['wall_s']:.3f}s/{f['max_rss_kb']}KiB. "
         '54 executed optimizer forks,1824 prediction-target evaluations; '
         'paid parent gradients are reused with no new counterfactual '
         'bank. Total diagnostic FLOPs/traffic/energy unknown, not zero. '
         'Other-host live-branch initialization comparisons and the AWS '
         'streaming10M quality matrix remain the integrated priorities.')])
    return pages
