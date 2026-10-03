"""Joint native branch-content admission, complete small-fit work and scope."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/actual_batch_credit_evidence.py'))
CONTRACT='diagnostics/local_conditional_branch_content_contracts_20261003T041000Z.json'
SMOKE='diagnostics/local_conditional_content_integrated_smoke_retry_20261003T042500Z.json'
HISTORY=runpy.run_path(str(ROOT/'report/depth_sampling_evidence.py'))
REFERENCES=[('Saved p16 D4 factorized','dvs_native/curie_dvs_batched_p16d4pool2_factorized_s7_20261002T230500Z.json'),
    ('Saved p16 D4 full replay','dvs_native/curie_dvs_batched_p16d4pool2_leall_s8_20261002T230500Z.json')]
def verify(r):
    if r['status']!='completed':raise ValueError('Completed evidence required')
    for name,digest in r['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed branch-credit source '+name)
def load(read):
    d=dict(prior=BASE['load'](read),contracts=read(CONTRACT),smoke=read(SMOKE),references=[])
    for key in ('contracts','smoke'):verify(d[key])
    if d['contracts']['contracts_passed']!=8 or d['smoke']['contracts_passed']!=4:raise ValueError('All admitted stages required')
    for label,path in REFERENCES:
        r=HISTORY['history_read'](read,path)
        if r['status']!='completed':raise ValueError('Completed reference required')
        d['references'].append((label,r))
    return d
def pages(d):
    pages=BASE['pages'](d['prior']);r=d['contracts'];e=r['exposure']
    pages.append([('h1','Appendix B. Live counterfactual messages: conditional content and route credit'),
        ('p','The prior replay learns route utility from detached alternative '
         'losses, while the realized winner supplies content gradients. This '
         'need not be biased, but losing message functions receive no direct '
         'payload derivative on that realization. The new training-only '
         'sibling differentiates through every alternative of ONE sampled '
         'race, including complete live prefixes and private writes.'),
        ('p','At a uniformly sampled legal race r, objective = sum '
         'stopgrad(pi_i)*L_i + R*sum pi_i*stopgrad(L_i), averaged over actual '
         'examples. The branch-content average REPLACES factual loss '
         'gradient; adding both would double-count. Only sampled choice '
         'credit gets legal-race scaling R. Native computational delays, '
         'factorized first-time gradients, sparse private state, key/value '
         'separation and inference remain unchanged.'),
        ('p','For ONE episode at fixed entering history/first time and '
         'independent future draws, the branch average is the conditional '
         'mean of the native pathwise estimator. It removes current-winner '
         'variance at that site. Conditioning on the entire candidate noise '
         'would reveal the winner. Shared noise across episodes means this '
         'is not a lower BATCH-variance theorem. Global clipping and Adam '
         'are nonlinear; their expected updates need not be preserved.'),
        ('table',(['Native one-event losing output map','Gradient L2'],[
            ['Factual payload gradient',f"{e['factual_loser_output_gradient_norm']:.9g}"],
            ['Joint conditional gradient',f"{e['joint_loser_output_gradient_norm']:.9g}"],
            ['Loser selection probability',f"{e['pi'][e['loser']]:.9g}"]],[118,56])),
        ('p','Eight contract groups cover EVERY parameter against explicit '
         'live branches/decomposition, independent pure-Torch factorized '
         'clock algebra, factual-forced-winner identity, actual first-time '
         'preservation, existing BLk1 choice credit, relabeling-invariant '
         'factual predictions, real normalized clip1 Adam, byte-serialized '
         'next-update recovery and complete operator coverage. Early/late '
         'and unequal-length native episodes are included.'),
        ('p','The independent reference holds each recorded branch topology '
         'and clock latent Z=Lambda*T fixed, then differentiates T=Z/Lambda. '
         'It does not differentiate a fixed-candidate-noise minimum or '
         'detach physical time. Future hard-choice boundaries remain scoped '
         'native derivatives; this is not exact whole-risk differentiation.'),
        ('small',f"Theory129;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. Known "
         'conditional averaging and stochastic-computation-graph primitives '
         '(Schulman et al.2015); the native coupled-clock/content application '
         'is implemented and contracted, not a priority claim. Source/kernel/'
         'RNG preserved; total contract-campaign work unknown, not zero.')])
    r=d['smoke'];rows=[]
    for arm in r['arms']:
        w=arm['work'];rows.append([arm['arm'].replace('_',' '),'24 / 2',
            f"FIT-anchor / {arm['after']['anchor']['nll']:.5f}",
            f"{w['whole_fit_unit_special_flops']/1e9:.6f}",f"{w['fit_unit_special_flops_per_target']/1e6:.6f}",
            f"{w['inference_unit_special_flops_per_target']/1e6:.6f}"])
    for label,v in d['references']:
        w=v['work'];rows.append([label,'984 / 8',f"DEV / {v['final']['nll']:.5f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.6f}",f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
    quality=[[a['arm'].replace('_',' '),f"{a['before']['fit']['nll']:.6f}",f"{a['after']['fit']['nll']:.6f}",
        f"{a['before']['anchor']['nll']:.6f}",f"{a['after']['anchor']['nll']:.6f}",str(a['learning_gate'])] for a in r['arms']]
    choice,joint=r['arms']; ratio=joint['work']['whole_fit_unit_special_flops']/choice['work']['whole_fit_unit_special_flops']
    raw_gaps=[v['work']['fit_unit_special_flops_per_target_estimate']/joint['work']['fit_unit_special_flops_per_target'] for _,v in d['references']]
    pages.append([('h1','Appendix B. Integrated depth4 content learning: paid FIT-only admission'),
        ('table',(['Credit','Initial FIT NLL','Final FIT NLL','Initial anchor','Final anchor','Learn gate'],quality,[36,29,29,29,29,22])),
        ('table',(['Model/credit','Data / passes','Eval split / NLL','Whole fit GF','Fit MF/target','Infer MF/target'],rows,[40,20,34,27,27,26])),
        ('p',f"Both pass the fixed learning gate, but joint content is worse by {joint['after']['fit']['nll']-choice['after']['fit']['nll']:.6f} FIT NLL and {joint['after']['anchor']['nll']-choice['after']['anchor']['nll']:.6f} anchor NLL, with {ratio:.3f}x fitting work. No promotion or retuning. Saved factorized/full replay fitting work per presentation is {raw_gaps[0]:.2f}x/{raw_gaps[1]:.2f}x the joint smoke; unequal width/data/quality prevent a superiority claim."),
        ('p','SAME fresh p4/D4/H2/pool2, original fine21-event FIT0..23, '
         'fixed disjoint FIT24..31 anchors, two passes/48 presentations/'
         '12 Adam updates per arm, B4/lr.003/clip1. Identical initialization/'
         'order/sites/common noise, ordinary gates, no growth or pretrained '
         'weights. Anchors are adjacent FIT recordings, not IID or official '
         'DEV/test. Every update is fully traced; source contracts include '
         'actual real-depth4 EVERY-gradient decomposition and BLk1 identity.'),
        ('p','Each smoke keeps16 available private receivers, eight selected '
         'updates and16 key scores per event;21 events means168 selected '
         'updates/336 scores per target. Full-prefix shadows96 lanes/2016 '
         'events per fit are charged, including ALL live-branch backwards '
         'in the joint arm. Objective normalization is charged inside '
         'forward; no second scaling. Native inference is identical in '
         'structure and counted on the same32-target boundary.'),
        ('p','The saved full-data rows retain stronger quality, other widths '
         'and7872 presentations, with first/last-window work estimates. '
         'Smoke work is exact over48 presentations. EVERY table column '
         'uses common units and presentation denominators; unequal '
         'quality/data make raw work gaps admission evidence, not '
         'comparable-quality superiority. Evaluation/verification and '
         'failed-attempt work are separate unknown work, not zero.'),
        ('p','Original042000Z run hit its180s guard after choice-only '
         'completed; joint was incomplete. Preserve original source/'
         'protocol/logs and48-96 failed fitting-presentation bound. Retry '
         'changes only the timeout to420s based on measured tracing cost; '
         'same data/passes/settings, no loss-based tuning or extension. '
         'Incomplete scores are excluded from this paired table.'),
        ('small',f"Theory130;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. Tiny "
         'single-seed learning/resource admission only: no useful deep-feature '
         'attribution, batch variance reduction, heldout confirmation or '
         'supremacy. Existing integrated live-gate comparisons and AWS '
         'streaming10M remain priority; no active source replaced.')])
    return pages
