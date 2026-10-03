"""Completed native clamp geometry, bounded restoration and replay confirmations."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES={
 'geometry':'diagnostics/local_dvs_conditional_geometry_20261002T234000Z.json',
 'bridge':'diagnostics/local_dvs_bounded_score_sensitivity_20261002T234600Z.json',
 'exposure':'diagnostics/local_score_bound_exposure_20261002T235200Z.json',
 'priority':'diagnostics/aws_parameter_replay_priority_20261002T232800Z.json',
 'query':'diagnostics/aws_query_only_native_20261002T233000Z.json',
 'deep_summary':'diagnostics/aws_deep_replay_20261002T234200Z_summary.json',
 'teacher':'dvs_native/aws_deep_replay_20261002T234200Z_d4_teacher_pilot_s7.json',
 'factorized':'dvs_native/aws_deep_replay_20261002T234200Z_d4_factorized_pilot_s7.json',
 'replay':'dvs_native/aws_deep_replay_20261002T234200Z_d4_replay_pilot_s7.json',
 'tied2':'dvs_native/aws_coarse_tied_20261002T233600Z_p2_pilot_s6.json',
 'tied8':'dvs_native/aws_coarse_tied_20261002T233600Z_p8_pilot_s6.json',
 'untied':'dvs_native/aws_coarse_native_20261002T212600Z_coarse_matchedclock_s6.json'}
FINE=['curie_dvs_batched_p16d2pool2_leall_s6_20261002T224500Z.json',
 'curie_dvs_batched_p16d2pool2_leall_s7_20261002T224500Z.json',
 'curie_dvs_batched_p16d2pool2_leall_s8_20261002T224500Z.json',
 'curie_dvs_batched_p16d4pool2_le8_s7_20261002T224500Z.json',
 'curie_dvs_batched_p16d4pool2_le8_s8_20261002T224500Z.json']
FIGURES=['report/figures/local_score_bound_evidence_20261002T235800Z_geometry.png',
         'report/figures/local_score_bound_evidence_20261002T235800Z_trap.png']


def load(read):
    data={key:read(name) for key,name in FILES.items()};data['fine']=[read('dvs_native/'+name) for name in FINE]
    for result in [*data['fine'],*(data[k] for k in FILES)]:
        if result['status']!='completed':raise ValueError('Completed score-bound evidence required')
        for name,digest in result.get('source_sha256',{}).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed source '+name)
        for parent in result.get('parents',[]):
            if hashlib.sha256((ROOT/parent['result']).read_bytes()).hexdigest()!=parent['result_sha256']:
                raise ValueError('Changed geometry/exposure parent')
    return data


def render_figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=[r for r in data['bridge']['rows'] if r['original_clamped']]
    fig,ax=plt.subplots(figsize=(8.2,3.4),layout='constrained');x=np.arange(len(rows))
    ax.bar(x-.18,[r['original_hard']['information_eigenvalue_ratio'] for r in rows],.36,label='Hard bound',color='#8a8984')
    ax.bar(x+.18,[r['bridge']['information_eigenvalue_ratio'] for r in rows],.36,label='Bounded bridge .1',color='#2a78d6')
    ax.set_yscale('log');ax.set_ylim(1e-22,1);ax.set_ylabel('Small / large information eigenvalue')
    ax.set_xticks(x,[f"FIT{r['fit_index']}\ne{r['site'][0]} / L{r['site'][1]}" for r in rows]);ax.legend(loc='upper left')
    ax.set_title('Same conditioned native histories: sensitivity restored, rare-choice conditioning remains')
    fig.savefig(ROOT/FIGURES[0],dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,2.8),layout='constrained')
    for trace in data['exposure']['gradient_trap']['traces']:
        ax.plot([r['step'] for r in trace['curve']],[r['risk'] for r in trace['curve']],
            label='Hard bound' if trace['alpha']==0 else 'Bounded bridge .1',color='#8a8984' if trace['alpha']==0 else '#2a78d6')
    ax.set_xlabel('Synthetic exact-gradient updates');ax.set_ylabel('Conditional utility + latency risk');ax.legend()
    ax.set_title('Two saturated emitters: a mathematical optimization witness, not DVS accuracy')
    fig.savefig(ROOT/FIGURES[1],dpi=170);plt.close(fig)


def pages(data):
    import numpy as np
    result=[];rows=[]
    for seed in (6,7):
        for encoder in ('initial','fixed_pass4'):
            group=[r for r in data['geometry']['rows'] if r['seed']==seed and r['encoder']==encoder]
            ratios=[r['full']['information_eigenvalue_ratio'] for r in group]
            rows.append([str(seed),encoder,f"{np.median([abs(r['full']['cosine']) for r in group]):.4f}",
                f'{min(ratios):.2e}',f'{max(ratios):.2e}',str(sum(any(abs(s)>=12 for s in r['scores']) for r in group))+'/8'])
    result.append([('h1','Appendix B. Native conditional choice-clock parameter geometry'),
        ('table',(['Seed','Encoder','Median |cos|','Min info ratio','Max info ratio','Capped sites'],rows,[16,38,30,31,31,27])),
        ('p','At a fixed observed earlier winner/time history, differentiate entering '
         'score contrast u=grad(s0-s1) and common log-rate v=grad(logsumexp(s)). '
         'The local winner/time metric is pi0*pi1*u*u^T+v*v^T; its two eigenvalues '
         'come from a2x2 Gram matrix, avoiding a dense15523-coordinate matrix. '
         'Past physical times are held fixed for conditional likelihood geometry; '
         'real earlier message/memory maps and sparse writes still receive derivatives.'),
        ('p','Four prespecified sites, two unused FIT inputs258/981, four frozen initial/ '
         'fixed-four-pass encoders:32 Jacobian pairs/64 VJPs. Native primal outputs, '
         'ALLstate and end-RNG reproduce exactly. Four-block direction arrays retain '
         'clock-bias, key/query, message/state/transport and classifier contributions. '
         'Directional finite differences use eps .002/.001, relative2%/absolute.005 '
         'tolerance; every error is retained rather than reported as exact algebra.'),
        ('p','Five of sixteen selected trained cases pin a score at12 and leave local '
         'choice/clock sensitivities collinear. Other confident trained cases are poorly '
         'conditioned without clipping. Negative cosine alone is not harmful credit: '
         'no target utility or proposed update is evaluated. Single-site bias control '
         'also does not imply independent control over all shared histories.'),
        ('small','Theory103;36.536s/348,112KiB. Exact originals/weights restored, no '
         'optimizer/DEV/test or replaced time-learning path. Original core fit2.285696GF '
         'each /2.232125MF per1024 presentations retained. Conditional diagnostic is '
         'not full expected-risk credit, parameter covariance or advantage. Full audit '
         'FLOPs, traffic and energy unknown, not zero.')])
    rows=[]
    for r in data['bridge']['rows']:
        if r['original_clamped']:rows.append([str(r['fit_index']),f"e{r['site'][0]}/L{r['site'][1]}",
            f"{max(r['raw_scores']):.4f}",f"{min(r['raw_slope']):.5f}",
            f"{r['original_hard']['information_eigenvalue_ratio']:.2e}",f"{r['bridge']['information_eigenvalue_ratio']:.2e}"])
    result.append([('h1','Appendix B. Bounded emitter bridge restores the missing direction'),
        ('table',(['FIT index','Site/head0','Raw max','Min slope','Hard info ratio','Bridge info ratio'],rows,[24,30,26,26,33,34])),
        ('figure',(FIGURES[0],173)),
        ('p','Fixed alpha.1 bridge: .9*clip(raw,-12,12)+.1*raw/sqrt(1+(raw/12)^2). '
         'It remains bounded, odd and monotone, with positive mathematical sensitivity '
         'beyond the hard clamp. On these same conditioned histories it restores the '
         'missing local direction. Ratios remain small because choices are rare; this '
         'is not measured noise reduction, useful-route credit or model-quality improvement. '
         'The hard-map near-zero ratios are numerical roundoff around theoretical rank1, '
         'not meaningful residual information.'),
        ('small','Theory104; five primitive contracts and eight native comparisons pass '
         '37.961s/371,992KiB. Alpha0 outputs/state/RNG and all parameter gradients exactly '
         'nest the old model. Positive complete native inference retains coupled clocks/ '
         'routes/real writes and target invariance; training refuses. Each prefix scores '
         '168 logical keys/writes84; diagnostic hooks RECOMPUTE168 extra dot products, paid '
         'work. Live state576..720bytes,8available units. Prior fits paid; audit FLOPs '
         'unknown. No automatic fit, free scoring or hardware/quality advantage.')])
    rows=[]
    for r in data['exposure']['rows']:
        a=r['summary']['all_sites'];q=r['summary']['query'];rows.append([str(r['seed']),r['encoder'],
            'Cap proxy' if r['role']=='cap_proxy' else 'Raw-confirmed',f"{100*a['any_fraction']:.2f}",
            f"{100*a['expected_capped_winner_probability']:.2f}",f"{100*q['any_fraction']:.2f}"])
    result.append([('h1','Appendix B. Score-bound exposure and an exact gradient trap'),
        ('table',(['Seed','Encoder','Evidence','Capped races %','Expected capped winner %','Query races %'],rows,[14,34,33,30,36,26])),
        ('figure',(FIGURES[1],173)),
        ('p','The43,008-race eight-history profile gives cap-occupancy proxies; the5,376 '
         'one-history score-pair decomposition independently confirms raw saturation '
         'using a2e-5 margin. Cohorts/noise scopes stay separate. No sampled race has '
         'both candidates saturated: the two-bound trap below is a mathematical witness, '
         'not an observed native condition. Late query cap exposure is stronger than all sites.'),
        ('small','Theory105 exact conditional utility+bounded computational-latency risk: '
         'two raw emitters13/13,400 updates/step1. Hard remains .600003; bridge .600004 '
         'to.103999. A finite hard-map hop improves risk, proving an optimization flat '
         'cell rather than a representational impossibility. No stochastic estimator '
         'noise/native fitting/quality gate;800 scalar updates paid. Saved-array study '
         '.332s/245,116KiB; original fits retained, total audit FLOPs unknown.')])
    rows=[]
    for r in data['fine']:
        a=r['args'];w=r['work'];rows.append([str(a['seed']),f"L{a['depth']}/"+('all' if a['route_races']==0 else 'k8'),
            f"{100*r['final']['accuracy']:.2f}",f"{r['final']['nll']:.6f}",f"{w['whole_fit_unit_special_flops_estimate']/1e9:.3f}",
            f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.3f}",f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
    result.append([('h1','Appendix B. Completed fine-packet replay confirmations'),
        ('table',(['Seed','Credit/depth','Dev acc %','Dev NLL','Whole fit GF','Fit MF/presentation','Infer MF/prefix'],rows,[14,32,23,25,27,28,24])),
        ('p','Full984 FIT/192 DEV,20 observed packets plus query,8passes/7872 fitting '
         'presentations,U16/lr.003. All-race depth2 now has three completed seeds; '
         'depth4 sampled8 has seeds7/8. Original/factorized controls remain on the preceding '
         'replay ledger; missing matched deeper controls are not filled with predictions. '
         'Seed8 depth2 reaches66.667%/.941821; preserve it as a completed result with '
         'its1040.397GF fitting cost, not an isolated practical advantage.'),
        ('p','Depth4 sampled8 is worse than same-seed depth2 on development loss in '
         'these runs. This is a restricted eight-site credit variant, not a test of '
         'all-race deeper credit or proof that depth cannot learn. The owner has queued '
         'matched depth4 all-race, depth6 sampled and growth-by-nesting comparisons.'),
        ('small','Every column uses consistent units/denominators; core whole-fit and '
         'per-presentation work include replay, backward and Adam. Inference mean first11 '
         'DEV prefixes,2FLOPs/MAC plus unit specials. L2:8available units/168key scores/ '
         '84writes per prefix;L4:16/336/168. FIT diagnostic scores use only32 fitting '
         'examples, not whole-FIT risk. DEV epoch selection and one-seed comparisons '
         'remain exploratory. Raw preprocessing, RNG, traffic and energy separate.')])
    rows=[]
    for key,label in [('teacher','Original teacher'),('factorized','Factorized control'),('replay','All40-race replay')]:
        r=data[key];w=r['work'];rows.append([label,f"{100*r['final']['accuracy']:.2f}",f"{r['final']['nll']:.6f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.6f}",f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
    result.append([('h1','Appendix B. AWS deeper all-race pilot: matched gates fail'),
        ('table',(['Coarse L4 seed7','Dev acc %','Dev NLL','Whole fit GF','Fit MF/presentation','Infer MF/prefix'],rows,[43,22,27,27,28,26])),
        ('p','Coarse4/.25-clock,p16/L4/H2/pool2,256FIT/192DEV,fourpasses/1024 '
         'presentations,U16/lr.003. Every-parameter sequential/forked replay contracts '
         'and three actual recovery/accounting smokes precede these pilots. Full40 '
         'race sites/80shadow lanes preserve factual first time and actual forced writes. '
         'Teacher/factorized/replay share data and observation protocol; no weight decay '
         'or input noise is mixed into this comparison.'),
        ('p','Replay NLL is worse by.032366 versus teacher and.005835 versus factorized, '
         'at27.861x/26.763x counted fitting work. Both fixed gates fail, confirmation '
         'is absent and no larger fit is nominated. This is a first small coarse/deep '
         'pilot; it does not replace the independently owned full-data fine-depth tests.'),
        ('small','Completed summary234200Z,108.202s total campaign. Same common-unit '
         'denominator in every fitting column.16available receivers,8writes/event, '
         '40factual races/prefix. Full replay/optimizer paid despite parallel wall '
         'execution. Inference first11 DEV prefixes; raw input preprocessing/traffic/ '
         'RNG/energy separate. No official test, comparable-quality work supremacy or '
         'claim that all deep counterfactual learning must fail.')])
    rows=[]
    for r in data['priority']['results']:
        rows.append([str(r['seed']),f"{r['heldout_score_variance_ratio']:.6f}",
            *[f"{c['ratio']:.6f}" for c in r['parameter_cases']],str(r['parameter_gate_passed'])])
    result.append([('h1','Appendix B. Parameter-targeted allocation retains a scoped positive gain'),
        ('table',(['Seed','Score variance/k4','Parameter FIT96 ratio','Parameter FIT97 ratio','Each-case gate'],rows,[18,41,40,40,34])),
        ('p','Frozen priority predictor trains on true shared-parameter route norms on '
         'FIT0..31 and confirms on new FIT96..127. Exact distinct weighted sampling '
         'and inclusion correction remain. Both aggregate score ratios pass; one seed8 '
         'parameter case fails. Combined nomination stays FAIL. The two-case weighted '
         'parameter aggregates .516746/.658755 support48.3%/34.1% reductions and are '
         'preserved as diagnostics, not replacements of the every-case gate.'),
        ('p','This is materially better allocation evidence than isolated score-space '
         'improvement, but four parameter examples cannot establish robustness or quality. '
         'No unchanged reduced-replay fit follows. Producer FIT labels were already seen; '
         'only priority supervision is held out. Every expensive training utility remains paid.'),
        ('small','11.439s/531,560KiB;1280 training plus80confirmation VJPs,2560diagnostic '
         'shadow lanes, saved predictors/proposals. Whole fitting/diagnostic FLOPs unknown, '
         'not zero. No inference change, DEV/test quality or6-versus8 execution-lane '
         'work advantage. Main deeper replay/growth remains independently owned.')])
    rows=[]
    for key,label in [('untied','Untied pool2'),('tied2','Shared pool2'),('tied8','Shared pool8')]:
        r=data[key];w=r['work'];rows.append([label,str(r['parameters']),str(w['native_available_receivers']),
            f"{100*r['final']['accuracy']:.2f}",f"{r['final']['nll']:.6f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.6f}",f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
    result.append([('h1','Appendix B. AWS private state and shared maps: pilot gates fail'),
        ('table',(['Variant','Params','Available','Dev acc %','Dev NLL','Fit GF','Fit MF/target','Infer MF'],rows,[33,21,19,21,25,22,17,17])),
        ('p','Same256FIT/192DEV,fourpasses/1024presentations,seed6,coarse4/.25-clock, '
         'p16/L2/H2. Shared input/output/gate maps retain private keys, clock biases, '
         'timescales and receiver memories. Delays perform computation, hard addressed '
         'races select four real writes/event and separate keys/values remain. '
         'Pool8 adds32available receivers while keeping four selected writes.'),
        ('p','Both fixed quality/resource nominations FAIL: tied2 loses.001754NLL '
         'and2.604accuracy points; tied8 improves NLL.041447 but misses.05 and loses '
         '1.042points. Lower parameters and positive tied8 loss improvement remain '
         'supported. Pool growth also changes total hazard and per-unit exposure, '
         'so this is not isolated proof of capacity beyond scored work. No confirmation/full fit.'),
        ('small','Every-parameter/state/batch and actual Adam/cursor/RNG/operator '
         'contracts pass before both learning smokes and pilots. Historical failed '
         'coalesce-adapter contract remains beside corrected admission. Campaign56.762s; '
         'all candidate keys/proposals/losing credit/optimizer work paid, same fitting '
         'denominator. Inference first11prefixes/2FLOPs perMAC plus unit specials; '
         'raw preprocessing, physical traffic/RNG/energy separate. Deeper replay '
         'owns the next integrated comparison; no unchanged shallow expansion.')])
    rows=[]
    for r in data['query']['rows']:
        seed=r['seed'];work=r['whole_prefix_arithmetic_plus_special_ops']
        original=work['baseline']/1e6;query=work['query_only']/1e6;fit=r['parent_fit_gflops']
        rows.append([str(seed),f"{100*r['accuracy']:.2f}",f"{r['quality']:.6f}",f'{fit:.6f}',f'{fit*1000/7872:.6f}',f'{original:.6f}',f'{query:.6f}'])
    result.append([('h1','Appendix B. Query-only native admission: exact arithmetic saving'),
        ('table',(['Seed','Dev acc %','Dev NLL','Whole fit GF','Fit MF/presentation','Old infer MF','Query infer MF'],rows,[15,22,25,26,30,27,28])),
        ('p','Three saved coarse4/.25-clock native producers,984FIT/eightpasses/7872 '
         'fitting presentations and192DEV. All576 paired prefix logits and ALLstate '
         'checks are bitwise identical. Only four unused intermediate affine classifiers '
         'are removed; clocks, races, keys, messages, persistent writes and terminal '
         'query remain. Original weights/fitting work and predictions are retained.'),
        ('p','2860 counted operations per prefix,2.07%, are saved under full ATen '
         'coverage. Three counterbalanced wall repeats show the Python wrapper about2% '
         'SLOWER. Arithmetic saving is supported; practical speed/supremacy is not. '
         'This explicit-query inference-only helper cannot replace ongoing-label or '
         'silence-supervised streams without a separate contract.'),
        ('small','65.025s/465,880KiB; target/repeat/query/parameter/training-rejection '
         'checks pass.8available units,4writes/event,all core scoring still paid. '
         'Same fit denominators/units and first11-prefix inference convention for both '
         'implementations.2FLOPs/MAC plus unit specials; raw preprocessing, traffic, '
         'RNG and measured energy remain separate. Verbatim former manual section '
         'preserved in appendices/aws_query_only_history_20261002T233000Z.md.')])
    return result
