"""Completed clock-safe calibration and cross-host allocation evidence."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES={
 'support':'diagnostics/local_dvs_race_support_20261002T225200Z.json',
 'keys':'diagnostics/local_dvs_key_score_decomposition_20261002T230300Z.json',
 'contracts':'diagnostics/local_clock_preserving_contracts_20261002T231200Z.json',
 'temperature':'diagnostics/local_dvs_clock_preserving_calibration_20261002T231600Z.json',
 'initial_critic':'diagnostics/aws_replay_variance_20261002T230000Z.json',
 'trained_critic':'diagnostics/aws_trained_replay_variance_20261002T230200Z.json',
 'signed_critic':'diagnostics/aws_signed_replay_variance_20261002T230400Z.json',
 'shrinkage':'diagnostics/aws_replay_calibration_20261002T231000Z.json',
 'ordinal':'diagnostics/aws_replay_importance_20261002T232200Z.json',
 'magnitude':'diagnostics/aws_learned_replay_priority_20261002T232400Z.json',
 'distinct':'diagnostics/aws_distinct_replay_priority_20261002T232600Z.json',
 'compact':'diagnostics/aws_compact_context_20261002T232000Z.json',
 'factorized':'dvs_native/curie_dvs_batched_p16d2pool2_factorized_s7_20261002T224500Z.json',
 'replay':'dvs_native/curie_dvs_batched_p16d2pool2_leall_s7_20261002T224500Z.json',
 'original':'dvs_native/curie_dvs_clock_p16d2pool2_s7_20261002T183000Z.json'}


def load(read):
    data={key:read(name) for key,name in FILES.items()}
    for key,r in data.items():
        if r['status']!='completed':raise ValueError('Completed route calibration evidence required: '+key)
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed route numerical source: '+name)
    return data


def pages(d):
    pages=[];rows=[]
    for r in d['support']['rows']:
        a=r['all_sites'];receiver=a['receivers']
        rows.append([str(r['seed']),r['encoder'],f"{a['mean_normalized_entropy']:.3f}",
            f"{100*a['top_probability_99_fraction']:.2f}",f"{receiver['fixed_1ms']['expected_mean']:.3f}",
            f"{receiver['fixed_3ms']['expected_mean']:.3f}",f"{receiver['relative_c4']['expected_mean']:.3f}"])
    pages.append([('h1','Appendix B. Race-scaled reception: available arrival support'),
        ('table',(['Seed','Encoder','Entropy','Top p>=.99 %','1ms heard','3ms heard','c4 heard'],rows,[14,37,23,31,23,23,22])),
        ('p','A relative raw deadline (1+c)T maps through the actual bounded temporal delay. '
         'Its heard set is unchanged under a common entering-score shift: all raw clocks scale '
         'together. Physical computational times still change. The exact expected receiver '
         'count is 1+sum_w pi_w sum_(j!=w) c*pi_j/(1+c*pi_j). The maximum additional '
         'physical waiting is .010*(sqrt(1+c)-1)/(sqrt(1+c)+1):1.716ms at c1,3.820ms at c4. '
         'The local11ms delay bound remains. This fixes speed-dependent membership, not utility.'),
        ('p','43,008 actual native races across all84 sites,16 producer-unseen FIT inputs, '
         'eight noise histories and four encoders. The earlier event19/layer0/head0 site is '
         'more concentrated than the whole model: top probability>=.99 in71.88%/57.03% '
         'of trained races. First events have the most alternative support. A previous '
         'one-history1ms lack of reception is not proof that all alternatives are absent.'),
        ('p','Trained relative c4 mean additional waiting1.406/1.675ms. These are frozen '
         'arrival profiles, with no changed deliveries/writes, new loss, optimizer, DEV '
         'or test evaluation. No learned window or quality nomination follows from counts.'),
        ('small','Theory99: six numerical contracts and original logit/state checks pass. '
         '59.553s/335,536KiB; exact candidate/clock arrays and checksums retained. Theory100 '
         'independently verifies collector end-RNG preservation. Prior trained core fits '
         '2.285696GF each remain paid. Whole audit FLOPs, traffic and energy unknown, not zero.')])
    rows=[]
    for r in d['keys']['rows']:
        a=r['all_sites'];rows.append([str(r['seed']),r['encoder'],f"{a['mean_absolute_static_gap']:.3f}",
            f"{a['mean_absolute_dynamic_gap']:.3f}",f"{a['static_only']['mean_normalized_entropy']:.3f}",
            f"{a['factual']['mean_normalized_entropy']:.3f}"])
    pages.append([('h1','Appendix B. Memory-conditioned keys explain routing confidence'),
        ('table',(['Seed','Encoder','Static gap','Memory gap','Static entropy','Full entropy'],rows,[16,38,29,29,30,31])),
        ('p','Native entering score = query dot static key /sqrt(payload) + clock bias '
         '+ query dot key_read(persistent memory)/sqrt(payload), then the existing clamp. '
         'Exact hooks reconstruct all candidate scores within1.91e-6. Trained memory gaps '
         'exceed static gaps in93.75%/92.86% of races. First-event memory is zero and '
         'routing is nearly uniform. Persistent-memory reads are the observed confidence source.'),
        ('p','Component-only entropies hold the current query fixed; they are algebraic '
         'local counterfactuals, not predictions from a modified model history. Large memory '
         'scores may express useful specialization or brittle commitment. This decomposition '
         'does not establish harmful confidence or a representation-learning regression.'),
        ('p','Conventional router z-loss penalizes logsumexp(scores) for numerical stability '
         '(ST-MoE, Zoph et al.,2022). Here that quantity is log total rate, so it controls '
         'actual first-arrival time. Subtracting it forces total rate1 and changes computation. '
         'Choice uncertainty and common speed must be distinguished before importing normalization.'),
        ('small','Theory100,5,376 race score pairs /10,752 candidate scalars; four frozen '
         'encoders x16 unused FIT prefixes xone noise history. Eight exact logit/state/end-RNG '
         'contracts;32.992s/328,876KiB. No loss, fitting, DEV/test or causal harm claim. '
         'All weights fixed; prior fits and diagnostic work retained. Full audit FLOPs unknown.')])
    for seed in (6,7):
        rows=[]
        for r in d['temperature']['rows']:
            if r['seed']!=seed:continue
            rows.append([r['encoder'],f"{r['temperature']:.0f}",r['scope'],f"{r['nll']:.6f}",
                f"{100*r['accuracy']:.2f}",f"{r['representative_first_prefix_unit_special_flops_estimate']/1e6:.6f}"])
        gate=next(g for g in d['temperature']['gates'] if g['seed']==seed)
        pages.append([('h1',f'Appendix B. Clock-preserving route calibration, seed{seed}'),
            ('table',(['Encoder','Temp.','Scope','FIT NLL','FIT accuracy %','Prefix MF est.'],rows,[37,19,24,33,31,29])),
            ('p','At each entering state, keep the original first raw time T and total rate '
             'Lambda. A losing exponential residual supplies a uniform independent of T; '
             'conditional inverse CDF changes the categorical winner to softmax(score/temperature). '
             'No extra random draws, losing value deliveries or receiver updates. The actual '
             'selected receiver changes, so subsequent memories and later clocks may change.'),
            ('p','Separate16 unused FIT examples, four prespecified noisy histories; every '
             'initial/fixed-four-pass model and setting retained. This is mean per-history '
             'loss/accuracy, not ensemble inference. No optimizer, new decoder, DEV or test. '
             f"Temperature2/all NLL gain {gate['nll_gain']:.6f}, accuracy change "
             f"{100*gate['accuracy_gain']:.3f} points; fixed both-seed smoke gate "
             +('PASSES.' if d['temperature']['integrated_smoke_nomination_passed'] else 'FAILS.')),
            ('small','Theory101. Distribution screens and exact tau1 output/state/all-gradient/RNG '
             'contracts pass29.010s/354,328KiB. Positive-temperature training is refused until '
             'correct choice/common-clock gradients and optimizer/recovery/accounting exist. '
             'Each prefix scores168 keys, writes84 receivers, has8 available units/720state bytes. '
             'Original trained fit2.285696GF/2.232125MF per presentation; table work is one '
             'traced inference prefix including calibration, not full-audit cost or benchmark advantage.')])
    rows=[]
    for key,label in [('initial_critic','Initial fine norm'),('trained_critic','Trained coarse norm'),('signed_critic','Signed/label critic')]:
        for r in d[key]['results']:rows.append([label,str(r['seed']),f"{r['critic_vs_plain_k4_mse_ratio']['2']:.6f}",'FAIL'])
    for r in d['shrinkage']['results']:
        rows.append(['Training-only shrinkage',str(r['seed']),f"{r['score_results']['training_shrinkage']['heldout_score_mse_ratios']['2']:.6f}",'FAIL'])
    pages.append([('h1','Appendix B. AWS frozen replay critics and parameter calibration'),
        ('table',(['Frozen conditional score study','Seed','k2/plain k4 MSE','Gate'],rows,[76,18,47,32])),
        ('p','Correct first-time-preserving actual-write replay and critics frozen before '
         'subset sampling retain unbiased credit. Every reduced-replay nomination fails. '
         'The initial fine screen uses84 races; trained coarse uses20. Signed-message and '
         'label-aware critic inputs are detached learning features, with32 critic-training '
         'and32 heldout FIT prefixes. Sixteen original numerical contracts precede screens.'),
        ('p','Score-coordinate MSE is insufficient for shared parameters: cross-site '
         'covariance contributes. Actual training-only calibrated parameter variance ratios '
         '2.383887/2.189517 in seed7 and37.462736/1.674462 in seed8 also fail. Exhaustive '
         'finite-population and cached factual-probability contracts pass.160 new VJPs paid.'),
        ('small','All outcomes, targets, critics and original costs retained. No new producer '
         'fit, DEV/test quality or reduced-work learning claim. Diagnostic FLOPs unmeasured, '
         'not zero. This rejects unchanged critic reduction rather than counterfactual credit '
         'in general. AWS_REPLAY_VARIANCE_FINDINGS and AWS_REPLAY_CALIBRATION_FINDINGS carry lineage.')])
    rows=[]
    for r in d['ordinal']['rows']:rows.append(['Ordinal k2',str(r['seed']),f"{r['heldout_importance_k_vs_uniform_without_replacement_k4']['2']:.6f}",'Not measured'])
    for r in d['magnitude']['rows']:rows.append(['Feature k2',str(r['seed']),f"{r['heldout_k2_vs_plaink4_ratio']:.6f}",'Not measured'])
    for r in d['distinct']['results']:rows.append(['Weighted distinct k3',str(r['seed']),f"{r['heldout_score_variance_ratio']:.6f}",'/'.join(f"{c['ratio']:.6f}" for c in r['parameter_cases'])])
    pages.append([('h1','Appendix B. Replay allocation: fresh score gains, parameter failure'),
        ('table',(['Proposal','Seed','Score variance/k4','Two parameter ratios'],rows,[51,18,43,61])),
        ('p','Ordinal and feature-conditioned k2 proposals sample WITH replacement, '
         'using exact1/(kp) correction and a positive floor. Baseline uniform k4 is WITHOUT '
         'replacement; uniform k2 with replacement is2.375x. Feature-conditioned priorities '
         'improve over that reference but fail reduced-budget gates. Diagnostic exact-return '
         'oracle ratios .631115/.519480 need all costly utilities and are not deployable.'),
        ('p','The frozen feature predictor then selects three distinct weighted sites on '
         'fresh FIT64..95, with exact marginal/pair inclusion probabilities and1/inclusion '
         'correction. Conditional score variance falls23.19%/30.62% versus uniform four; '
         'uniform three would increase it41.67%. Preserve this positive allocation result. '
         'The combined gate fails on actual parameter cases shown above.'),
        ('small','Exact expectation/shared-vector variance/uniform nesting contracts pass. '
         '2560 actual diagnostic lanes and80 confirmation VJPs; distinct screen3.578s/ '
         '518,380KiB. Inclusion computation .742/.747ms per prefix, plus tree/discovery/VJP '
         'costs. Proposed6 versus8 execution lanes is projected, not measured reduced-work '
         'training. Full diagnostic FLOPs unknown; no quality/supremacy claim or unchanged fit.')])
    rows=[]
    for r in d['compact']['rows']:
        name=r['name'];rows.append([name,f"{100*r['final']['accuracy']:.2f}",f"{r['final']['nll']:.6f}",'Unknown','Unknown'])
    pages.append([('h1','Appendix B. AWS compact context head: practical controls remain stronger'),
        ('table',(['Predictor','Dev accuracy %','Dev NLL','Combined fit GF','Fit MF/target'],rows,[49,29,29,33,33])),
        ('p','33-anchor dense local RBF readout uses all distances/basis outputs on frozen '
         'native query contexts, plus mandatory raw4/raw20 prototype controls.984 FIT/ '
         '192 DEV, three native trained/initial pairs, fitting-user three-fold regularization '
         'selection.72 fold fits,8 refits and352 class-specific KMeans fits are paid. '
         'Six Torch and48 query/state contracts pass before decoder fitting.'),
        ('p','Both fixed stage/storage gates FAIL. Learned-over-initial mean NLL gain '
         '.277040 is supported. Native standalone exports ~115KB versus raw4 40.966KB '
         'and raw20 184.334KB; fine-control byte savings do not overcome stronger coarse '
         'quality/storage. Raw4 SVC77.604%/.686661 and earlier valid leaders remain.'),
        ('small','Whole/per-target combined fitting and inference FLOPs unknown for EVERY '
         'arm because solver/replay work is unmeasured. Trained native parent fit4.218015GF/ '
         '.535825MF per7872 presentations retained, not total fit. Initial/raw producer fit0 '
         'is not zero solver work. Study23.296s/690,896KiB. Portable prediction/roundtrip '
         'and three sequential latency repeats pass; exports include normalization/core/ '
         'anchors/decoder metadata. This is a dense readout diagnostic, not sparse race attention.')])
    rows=[]
    for key,label in [('original','Original route teacher'),('factorized','Factorized clock only'),('replay','All-race exact replay')]:
        r=d[key];w=r['work'];rows.append([label,f"{100*r['final']['accuracy']:.2f}",f"{r['final']['nll']:.6f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.3f}",f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.3f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
    pages.append([('h1','Appendix B. Integrated exact replay: first seed does not improve quality'),
        ('table',(['Native seed7 method','Dev acc %','Dev NLL','Whole fit GF','Fit MF/presentation','Infer MF/prefix'],rows,[43,22,24,28,31,25])),
        ('p','Same984 fitting/192 development inputs, eight passes/7872 fitting presentations, '
         'p16/layer2/head2/pool2. Every method retains computational delays, temporal races, '
         'persistent sparse writes and separate keys/values. All-race replay forces alternatives '
         'at the factual first time with real writes; vectorized shadow histories include '
         'unrealized downstream effects. Factorized-clock-only removes the original local route teacher.'),
        ('p','All-race replay58.333%/1.141794 versus factorized56.771%/1.116189 and '
         'original57.812%/1.105326 is not a quality improvement under the prespecified '
         'loss criterion. Approximately1040GF counted fit versus18.24GF factorized '
         'is paid despite batched CPU execution. A large fit/dev gap motivates independent '
         'generalization diagnostics; it does not prove routing credit cannot help.'),
        ('small','Whole fitting and per-presentation fitting columns share units and7872 '
         'denominator. Inference mean first11 DEV prefixes;2FLOPs/MAC plus unit specials. '
         'Eight available receivers,168 scored keys/84 writes per21-event prefix. '
         'Candidate discovery, replay, backward and Adam charged; preprocessing, traffic, '
         'RNG and energy separate. Source/class/estimator variants and one-seed exploratory '
         'scope retained. Other host owns remaining replay/regularization comparisons.')])
    return pages
