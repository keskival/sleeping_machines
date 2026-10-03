"""Conditional site-noise separation and actual existing-teacher fitting work."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1];BASE=runpy.run_path(str(ROOT/'report/trained_content_variance_evidence.py'))
SITE='diagnostics/local_trained_route_site_noise_20261003T054100Z.json'
WORK='diagnostics/local_trained_route_coverage_work_20261003T054700Z.json'
def verify(r):
    if r['status']!='completed':raise ValueError('Completed route evidence required')
    for n,h in r['source_sha256'].items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise ValueError('Changed trained-route source '+n)
    for name in ('parent','snapshot'):
        if hashlib.sha256((ROOT/r[name]).read_bytes()).hexdigest()!=r[name+'_sha256']:raise ValueError('Changed route evidence '+name)
def load(read):
    d=dict(prior=BASE['load'](read),site=read(SITE),work=read(WORK));verify(d['site']);verify(d['work'])
    if d['site']['contracts_passed']!=5 or d['work']['contracts_passed']!=6:raise ValueError('All route contracts required')
    if hashlib.sha256((ROOT/d['site']['projections']).read_bytes()).hexdigest()!=d['site']['projections_sha256']:raise ValueError('Changed saved projections')
    return d
def pages(d):
    pages=BASE['pages'](d['prior']);r=d['site'];rows=[]
    for s in r['summaries']:rows.append([str(s['k']),f"{s['mean_site_trace_estimate']:.6f}",f"{s['between_history_full_mean_sample_trace']:.6f}",f"{s['total_trace_estimate']:.6f}",f"{100*s['site_share_descriptive_ratio']:.2f}"])
    pages.append([('h1','Appendix B. Separate site sampling from native race history'),
        ('table',(['Sites per episode','Mean site trace','History mean trace','Total raw trace','Site share %'],rows,[36,36,36,36,30])),
        ('p','SAME source-bound12-step trained p4/D4/H2/pool2 and '
         'original FIT0..3/B4, FIRST FOUR prior noise histories, with '
         'represented weights promoted to DOUBLE. Each history pays '
         'complete detached returns for ALL168 races per episode; '
         'the original native full-site teacher is the conditional mean. '
         'Uniform without-replacement site sampling is the only random '
         'axis in each conditional covariance.'),
        ('p','At fixed E, v_jr=pi0*pi1*(Q0-Q1)/B times the COMPLETE '
         'parameter Jacobian of score0-score1. Site covariance trace '
         'S_k=sum_j R_j*(R_j-k)/(k*(R_j-1))*sum_r '
         '||v_jr-mean_r(v_jr)||^2. Full-site sampling has zero such '
         'variance. Across histories, mu(E)=factual+all-site credit; '
         'Var(G)=E[Var_sites(G|E)]+Var(mu(E)). Native computational '
         'clock/content coupling remains in the Jacobian.'),
        ('p','Here about95% of estimated k1 total variance is site '
         'sampling. k8 reduces estimated total raw trace by about84%. '
         'These are FOUR-history descriptive estimates, not an attribution '
         'of all deep training failure.32 Rademacher parameter signs per '
         'history estimate conditional traces without dimension scaling; '
         'all probes saved, empirical SE descriptive, worst-case relative '
         'RMS25%, no guaranteed interval or actual Adam variance.'),
        ('p','Same TRAINED model short2/1-event contract matches EVERY '
         'route VJP under three projection signs, full original teacher '
         'and explicit subset mean/covariance. Maximum error4.58e-16. '
         'All128 main projected sums match actual full gradients. '
         'Float32 factual CE reproduces prior draws exactly; all2688 '
         'factual winners agree with double, max logit error2.85e-7. '
         'This does not prove forced-branch cross-precision equivalence.'),
        ('small',f"Theory134; {r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         'Main5376 full-return lanes/112896 events and128 projections; '
         'tiny equivalence evaluates returns TWICE,96 lanes/160 events. '
         'No optimizer step; total FLOPs/traffic/energy unknown, not zero. '
         'Stored gradients/returns/signs/projections retain scope; no policy '
         'or benchmark promotion from raw variance alone.')])
    r=d['work'];rows=[]
    for a in r['arms']:rows.append([str(a['k']) if a['k'] else 'No choice',f"{a['whole_step_unit_special_flops']/1e9:.9f}",f"{a['fit_unit_special_flops_per_target']/1e6:.6f}",f"{a['inference_unit_special_flops_per_target']/1e6:.6f}",str(a['stats']['route_replays'])])
    comparison=[[str(x['k']),f"{x['raw_total_variance_ratio_estimate']:.6f}",f"{x['actual_step_work_ratio']:.6f}",f"{x['variance_times_work_ratio']:.6f}"] for x in r['comparisons']]
    pages.append([('h1','Appendix B. Existing route coverage: actual fitting cost versus variance'),
        ('table',(['Sites','Actual step GF','Fit MF/presentation','Infer MF/target','Shadow lanes'],rows,[22,41,41,41,29])),
        ('table',(['Sites','Raw variance ratio','Actual work ratio','Variance x work'],comparison,[28,48,48,50])),
        ('p','FIVE actual source-bound EXISTING BL fitting windows, '
         'four targets/one update each, same recovered trained weights '
         'AND historical moments. Original driver sampled1/8/32 '
         'and all168 sites; unchanged factorized no-choice control. '
         'FIRST noise seed fixed before work, original sampler RNG. '
         'Every forward/loss/return/backward/normalization/clip1/Adam.003 '
         'operation traced with full formula coverage. Traced/untraced '
         'next weights AND moments agree exactly for every configuration; '
         'serialized full recovery also agrees. All updates discarded.'),
        ('p','All five use2379 parameters,16 available private receivers, '
         '168 selected updates/336 key scores per target. Losing full '
         'returns still compute candidate keys and values; optimizer work '
         'is charged. Native inference uses the same12-target boundary. '
         'Whole-step and per-presentation work share units/denominators '
         'across ALL rows. No choice changes the mean teacher and is '
         'not a k0 point on the same route-variance curve.'),
        ('p','With equal R, V(k)=a+b/k; approximately affine C(k)=C0+c*k. '
         'The interior variance-work optimum is sqrt(C0*b/(c*a)) when '
         'a>0, bounded by1..R. Here it is about6.09 sites; k8 is the best '
         'MEASURED heuristic point. This is fixed-state estimation math, '
         'not a validated adaptive policy or sustained learning guarantee.'),
        ('p','Variance-times-work pairs four-history DOUBLE projected '
         'raw variance with one-window FLOAT32 actual work. It is an '
         'optimization HEURISTIC, not an actual Adam-variance, convergence '
         'or quality guarantee. Broader site support buys less sampling '
         'noise but costs additional full returns; the remaining race '
         'history floor prevents unlimited benefit. No larger fit is '
         'selected from local FIT predictions.'),
        ('p','Eleven discarded updates: five traced, five exact untraced '
         'verifications and one serialized full recovery. Fixed FIT0..3 '
         'and adjacent unused FIT24..31 predictions saved for every arm; '
         'not IID, DEV or test. Inference traces/reconstruction/admission/'
         'verification/evaluation remain additional work. Total campaign '
         'FLOPs/traffic/energy unknown, not zero; measured step work is '
         'not the campaign total. Earlier negative content evidence retained.'),
        ('small',f"Theory136; {r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         'Numerically admitted cost/variance tradeoff only. Temporal races, '
         'private sparse state/key-values and counterfactual learning '
         'retained. AWS corrected replay10M plus exact teachers/controls '
         'and other-host live-gate/calibrated language quality remain priority.')])
    return pages
