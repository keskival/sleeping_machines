"""Completed contracted projection of actual-batch native race covariance."""
import hashlib
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/depth_sampling_evidence.py'))
RESULT='diagnostics/local_projected_batch_credit_variance_20261003T035500Z.json'


def load(read):
    d=dict(prior=BASE['load'](read),projection=read(RESULT));r=d['projection']
    if r['status']!='completed' or r['contracts_passed']!=4:raise ValueError('Completed projection contracts required')
    for name,digest in r['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed projection source '+name)
    parent=ROOT/r['parent']
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=r['parent_sha256']:raise ValueError('Changed exact-vector parent')
    if [c['depth'] for c in r['cases']]!=[4,6] or any(c['batch']!=16 for c in r['cases']):
        raise ValueError('Actual16-target depth comparison required')
    return d


def pages(d):
    pages=BASE['pages'](d['prior']);r=d['projection'];rows=[]
    for c in r['cases']:
        for e in c['estimates']:
            rows.append([str(c['depth']),str(c['batch']),str(e['k']),
                f"{e['combined_MSE_ratio_estimate']:.6f}",f"{e['combined_MSE_ratio_empirical_SE']:.6f}",
                f"{100*e['empirical_relative_SE']:.2f}",str(e['directions'])])
    pages.append([('h1','Appendix B. Actual training batch: contracted parameter projections'),
        ('table',(['Depth','Batch','k','Combined MSE ratio estimate','Empirical SE','Relative SE %','Sign probes'],rows,[16,16,14,44,32,28,24])),
        ('p','p16/H2/pool2, initialized seed7, SAME original fine FIT0..15 '
         'transform and fixed common race noise. Each full16-example '
         'episode graph retains computational delays, temporal races, '
         'private addressed state and separate keys/values. No DEV/test '
         'arrays, optimizer step, fit quality or inference saving. '
         'Different depth factories are a configuration ladder, not '
         'an isolated causal depth perturbation.'),
        ('p','Pool2 conditional choice credit equals pi0*pi1*(Q0-Q1)/B '
         'times the complete parameter Jacobian of score0-score1. An '
         'artificial score cotangent gives that Jacobian times a random '
         'parameter-sign vector through reverse-over-reverse pullback. '
         'This differentiates cotangents, not a physical stochastic '
         'Hessian. Detached alternative returns and the native factorized '
         'clock/value backward remain unchanged.'),
        ('p','THREE directions per D2/D4/D6 reproduce EVERY projected '
         'race contribution against Theory124\'s exact vectors, maximum '
         'error1.38e-14. Main estimates use32 independent Rademacher '
         'directions with no1/sqrt(parameter-count) scaling. Exact '
         'finite-population sampling covariance is projected and centered '
         'within each legal episode; its normalized trace estimate is '
         'unbiased. All probes are saved; empirical SE is descriptive, '
         'with no guaranteed95% interval. Worst-case relative RMS bound25%.'),
        ('p','At actual B16, k8 relative raw-noise RMS is about.283/.383 '
         'for D4/D6. Compared with B2 exact MSE.08463/.30506, D4 scarcely '
         'changes while D6 about halves. Sampling interference remains '
         'measurable in this initialization history. Larger k reduces '
         'raw variance, but does not establish Adam-update precision, '
         'trained causality, heldout gain or a cure; the finite functional '
         'forks retain mixed anchor behavior.'),
        ('p','Decision: reuse owned live-gate and gain-lineage comparisons '
         'and the original streaming AWS10M quality matrix. No sampling '
         'sweep admitted. The90M configuration-selection correction uses '
         'DEV metrics, preserving the old written test-based rule beside '
         'its correction; completed test scores remain reporting-only.'),
        ('small',f"Theory127/128;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         '13440 full shadow lanes/282240 shadow events,64 main mixed '
         'pullbacks plus9 contract projections and factual/full backwards; '
         'total diagnostic FLOPs/traffic/energy unknown, not zero. '
         'Randomized trace estimation is a known primitive (Avron/Toledo2011); '
         'our native conditional-covariance application is contracted, '
         'not a novelty or equivalent-work wall-speedup claim.')])
    return pages
