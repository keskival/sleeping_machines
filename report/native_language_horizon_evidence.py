"""Frozen native horizon observations and unchanged independent confirmation."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/causal_language_replay_evidence.py'))
FILES={'first':'diagnostics/local_native_language_horizon_20261003T010500Z.json',
       'confirmation':'diagnostics/local_native_language_horizon_confirmation_20261003T011000Z.json'}


def load(read):
    data={'prior':BASE['load'](read),**{k:read(v) for k,v in FILES.items()}}
    for r in [data[k] for k in FILES]:
        if r['status']!='completed':raise ValueError('Completed native horizon audit required')
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed native horizon source '+name)
        if hashlib.sha256((ROOT/r['parent']['result']).read_bytes()).hexdigest()!=r['parent']['result_sha256']:
            raise ValueError('Changed native language parent')
        if hashlib.sha256((ROOT/r['artifact']).read_bytes()).hexdigest()!=r['artifact_sha256']:
            raise ValueError('Changed full horizon parameter-credit arrays')
    return data


def pages(data):
    pages=BASE['pages'](data['prior']);r=data['first'];confirm=data['confirmation'];rows=[]
    for row in r['rows']:
        rows.append([str(row['fit_start']),'/'.join(str(x) for x in row['site']),
            f"{row['utility_contrast16']:+.6f}",f"{row['utility_contrast32']:+.6f}",
            f"{row['late_utility_contrast']:+.6f}",f"{row['parameter_credit_norm16']:.6f}",
            f"{row['parameter_credit_norm32']:.6f}",'YES' if row['horizon_sign_reversal'] else 'No'])
    pages.append([('h1','Appendix B. Native text8 horizon return: four credit reversals'),
        ('table',(['FIT start','Event/L/H','Q0-Q1 H16','Q0-Q1 H32','Late contrast','Param norm16','Param norm32','Flip'],rows,[22,23,27,27,27,24,24,12])),
        ('p','Frozen private native p16/L8/H2/pool2,54907parameters,original '
         '8192-character/fourpass seed6 producer. Fixed final-pass4 ONLINE weights, '
         'not DEV-selected state, converted to double for numerical VJPs. Two '
         'producer-unseen FIT spans8192/8320,16observed warmup tokens and32 '
         'causal next-token targets each. Prespecified sites(event0/8,layer0/7,head0), '
         'one fixed common-noise draw. Both receivers actually write private state '
         'at the factual FIRST time, then messages/clocks/memory evolve through '
         'all32targets. Original winner recovers ALLfactual logits/state/end RNG.'),
        ('p','Four of eight sites reverse the conditional choice-credit sign when '
         'losses16..31 are included. Four-site aggregate native parameter-credit '
         'cosines -.600832/-.792626 show directional opposition on these contexts; '
         'norms .051372/.032006 and.010178/.029475 for16/32returns. None of '
         'these chosen scores is capped. This is real omitted downstream utility '
         'at this frozen point, beyond theory110 synthetic witnesses, not proof '
         'that truncation dominates global learning or longer credit improves quality.'),
        ('p','Per-site credit is pi0*pi1*(Q0-Q1)*grad(scores0-scores1), with the '
         'SAMErealized-branch native score Jacobian for both horizons. Earlier '
         'pathwise clocks remain differentiable; this is not note103 fixed-clock '
         'likelihood geometry. The shown contrasts are SUMsuffix NLL, not average '
         'prediction quality. Future observed tokens/changed target labels leave '
         'earlier factual output unchanged. Allcases/full54907coordinate arrays '
         'saved; sampling only four of512sites excludes a full-chunk gradient claim.'),
        ('small','Theory111;7.026s/426172KiB.16forced32token suffixes/two factual '
         'graphs/eightVJPs plus warmup/causality checks paid; diagnostic totalFLOPs '
         'unknown, not zero. Double live state4488..4624bytes,32available/32scores/ '
         '16writes pertoken. No optimizer/DEV/test/changed-parent fit or benchmark '
         'advantage. The independent confirmation below FAILS aggregate-opposition '
         'replication; retain this original positive mechanism observation beside it.')])
    rows=[]
    for row in confirm['confirmation']:
        rows.append([str(row['fit_start']),str(row['noise_draws']),str(row['sign_reversals'])+'/12',
            f"{row['mean_credit_norm16']:.6f}",f"{row['mean_credit_norm32']:.6f}",
            f"{row['mean_credit_cosine']:+.6f}",str(row['descriptive_gate_passed'])])
    pages.append([('h1','Appendix B. Horizon confirmation: downstream effects vary by context'),
        ('table',(['Fresh FIT','Draws','Site flips','Mean norm16','Mean norm32','Mean cosine','Opposition gate'],rows,[25,17,22,29,29,29,39])),
        ('p','Same frozen producer/sites/budgets,NEW FIT8448/8576 spans and '
         'three independent evaluated race draws111330/111331/111332. Warmup '
         'seed111329 stays fixed across draws.24cases/48actual forced suffix '
         'replays,24parameter VJPs,all outcomes and coordinate arrays retained. '
         'Before observing results, diagnostic replication required both span '
         'mean16/32parameter-credit cosines<0 and at least one site flip per span. '
         'Both gates FAIL: mean directions agree,despite three of24 individual '
         'site reversals. Do not replace this gate with a larger-gradient criterion.'),
        ('p','First fresh span preserves direction across all draws,with mean '
         'longer-return norm.536700 versus.303282. Second has more variable '
         'perdraw norms/angles and mean cosine+.455223. These are supported '
         'context-dependent downstream effects; universal destructive short '
         'credit is not established. Neither frozen study measures expected '
         'fullstream risk, semantic feature content or a fitted benchmark gain. '
         'Original first-audit reversals remain valid within their scope.'),
        ('p','The current mechanism priorities stay separate: corrected actual '
         'language shadow return is numerically installed; hard-score sensitivity '
         'repair has no demonstrated quality gain; full replay work is expensive; '
         'horizon changes utility but robust allocation/generalization remain open. '
         'Next broader frozen utility/covariance accounting can inform H/k; '
         'no unchanged long sampled-horizon training follows this failed diagnostic '
         'replication. Existing10M original-teacher controls continue independently.'),
        ('small','Theory112;20.355s/470848KiB,one guarded CPU job/no optimizer/ '
         'DEV/test. Earlier-loss/factual-first-time/ALLoriginal-winner-state/RNG/ '
         'future-token/target invariance and unchanged parameters pass on all '
         'three draws. Same original fit work paid; whole audit arithmetic, '
         'physical traffic and energy unknown. More counterfactual support cannot '
         'automatically repair omitted future loss, while a longer return is '
         'also not automatically a better finite-data update. No supremacy claim.')])
    return pages
