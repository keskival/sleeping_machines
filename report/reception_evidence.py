"""Completed window-credit/reference and native intervention, plus AWS decoder update."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=['local_race_window_credit_contracts_20261002T222000Z.json',
       'local_dvs_native_window_intervention_20261002T223000Z.json',
       'aws_frozen_polynomial_20261002T215500Z.json']


def load(read):
    data=[read('diagnostics/'+name) for name in FILES]
    for r in data:
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed reception/decoder numerical source: '+name)
        for parent in r.get('parents',[]):
            if hashlib.sha256((ROOT/parent['result']).read_bytes()).hexdigest()!=parent['result_sha256']:
                raise ValueError('Changed reception producer')
    return data


def pages(data):
    law,native,polynomial=data;result=[]
    result.append([('h1','Appendix B. Receiving losing arrivals requires boundary credit'),
        ('table',(['Constructed gradient','Complete conditional reference','Ordinary history-cell autograd'],
            [['Width dL/dH',f"{law['exact_width_gradient']:.6f}",f"{law['ordinary_history_width_gradient']:.6f}"],
             ['Zero-width right derivative',f"{law['zero_width_one_sided_gradient']:.6f}",'Birth credit required']],
             [49,62,62])),
        ('p','For a common-start exponential race, winner W and first time T factor as '
         'W~Categorical(lambda/sum lambda), T=E0/sum lambda. Conditional losing residual '
         'arrivals are independent Exp(lambda_j). Winner-only replay omits those residuals '
         'when a layer receives several messages. Under the native increasing bounded '
         'delay map, a deadline H after the first arrival gives an explicit raw residual '
         'cutoff R(T,H) and losing membership q_j=1-exp(-lambda_j R).'),
        ('p','The complete expected gradient decomposes into ordinary fixed-history derivatives, '
         'winner choice and paired boundary flux. At a losing arrival crossing the deadline, '
         'dq_j multiplies the downstream loss difference between heard and unheard outcomes, '
         'including the actual separate receiver write. This produces faster/slower emitter '
         'and reception-width credit while preserving coupled contents and physical times.'),
        ('p','Seven numerical contracts verify joint arrival density, physical cutoff/cap, '
         'membership normalization/activity, every score/content/decay/width derivative, '
         'independent finite differences, zero-width birth and actual-write effects. '
         'Three candidates, decaying messages and a later addressed-memory read; quadrature '
         'agrees with the boundary/winner/history formula. Maximum width finite-difference '
         'error 4.41e-9. The constructed sign reversal shows ordinary autograd can widen '
         'or narrow in the wrong direction; it does not identify a benchmark failure.'),
        ('p','Enumerating membership histories grows exponentially. Paired boundary sampling '
         'can avoid enumeration, but every candidate discovery, actual timed counterfactual '
         'and future replay is paid and variance remains. Conditional inverse-CDF derivatives '
         'already redistribute heard arrival times; adding the full unconditional boundary '
         'term to them double counts credit. Neither primitive supplies a free whole-model gradient.'),
        ('small','Theory97, guarded .743s/295,904KiB reference. No optimizer or fitted native '
         'window. Fixed-first deadline differs from silence-reset popcorn, whose scheduler '
         'and merge/split credit remain separate. FLOPs/traffic/energy unmeasured, not zero. '
         'The next pages test real native information/access before any training admission.')])
    modes={'winner_only':'Winner only','window_sum':'Heard sum + writes','window_mean':'Heard mean + writes',
        'winner_gain':'Winner x heard count','winner_wait':'Winner + same wait','delivery_only_sum':'Heard sum, winner write'}
    for seed in (6,7):
        rows=[];selected=[r for r in native['rows'] if r['seed']==seed]
        initial=[r for r in selected if r['encoder']=='initial']
        trained=[r for r in selected if r['encoder']=='fixed_pass4']
        for x,y in zip(initial,trained):
            assert (x['width'],x['mode'])==(y['width'],y['mode'])
            extra=sum(len(c['window']['heard'])>1 for c in y['cases'] if c['window'])
            rows.append([f"{x['width']*1000:.0f}",modes[x['mode']],f"{x['nll']:.6f}",f"{y['nll']:.6f}",
                f"{100*x['accuracy']:.2f}",f"{100*y['accuracy']:.2f}",str(extra)+'/16'])
        representative=[]
        for width in (0.,.001,.003):
            r=next(r for r in trained if r['width']==width and r['mode'] in ('winner_only','window_mean'))
            core=(r['representative_first_prefix_core_arithmetic_flops']+r['representative_first_prefix_special_evaluations'])/1e6
            representative.append([f'{width*1000:.0f}',f'{core:.6f}',
                f"{min(c['final_ready_time'] for c in r['cases']):.6f}",f"{max(c['final_ready_time'] for c in r['cases']):.6f}"])
        result.append([('h1',f'Appendix B. Native reception intervention, seed{seed}: no smoke admission'),
            ('table',(['H ms','One-site mode','Initial NLL','Trained NLL','Initial acc %','Trained acc %','Extra arrivals'],rows,[12,43,26,26,22,22,22])),
            ('table',(['Mean/winner H ms','First-prefix core MF est.','Minimum ready time s','Maximum ready time s'],representative,[36,48,44,45])),
            ('p','Frozen fixed-four-pass producer and its initial reservoir; 16 prespecified '
             'evenly spaced FIT examples unused by the256-label producer. No optimizer, '
             'decoder refit, development or official-test evaluation. One actual site: '
             'observed packet event19/layer0/head0; the following deeper computation/query '
             'sees real changed receiver memories. Transport runs to the actual deadline.'),
            ('p','Prespecified1ms heard-mean versus same-wait winner gate FAILS in both '
             'trained seeds: zero extra arrivals and zero NLL difference. At3ms seed6 '
             'hears extras4/16, seed7 7/16; mean-over-wait NLL gains .001108/.002312. '
             'Seed6 sum worsens loss. Those diagnostic settings cannot replace the frozen '
             'gate. Failure concerns this site/width/sample, not all learned windows.'),
            ('small','Theory98. Eight available receivers, four original writes/event and '
             '168 scored keys/prefix; only this site can add one actual write. Original '
             'trained fit2.285696GF/2.232125MF per fitting presentation retained. Table '
             'inference MF is the first selected prefix only, not full-audit FLOPs. '
             'All44 model/configurations104.222s/396,720KiB; twelve zero-nesting/gradient/ '
             'causality/write/target-invariance/refused-training contracts pass. Frozen '
             'weights remain bitwise equal. Unknown audit/scheduler/traffic/energy work '
             'is not zero. Processing readiness is charged beyond the1s observation cutoff.')])
    rows=[]
    for r in polynomial['rows']:
        infer=r['inference_mflops_per_query']
        rows.append(['Fitted' if r['fitted_encoder'] else 'Initial',str(r['seed']),str(r['degree']),
            f"{100*r['final']['accuracy']:.2f}",f"{r['final']['nll']:.6f}",
            'Unmeasured' if infer is None else f'{infer:.6f}'])
    result.append([('h1','Appendix B. AWS frozen polynomial decoder: both nomination gates fail'),
        ('table',(['Encoder','Seed','Degree','Accuracy %','Dev NLL','Native infer MF/query'],rows,[27,17,19,31,33,46])),
        ('p','All three coarse trained encoders and exact initial reservoirs,984 fitting/ '
         '192 development examples. Local affine/quadratic heads selected by fitting-user '
         'GroupKFold over C(.01,.1,1);108 fold fits,12 final fits and six feature replays. '
         'No encoder retraining, dense prefix carrier or resident-memory read. Algebraic '
         'folding reproduces actual native predictions/state; every sequential head is charged.'),
        ('p','Fitted affine mean68.056%/.903627; quadratic69.965%/.885142. Only .018485 '
         'mean NLL gain and1.910 accuracy points, with seed6 NLL regression .035176, '
         'fail the polynomial nomination. Gains over parent native .020633/.157749/ '
         '.034854 miss the each-seed .05 requirement. Initial affine58.507%/1.138019 '
         'and quadratic62.847%/1.012070 remain visible. Both gates FAIL; no unchanged scaling.'),
        ('small','Original trained encoder fit4.218015GF each, .535825MF per7,872 fitting '
         'presentations; initial optimizer work zero. Combined fitting and solver FLOPs '
         'UNKNOWN, so parent fit alone is not total work. Whole study126.813s/608,044KiB '
         'includes every fit/replay/evaluation/export; one convergence warning retained. '
         'Tensor bytes exclude shared input normalization and metadata. Conditional '
         'decoder folds reuse label-trained encoders; no pipeline cross-fit or test. '
         'RBF context evidence remains stronger; raw coarse77.604%/.686661 and historical '
         'compact66.667%/.902951 controls preserved. AWS owns the next compact context head.')])
    return result
