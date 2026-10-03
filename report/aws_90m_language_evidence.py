"""Completed source-file evidence only; pending arms never supply scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NATIVE=[('p32/D4/pool2','aws_language_batched_90M_r2_p32d4_linear_l64_lr004_cmp_s6_20261003T103000Z'),
        ('p32/D4/pool4','aws_language_batched_90M_r2_p32d4_pool4_linear_l64_lr004_cmp_s6_20261003T110000Z')]


def pages():
    rows=[];models=[]
    for label,tag in NATIVE:
        path=ROOT/'experiments/results/language_batched'/f'{tag}.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        if r['status']!='completed':continue
        assert r['args']['fit']==90_000_000 and r['args']['route_credit']=='linear'
        models.append(r)
        rows.append([label,str(r['parameters']),f"{r['test_bpc_eval_segment']:.6f}",
                     f"{r['work']['whole_fit_unit_special_flops_estimate']/1e9:,.1f}",
                     f"{r['work']['fit_unit_special_flops_per_char_estimate']/1e6:.6f}"])
    for pattern,label in [('*lstm_D90000000*.json','LSTM (6 passes)'),('*tf_D90000000*.json','Transformer (4 passes)')]:
        path=next((ROOT/'experiments/results/aws_20260929').rglob(pattern))
        r=json.loads(path.read_text());work=r['training_flops_estimate']['total_training_flops']
        rows.append([label,str(r['params']),f"{r['test_bpc']:.6f}",f'{work/1e9:,.1f}',
                     f"{work/r['training_token_positions']/1e6:.6f}"])
    if not models:return []
    return [[('h1','Completed AWS 90M native language comparisons'),
        ('p','<b>Native pool4 reaches 1.998 test BPC at T256, versus 2.045 for pool2.</b> '
         'Both use payload32/depth4, two heads, linear local value-informed route credit and '
         'eight selected addressed writes per character. Pool4 doubles scored keys from16 to32; '
         'available receiver capacity increases. These are single-seed results, not replicated advantage.'),
        ('table',(['Model','Params','Test BPC','Fit GFLOPs','Fit MF/target'],rows,[49,23,25,37,35])),
        ('p','Both native fits use FIT[0,90M), 89,997,312 presentations, 10,986 Adam windows, '
         '64 lanes of128-character reset segments, cosine lr0.004, seed6. DEV[90M,91M) and '
         'test[95M,96M), 999,936 targets each; T256 is an additional scorer of the same final weights. '
         'Random segments with replacement constitute a pass-equivalent budget, not full unique coverage.'),
        ('p','Pool4 improves its matched-window10M score2.345157 to1.998416, a0.346742BPC scaling gain. '
         'At90M its0.046941BPC gain over pool2 costs1.649917x estimated fitting work. '
         'This is useful capacity with unchanged selected activity, not unchanged total work or an iso-FLOP gain.'),
        ('small','All work columns share units and divide by each model’s actual fitting presentations. '
         'Native estimates extrapolate two complete traced optimizer windows; controls use shape-based forward, '
         '2x-forward backward and approximate clip/Adam. Validation/test, traffic and energy are excluded. '
         'Unequal parameters/passes/quality and different estimate conventions prevent comparable-quality supremacy claims.')],
        [('h1','90M protocol, resource boundaries and next decisions'),
         ('p','Native pool4 final DEV is1.913993(T128)/1.915118(T256), test1.997194/1.998416. '
          'Pool2 DEV is1.963433/1.962040, test2.045381/2.045356. Configuration decisions use DEV; '
          'test is reporting-only. Final weights, immutable recovery milestones, source hashes and guarded logs are saved.'),
         ('p','Pool4 took10,712.749s, pool2 took7,642.345s, with fitting rates8773.50 and12361.17 targets/s. '
          'PeakRSS is recorded in their JSONs. These wall times include their own evaluation and do not establish a '
          'paired wall/energy comparison with older dense runs. The saved controls remain better in quality.'),
         ('p','Pool4 fitting work is107,606.868GF, or1.195668MF/presentation. Saved LSTM and Transformer '
          'whole-fit estimates are36.18x and74.35x greater, but their test scores are1.661 and1.604. '
          'These are raw unequal-quality/data work gaps, not achieved quality-matched savings.'),
         ('p','Inference work for these trained90M weights remains pending. A prior same-shape winner-only '
          'arithmetic trace is a separate diagnostic: trained winner/state/cache/RNG contracts and complete '
          'scoring work are required before assigning it to this quality result. Candidate scoring, cache creation '
          'and optimizer work cannot be inferred away from selected-write counts.'),
         ('p','The active90M depth8 fit and queued width64 fit test remaining scale choices. Original streaming '
          'private full replay and teacher fits continue separately: they preserve chronological state and a different '
          'counterfactual estimator. No pending score is predicted. Full causal replay, useful distant-state credit, '
          'replication and comparable-quality resources remain open.'),
         ('small','Sources: experiments/results/language_batched/aws_language_batched_90M_r2_*.json '
          '(completed full fits only), and saved aws_20260929 LSTM/Transformer90M results. '
          'The temporal-race, persistent private-state and key/value architectural case is retained.')]]
