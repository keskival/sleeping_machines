"""Add completed shared depth8 confirmation without changing frozen bridge pages."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/bridge_training_evidence.py'))


def load(read):
    data={'bridge':BASE['load'](read),'gesture':{},'language':{}}
    for seed,tag,stage in [(7,'aws_depth8_replay_20261002T235500Z','pilot'),
                           (8,'aws_depth8_replay_20261002T234300Z','confirmation')]:
        data['gesture'][seed]={}
        for family in ('private','depth'):
            for credit in ('teacher','factorized','replay'):
                name=f'dvs_native/{tag}_{family}_{credit}_{stage}_s{seed}.json'
                r=read(name)
                if r['status']!='completed':raise ValueError('Completed depth8 evidence required')
                if r['work']['fitting_targets']!=1024:raise ValueError('Depth8 presentation denominator')
                data['gesture'][seed][family,credit]=r
    for family in ('private','depth'):
        data['language'][family]=read(f'aws_depth8_language/aws_depth8_language_20261002T234100Z_{family}_smoke_s7.json')
    for r in [*data['language'].values(),*(r for group in data['gesture'].values() for r in group.values())]:
        if r['status']!='completed':raise ValueError('Completed depth8 sources')
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed depth8 source '+name)
    return data


def pages(data):
    pages=BASE['pages'](data['bridge'])
    for seed in (7,8):
        rows=[];group=data['gesture'][seed]
        for family in ('private','depth'):
            for credit in ('teacher','factorized','replay'):
                r=group[family,credit];w=r['work']
                rows.append([family+'/'+credit,str(r['parameters']),f"{100*r['final']['accuracy']:.3f}",
                    f"{r['final']['nll']:.6f}",f"{w['whole_fit_unit_special_flops_estimate']/1e9:.6f}",
                    f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.6f}",
                    f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.6f}"])
        interpretation=(
            'Both seed7 families pass the predeclared .03NLL gain versus BOTH matched '
            'controls and at most1pp accuracy loss. Private replay improves NLL '
            '.052409/.082802 versus teacher/factorized and accuracy8.333/9.375pp. '
            'Depth-shared replay improves .062138/.046883NLL; teacher accuracy '
            'declines .521pp within the gate. These positive heldout results are '
            'preserved. The unchanged seed8 confirmation below subsequently fails '
            'both family gates: nomination is not robust, and no full984FIT promotion follows.'
            if seed==7 else
            'Both unchanged seed8 confirmation gates FAIL. Private replay preserves '
            '.048736NLL improvement over teacher, but loses1.5625pp accuracy and '
            'is .012051NLL worse than factorized. Depth-shared replay improves only '
            '.012207/.000455NLL versus teacher/factorized, below .03, and declines '
            '1.0417pp versus factorized accuracy. Keep the earlier positive seed7 '
            'result beside this failure; neither establishes broad failure of deep '
            'representation learning or practical advantage.')
        pages.append([('h1',f'Appendix B. AWS depth8 replay: seed{seed} '+('positive pilot' if seed==7 else 'confirmation fails')),
            ('table',(['Family/credit','Params','Dev acc %','Dev NLL','Whole fit GF','Fit MF/target','Infer MF'],rows,[36,20,22,25,28,31,25])),
            ('p','Both families retain native computational delays/races, key/value '
             'separation, source carry, sparse writes and private persistent receiver '
             'state. Private maps differ perdepth/receiver; depth-shared maps are '
             'shared acrossdepth/pool perhead, with private keys/clocks/timescales. '
             'p16/L8/H2/pool2,256FIT/192DEV,fourpasses/1024presentations,U16/lr.003, '
             'coarse4/.25clock.32available units,32key scores/16writes per event; '
             'five-event prefix80factual races/160full shadow lanes. All163840 '
             'shadow lanes and819200 replay events perfit are charged.'),
            ('p',interpretation),
            ('p','Original teacher, factorized clock/content control and corrected '
             'all-race first-time-preserving actual-write replay share data/order/noise/ '
             'optimizer budgets. Full every-parameter/recovery/state/shared-alias '
             'contracts and allsix learning smokes pass before pilot/confirmation. '
             'Replay costs about52–55x the controls in complete fitting arithmetic; '
             'equal sparse inference does not remove that expense. No comparable-quality '
             'total-resource advantage, officialtest or unchanged fullfit is claimed.'),
            ('small','Same fitting denominator and units for every row/column; '
             '2FLOPs/MAC plus unit specials, first11 DEV prefix inference mean. '
             'Reused DEV epoch selection, two seeds and fixed nomination scope. '
             'Input preprocessing, candidate index/RNG, traffic and measured energy '
             'separate. Verbatim shared report evidence preserved in appendices/ '
             'aws_depth8_history_20261003T003500Z.md. Source/result hashes checked; '
             'historical depth4 failed gate and local bridge evidence retained.')])
    rows=[]
    for family,r in data['language'].items():
        w=r['work'];cpu=w['cpu_emulator'];n=w['fitting_targets']
        rows.append([family,str(r['parameters']),str(n),f"{r['initial_dev']['bpc']:.6f}",
            f"{r['final']['dev']['bpc']:.6f}",f"{cpu['total_training_unit_special_flops']/1e9:.6f}",
            f"{cpu['total_training_unit_special_flops']/n/1e6:.6f}",
            f"{(cpu['inference_arithmetic_flops_per_character']+cpu['inference_special_functions_per_character'])/1e6:.6f}"])
    pages.append([('h1','Appendix B. Deep causal language: admission only; ten-million fits pending'),
        ('table',(['Family','Params','Fit targets','Initial BPC','Final BPC','Whole fit GF','Fit MF/char','Infer MF/char'],rows,[25,23,22,24,24,27,24,24])),
        ('p','Both integrated native depth8 families pass causal token/teacher parity '
         'and actual private-state/Adam/cursor/RNG/partial-window recovery contracts. '
         'These1025-character/1024target,513character/512target DEV,onepass,credit16/ '
         'U256 learning smokes use the ORIGINAL counterfactual teacher, not corrected '
         'full replay. Model key/value separation, sparse receiver writes and physical '
         'time evolution remain.32available units,32scored keys/16writes perchar; '
         '512target DEV retains2448bytes addressed state in both families. '
         'Four optimizer updates, all core fitting work charged.'),
        ('p','Private BPC5.311638 to4.832740,shared5.273659 to4.807838 are admission '
         'checks, not language advantage or evidence that deeper features generalize. '
         'AWS reserved coordinator now owns private/shared first10M-character FIT '
         'models,onepass,disjoint1M DEV at90M,seed7,credit16/U256/lr.002/warmup4096. '
         'No completed10M quality cell is filled here. Gesture replay nominations '
         'failed confirmation; language teacher runs remain distinct controls.'),
        ('p','The corrected language choice return must sum ALL downstream token '
         'losses after the forced route, actually changing private memory at the '
         'factual FIRST time. A detached chunk omits later-chunk credit. Full replay '
         'at T16/L8/H2/pool2 costs512shadow lanes/8192shadow events perchunk; '
         'causal suffix snapshots can reduce about2x, not erase the quadratic work. '
         'A future language port needs full-parameter sequential/batched sum-return '
         'contracts, causal intervention, actual recovery and complete paid work.'),
        ('small','Completed smokes250.174/251.253s,peak<534MiB; bounded long jobs '
         'stay in AWS tmux/coordinator with reserved2CPU slots and8GiB floor. '
         'Table uses CPU emulator arithmetic plus unit-special convention for both '
         'models; projected physical-event numbers remain separate. Credit graph '
         'freed each16target microchunk, persistent state crosses updates. DEV passes/ '
         'RNG/traffic/energy outside fitting FLOPs, not zero. Source snapshots frozen; '
         'all pending runs retained as pending and independently owned.')])
    return pages
