"""Production precision and completed shared AWS causal-language admission."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/language_winner_reuse_evidence.py'))
PRECISION='diagnostics/local_language_precision_audit_20261003T015800Z.json'
DIAGNOSTICS=dict(driver='diagnostics/aws_language_credit_driver_contracts_20261003T013300Z.json',
    winner='diagnostics/aws_language_winner_reuse_contracts_20261003T013700Z.json',
    winner_driver='diagnostics/aws_language_winner_driver_contracts_20261003T014000Z.json')


def load(read):
    d=dict(prior=BASE['load'](read),precision=read(PRECISION),checks={k:read(p) for k,p in DIAGNOSTICS.items()},driver_rows={},smokes={})
    for family in ('private','depth'):
        for credit in ('teacher','factorized','replay'):
            d['driver_rows'][family,credit]=read(f'aws_depth8_language_credit/aws_language_credit_driver_contracts_20261003T013300Z_{family}_{credit}_full.json')
            d['smokes'][family,credit]=read(f'aws_depth8_language_credit/aws_language_credit_matrix_20261003T013500Z_{family}_{credit}_smoke_s7.json')
        d['smokes'][family,'reuse']=read(f'aws_depth8_language_credit/aws_language_winner_matrix_20261003T014100Z_{family}_replay_smoke_s7.json')
    for r in [d['precision'],*d['checks'].values(),*d['driver_rows'].values(),*d['smokes'].values()]:
        if r['status']!='completed':raise ValueError('Completed actual language evidence required')
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed actual language source '+name)
    for key in ('parent_result','vectors'):
        r=d['precision'];path=r[key] if key=='parent_result' else r[key]['path']
        digest=r['parent_result_sha256'] if key=='parent_result' else r[key]['sha256']
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed precision source vectors/parent')
    if len(d['checks']['driver']['cases'])!=6 or len(d['checks']['winner']['cases'])!=4 or len(d['checks']['winner_driver']['cases'])!=2:
        raise ValueError('All AWS driver/winner cases required')
    for r in d['driver_rows'].values():
        if r['work']['fitting_targets']!=48:raise ValueError('Actual 48-target numerical protocol required')
    for r in d['smokes'].values():
        if r['work']['fitting_targets']!=1024 or r['args']['dev']!=129:raise ValueError('Actual 1025/129-character smoke required')
    return d


def pages(d):
    pages=BASE['pages'](d['prior']);r=d['precision'];rows=[]
    for f in r['families']:
        comparisons=f['precision_comparisons']
        counts={k:sum(t['lanes']*t['races'] for t in v['all_factual_and_shadow_route_precision_mismatches']) for k,v in comparisons.items()}
        rows.append([f['family'],f"{f['double_original_reuse_error']['relative_l2']:.3g}",
            f"{comparisons['original']['gradient_error_against_double']['relative_l2']:.3g}",
            f"{comparisons['reuse']['gradient_error_against_double']['relative_l2']:.3g}",
            str(counts['original'])+'/'+str(counts['reuse']),
            str(sum(t['mismatches'] for v in comparisons.values() for t in v['all_factual_and_shadow_route_precision_mismatches']))])
    pages.append([('h1','Appendix B. Production precision: equivalent gradients and every shadow route agree'),
        ('table',(['Family','Double reuse error','Float32 old error','Float32 reuse error','Old/new decisions','Mismatches'],rows,[21,29,30,31,38,25])),
        ('p','Same represented float32 p16/L8/H2/pool2 weights and nonempty '
         'private memory as Theory116, promoted to double without reinitializing '
         'the model. Same synthetic sixteen-target chunk and entering RNG. '
         'Compare EVERY native parameter gradient of original full enumeration '
         'and winner reuse at both precisions. In double the complete production '
         'gradients agree to1.44e-15/2.73e-15 relative error; factual logits/state '
         'are bitwise identical. All caller and factual-end RNG states match.'),
        ('p','The table compares each float32 implementation against its OWN '
         'double program. Both show roughly one-to-two parts per million '
         'global gradient error. ALL131328 original and65792 optimized '
         'factual/shadow race decisions per family agree across precisions; '
         'there is no branch crossing to explain away the comparison. '
         'The prior tight-coordinate failure therefore coexists with exact '
         'mathematical estimator equivalence and floating arithmetic error '
         'in BOTH implementations. Prior failed gates remain recorded, '
         'and this diagnostic applies no optimizer or precision repair.'),
        ('p','Different batch shapes can round contractions and return reductions '
         'differently despite representing the same trajectory. For local '
         'categorical credit g_i=pi_i(Q_i-mean_pi(Q)), a return error bounded '
         'by delta gives score-credit error at most2pi_i delta, before the '
         'score Jacobian pullback and its own rounding. A large common loss '
         'can amplify cancellation relative to a tiny advantage. A detached '
         'baseline leaves exact credit unchanged, but its numerical benefit '
         'must be measured before any fitted-rule change.'),
        ('small',f"Theory117;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB, four contracts. "
         'Full gradient vectors and actual factual/shadow winner histories '
         'are preserved in the source-hashed vectors artifact. Total audit '
         'arithmetic/traffic/energy are unknown, not zero. No trained text8, '
         'DEV/test or Adam step. This small rounding floor does not establish '
         'a cause of underfitting or absent deep features; larger utility, '
         'horizon, exposure and resource questions remain separate.')])
    driver_rows=[]
    for (family,credit),v in d['driver_rows'].items():
        w=v['work']['cpu_emulator'];whole=w['total_training_unit_special_flops'];driver_rows.append([family+'/'+credit,
            f"{whole/1e9:.6f}",f"{whole/48/1e6:.6f}",
            f"{(w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'])/1e6:.6f}",
            str(v['learner_counters']['shadow_lanes'])])
    proof=d['checks']['winner'];proof_rows=[]
    for name,stages in proof['work_audits'].items():
        whole=sum(s['arithmetic_flops']+s['special_function_evaluations'] for s in stages.values() if isinstance(s,dict))
        if whole!=stages['whole_arithmetic_and_unit_special_flops']:raise ValueError('Winner-proof stage total mismatch')
        proof_rows.append([name,f"{whole/1e9:.9f}",f"{whole/3/1e6:.6f}",str(96 if name=='full' else 48)])
    pages.append([('h1','Appendix B. Production causal depth8 replay driver: numerical admission complete'),
        ('table',(['L8/p4 family/credit','Whole fit GF','Fit MF/target','Infer MF/char','Shadow lanes'],driver_rows,[44,32,35,34,29])),
        ('p','All SIX actual private/shared teacher/factorized/full-replay '
         'drivers pass bitwise interrupted/resumed model/Adam/private state/'
         'cursor/RNG/counter/work recovery, including an UNTRACED third '
         'credit chunk. Each row is the SAME48 text8 fitting targets, '
         'one pass, credit16/U16/lr.002/warmup32,33 disjoint DEV characters. '
         'Every stage covers actual shadows/backward/normalize/clip/Adam; '
         'the accumulator includes backward in its combined work bucket. '
         'None of these tiny correctness fits is a language quality comparison.'),
        ('h2','Full corrected language replay: avoid the redundant winning shadow'),
        ('table',(['p4/L8 private replay','Fwd+back GF','Fwd+back MF/target','Shadow lanes'],proof_rows,[48,38,52,36])),
        ('p','Independent AWS winner-reuse proof: four private/shared one/three '
         'target double cases agree for EVERY gradient within3.26e-15 absolute '
         'error and have bitwise factual logits/state/end RNG. The second '
         'table uses the SAMEthree-target denominator and includes only '
         'forward+backward:48.2734% less work, optimizer/inference/traffic/RNG/'
         'energy excluded and not zero. Its subsequent TWO actual optimized '
         'driver recovery contracts pass, with768lanes/12288shadowevents '
         'per48-target fit. No candidate sampling or inference substitution.'),
        ('small','All columns use2FLOPs/MAC+unit-special and consistent denominators '
         'within their table; whole actual fit and per-target work appear together. '
         'The isolated three-target audit is explicitly separate from full '
         '48-target driver fits. Original teacher10M checkpoints were preserved '
         'at73728targets/288updates each for exact continuation. Original '
         'failed object-comparison test and source-scoped coordinator recovery '
         'remain historical; no learned weights were discarded. Source-backed '
         'AWS findings/manual report history are retained alongside these pages.')])
    smoke_rows=[];details=[]
    for (family,credit),v in d['smokes'].items():
        w=v['work']['cpu_emulator'];whole=w['total_training_unit_special_flops'];smoke_rows.append([family+'/'+credit,
            f"{v['final']['dev']['bpc']:.6f}",f"{whole/1e9:.6f}",f"{whole/1024/1e6:.6f}",
            f"{(w['inference_arithmetic_flops_per_character']+w['inference_special_functions_per_character'])/1e6:.6f}"])
    for family in ('private','depth'):
        old=d['smokes'][family,'replay'];new=d['smokes'][family,'reuse']
        gap=new['final']['dev']['bpc']-old['final']['dev']['bpc']
        saving=1-new['work']['cpu_emulator']['total_training_unit_special_flops']/old['work']['cpu_emulator']['total_training_unit_special_flops']
        details.append(f"{family}: original→reuse BPC difference {gap:+.3g}; counted whole-fit work -{100*saving:.3f}%; "
            f"measured wall {old['wall_s']:.1f}→{new['wall_s']:.1f}s.")
    pages.append([('h1','Appendix B. AWS deep language smokes: same learning with half the replay work'),
        ('table',(['L8/p16 family/credit','Final DEV BPC','Whole fit GF','Fit MF/target','Infer MF/char'],smoke_rows,[44,32,32,34,32])),
        ('p','ALL eight completed real causal text8 learning/RSS admissions: '
         '1025 observed FIT characters/1024 next-character targets, one pass, '
         '129 disjoint DEV characters at90M/128targets, seed7, p16/L8/H2/'
         'pool2, credit16/U256/lr.002/warmup4096/clip1. Same chronological '
         'persistent sparse state, native computational delays, key/value '
         'separation and receiver maps. Private54907/shared22687parameters; '
         '32available/32scored/16selected receiver writes pertarget. '
         'Original teacher, factorized clock/content and full-write choice '
         'credit are different learning estimators of this integrated model.'),
        ('p',' '.join(details)+' Full replay pays32768shadow lanes/524288 '
         'shadow events; reuse pays16384/262144, retaining ALLalternative '
         'write returns and unchanged sparse inference. All fitting stages '
         'are counted together; inference is the same warm selected-value '
         'native prefix including next-character loss. The measured wall '
         'comparison spans separate same-host admitted jobs, not a controlled '
         'throughput or energy benchmark.'),
        ('p','Every smoke lowers its initial DEV BPC and stays below1GBRSS. '
         'Private initial5.209543, shared5.134375; replay/reuse final scores '
         'agree within3.44e-7 BPC. These small fits demonstrate learning and '
         'implementation admission. They do not satisfy the user\'s10M '
         'character quality protocol, establish generalization advantage '
         'against counts/Transformers, or imply full replay\'s total training '
         'work is competitive with the much cheaper teacher/factorized controls.'),
        ('small','Prioritized matrix aws_language_winner_matrix_20261003T014100Z: '
         'exact original10M teacher continuations, private and depth-shared '
         'optimized full-write replay10M, and factorized10M controls as slots '
         'permit. Long outcomes remain PENDING; no predicted BPC is entered. '
         'Three bounded one-thread CPU slots on the authorized AWS host, '
         'host+slot locks/RSS watchdogs/min8GiB and serialized publication. '
         'All earlier positive/failed depth8 gesture evidence and original full '
         'replay learning rows are retained. CPU arithmetic estimates include '
         'clock simulation/gradients/optimizer; DEV passes/RNG/traffic/energy '
         'separate, no physical cost projection or supremacy claim. '
         'Original teacher recovery may discard at most4095 uncheckpointed '
         'targets each (<=.041% extra relative to10M); exact discarded work '
         'is unknown. Original1M initial DEV versus new1025-character initial '
         'diagnostics also makes startup wall unequal; final1M DEV is matched.')])
    return pages
