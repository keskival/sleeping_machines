"""Corrected causal language numerical port, full-size work and horizon limits."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/cross_host_bridge_training_evidence.py'))
FILES={'contracts':'diagnostics/local_causal_language_replay_contracts_20261003T004700Z.json',
       'resource':'diagnostics/local_causal_language_replay_resource_20261003T005100Z.json',
       'horizon':'diagnostics/local_credit_horizon_contracts_20261003T005700Z.json'}


def load(read):
    data={'prior':BASE['load'](read),**{k:read(v) for k,v in FILES.items()}}
    for r in [data[k] for k in FILES]:
        if r['status']!='completed':raise ValueError('Completed causal replay evidence required')
        for name,digest in r['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed causal replay source '+name)
    return data


def pages(data):
    pages=BASE['pages'](data['prior']);contract=data['contracts'];resource=data['resource'];horizon=data['horizon']
    rows=[[r['family'],str(r['parameters']),str(r['entering_events']),str(r['entering_bytes']),
           str(r['final_bytes']),str(r['compared_nonzero_parameter_gradients']),str(r['all_route_shadow_lanes'])]
          for r in contract['families']]
    pages.append([('h1','Appendix B. Corrected causal language replay: sixteen contracts pass'),
        ('table',(['L8/p4 family','Params','Entering events','Entering bytes','Final bytes','Parameter grads','Shadow lanes'],rows,[33,21,27,29,24,29,24])),
        ('p','A new sibling language helper emits EVERY token prediction from '
         'nonempty detached native receiver memories/arrivals/source context. For '
         'each race at event t, replay each alternative at the factual FIRST time '
         'with common future randomness, real private memory writes and coupled '
         'suffix clocks/messages; stopped-gradient utility sums losses t..T-1. '
         'Categorical local-expectation credit is added to factorized common-clock '
         'and realized-content derivatives. Original hard score map retained, '
         'no bounded bridge/calibration, token target in input or per-position KV bank.'),
        ('p','Both private and depth-shared L8/H2/p4/pool2 double models pass '
         'EVERY-parameter sum-return gradient comparison with independent sequential '
         'full-write replays, all private state/arrivals/context comparisons, '
         'original-teacher/factorized exact forward-state parity and variable2/3 '
         'token lanes. Losing routes preserve factual first time, change actual '
         'suffix predictions and preserve preceding predictions. Future observed '
         'tokens and changed target labels cannot affect earlier factual output.'),
        ('p','Actual partly accumulated gradients, native state, Adam, cursor and '
         'RNG recover bitwise exactly; independent sequential summed gradients '
         'agree after target normalization/clipping/Adam. All factual/shadow/backward '
         'and optimizer operations covered.96shadow lanes/288shadow events per '
         'three-target case are charged. The port closes a numerical mechanism '
         'gap; no benchmark fit or deep-feature advantage is inferred.'),
        ('small','Theory108;23.973s/361084KiB,16contracts. State bytes are actual '
         'double-precision numerical states, not the production float ledger. '
         'Truncating/detaching entering credit still omits later-chunk derivatives '
         'and gives no whole expected-stream gradient theorem. Existing AWS10M '
         'original-teacher controls and their immutable sources stay distinct; '
         'no new10M quality cell or claimed language replay gain.')])
    rows=[]
    for r in resource['common_unit_ledger']:
        rows.append([r['family']+'/'+r['window'],str(r['targets']),f"{r['whole_step_gflops']:.6f}",
            f"{r['fit_mflops_per_target']:.6f}",f"{r['inference_mflops_per_target']:.6f}",str(r['shadow_lanes'])])
    for family,r in data['prior']['language'].items():
        w=r['work'];cpu=w['cpu_emulator'];n=w['fitting_targets']
        rows.append([family+'/teacher smoke',str(n),f"{cpu['total_training_unit_special_flops']/1e9:.6f}",
            f"{cpu['total_training_unit_special_flops']/n/1e6:.6f}",
            f"{(cpu['inference_arithmetic_flops_per_character']+cpu['inference_special_functions_per_character'])/1e6:.6f}",'0'])
    pages.append([('h1','Appendix B. Production-size language replay: paid resource admission'),
        ('table',(['p16/L8 case','Targets','Whole step/fit GF','Fit MF/target','Infer MF/target','Shadow lanes'],rows,[43,18,37,30,29,27])),
        ('p','Private54907/depth-shared22687 parameters,H2/pool2,32available '
         'receivers,32key scores/16actual writes per observed token. A full16target '
         'chunk executes512shadow lanes/8192shadow events; partial3 executes96/288. '
         'Actual original temporal state and hard races remain. Both full factual '
         'primal outputs/state match independent sequential native computation. '
         'Four prespecified suffix returns per family match sequential alternatives '
         'at factual first time; all eight earlier-loss/end-RNG checks pass.'),
        ('p','After each real full-step Adam update, save model/nonempty optimizer/ '
         'private state/cursor/RNG and recover the next actual three-target partial '
         'update bitwise exactly. Total four test updates plus two recovery repeats '
         'are paid. First-state2376/2304bytes, final2448bytes, float32. Every '
         'factual/shadow/backward/normalization/clipping/Adam/native-inference '
         'operation has complete coverage. Numerical admission succeeds86.859s/ '
         '429912KiB under guarded one-thread tmux,~11GiB host memory available.'),
        ('p','Full replay costs67.78/67.74MF per target here versus saved original '
         'teacher smoke about.45MF. Tables show whole executed-step or whole fit '
         'work and the corresponding actual target denominator in the SAME units. '
         'Synthetic16/3target correctness strings and saved1024target text8 smokes '
         'have unequal data/quality/update context; raw work gaps are not efficiency '
         'claims. Native sparse inference stays~.097MF/target but does not remove '
         'counterfactual learning work. No DEV/test/BPC or fitted replay gain.'),
        ('small','Theory109; production tests cover full execution and selected '
         'returns/recovery. Full EVERY-parameter sequential-return comparison is '
         'the prior T3/p4 double contract, not relabelled full T16 equality. '
         'TwoFLOPs/MAC plus unit specials; shadow-state export and taps are paid '
         'diagnostic execution/traffic. Development passes, physical traffic/RNG/ '
         'energy remain separate. Whole diagnostic-audit FLOPs unknown, not zero. '
         'No unchanged language training promotion follows from resource admission.')])
    rows=[[str(r['horizon']),f"{100*r['geometric_tail_fraction']:.3f}",f"{r['absolute_geometric_tail']:.6f}"]
          for r in horizon['geometric_tail']]
    pages.append([('h1','Appendix B. Persistent representation and missing delayed credit'),
        ('table',(['Credit horizon H','Geometric weighting after H (%)','Absolute geometric tail'],rows,[44,80,61])),
        ('p','Exact constructed encoder: observe x,write theta*x,transport by '
         'rho^32 with rho=exp(-.01),predict delayed label x. At theta.3,full '
         'encoder gradient -.567961,credit16 with detached future state0; '
         'theta.31 reduces loss.305883 to.300230. Race writes x or0; delayed '
         'conditional choice gradient -.114477,within-chunk return gradient0 '
         'despite replaying both choices. Seven analytic/autograd/finite-difference '
         'and exhaustive-sampling contracts pass. This proves a training-credit '
         'obstruction with representational capacity retained, not a measured text8 failure.'),
        ('p','The linear contraction-mode tail fraction rho^H is large for a '
         '100event decay. Native whole-state contraction is NOT established: '
         'fixed-input unit damping/orthogonal rotation do not bound learned '
         'source carry, gates, clock history and addressed interactions. Shared '
         'parameters can get other later derivatives; that is not automatic '
         'recovery of the omitted original-write path. Decoder adaptation is '
         'also not proof of learning new long-range representations.'),
        ('p','Fixed execution-count alternative: fullT16/L8/H2/pool2 has512 '
         'shadow events/target plus one factual; T64/k8 uniform distinct sites '
         'has16shadow events/target plus one factual. The longer factual graph '
         'and score discovery still cost work. Horvitz scalingR/k=128 makes '
         'single-informative-route covariance127*g*g^T. Exhaustive12race/k3 '
         'subsets independently verify mean and covariance. Unbiasedness can '
         'coexist with poor learning; these counts are not FLOPs/wall/quality advantage.'),
        ('small','Theory110; .529s/282228KiB,no optimizer/DEV/test. Corrected '
         'within-chunk routing support, confidence/sensitivity and credit horizon '
         'are separate constraints. Next use exact-source saved language checkpoints '
         'for a frozen longer-suffix shared-parameter return/variance audit on '
         'unseen FIT inputs before selecting H/k. Existing10M teacher runs '
         'remain unchanged; no long sampled-credit fit or predicted benchmark '
         'gain is admitted by this synthetic witness.')])
    return pages
