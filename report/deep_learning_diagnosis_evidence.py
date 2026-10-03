"""Completed causal-prefix work tradeoff and systematic deep-learning diagnosis."""
import hashlib
import copy
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/production_language_replay_evidence.py'))
HISTORY=runpy.run_path(str(ROOT/'report/language_replay_driver_evidence.py'))
CACHE_CONTRACT='diagnostics/local_language_cached_prefix_contracts_retry_20261003T021900Z.json'
CACHE_WORK='diagnostics/local_language_cached_prefix_resource_20261003T022600Z.json'
PROBE='diagnostics/local_deep_clipping_optimizer_probe_current_inputs_20261003T025400Z.json'
GROWTH='diagnostics/local_depth_growth_lineage_contracts_20261003T025000Z.json'
PLASTICITY='diagnostics/local_depth_growth_plasticity_probe_20261003T030400Z.json'
DEPTH_ROWS=[
    ('factorized',7,'dvs_native/curie_dvs_batched_p16d2pool2_factorized_s7_20261002T224500Z.json'),
    ('clip4',7,'dvs_native/curie_dvs_batched_p16d2pool2_clip4_s7_20261003T004500Z.json'),
    ('factorized',7,'dvs_native/curie_dvs_batched_p16d4pool2_factorized_s7_20261002T230500Z.json'),
    ('clip4',7,'dvs_native/curie_dvs_batched_p16d4pool2_clip4_s7_20261003T004500Z.json'),
    ('all-race',7,'dvs_native/curie_dvs_batched_p16d4pool2_leall_s7_20261002T230500Z.json'),
    ('factorized',8,'dvs_native/curie_dvs_batched_p16d4pool2_factorized_s8_20261002T230500Z.json'),
    ('clip4',8,'dvs_native/curie_dvs_batched_p16d4pool2_clip4_s8_20261003T004500Z.json'),
    ('all-race',8,'dvs_native/curie_dvs_batched_p16d4pool2_leall_s8_20261002T230500Z.json'),
    ('lr.006',7,'dvs_native/curie_dvs_batched_p16d4pool2_lr006_s7_20261003T004500Z.json'),
]


def history_read(read,path):
    # Publication copy only: preserve exact archived producer bindings.
    # Original JSON and frozen earlier report modules remain unchanged.
    r=copy.deepcopy(HISTORY['historical_read'](read,path))
    source=r.get('source_sha256',{})
    name='sleeping_machines/batched_episodes.py'
    digest='8f93f5f0ec60544cce8ddad3379113578ed5f6d90f17a1abf30edd0b9629c4c8'
    if source.get(name)==digest and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        source.pop(name)
        source[f'experiments/archive/frozen_sources/{digest}/batched_episodes.py']=digest
    return r


def verify(result):
    if result['status']!='completed':raise ValueError('Completed evidence required')
    for name,digest in result['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            historical=ROOT/'experiments/archive/frozen_sources'/digest/Path(name).name
            allowed={'ee3d8ada5648c9a152407d7b78850d37838537dfcb11387f06b462779298285c',
                     '8f93f5f0ec60544cce8ddad3379113578ed5f6d90f17a1abf30edd0b9629c4c8'}
            if digest not in allowed or not historical.is_file() or hashlib.sha256(historical.read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed diagnosis source '+name)


def load(read):
    d=dict(prior=BASE['load'](lambda p:history_read(read,p)),cache_contract=read(CACHE_CONTRACT),cache=read(CACHE_WORK),
        probe=read(PROBE),growth=read(GROWTH),plasticity=read(PLASTICITY),depth_rows=[])
    for key in ('cache_contract','cache','probe','growth','plasticity'):verify(d[key])
    if d['cache_contract']['contracts_passed']!=36 or d['cache']['contracts_passed']!=6:
        raise ValueError('All cached-prefix numerical/resource cases required')
    for variant,seed,path in DEPTH_ROWS:
        r=history_read(read,path);verify(r)
        if r['work']['fitting_targets']!=7872:raise ValueError('Same eight-pass fitting denominator required')
        d['depth_rows'].append((variant,seed,r))
    v=d['probe']['vectors']
    if hashlib.sha256((ROOT/v['path']).read_bytes()).hexdigest()!=v['sha256']:raise ValueError('Changed Adam vectors')
    r=d['plasticity']
    if hashlib.sha256((ROOT/r['artifact']).read_bytes()).hexdigest()!=r['artifact_sha256']:raise ValueError('Changed plasticity vectors')
    return d


def pages(d):
    pages=BASE['pages'](d['prior']);r=d['cache'];rows=[]
    for row in r['common_unit_ledger']:
        if row['targets']!=4:raise ValueError('No T4-to-T16 work projection')
        rows.append([row['family']+'/'+row['arm'],f"{row['whole_fit_unit_special_flops']/1e9:.6f}",
            f"{row['fit_unit_special_flops_per_target']/1e6:.6f}",
            f"{row['inference_unit_special_flops_per_target']/1e6:.6f}",
            str(row['shadow_lanes']),str(row['shadow_events'])])
    pages.append([('h1','Appendix B. Cached causal prefixes: equivalent credit with fewer operations'),
        ('table',(['Family/arm','Whole fit GF','Fit MF/target','Infer MF/target','Shadow lanes','Shadow events'],rows,[35,28,33,34,22,22])),
        ('p','SAME synthetic FOUR targets, p16/L8/H2/pool2 represented weights, nonempty '
         'private state, entering RNG and targetweighted Adam update. All forward/shadow '
         'backward, cache copies/stacking, normalization, clipping and Adam stages have '
         'complete operator accounting. Both native families keep32 available receivers, '
         '32 scored keys and16 selected updates per target. No quality/data comparison '
         'or T4-to-T16 FLOP projection. Inference uses selected values; cold native '
         'prefix accounting differs from AWS warm-prefix-plus-loss accounting.'),
        ('p','Full enumeration repeats the factual prefix for every forced alternative. '
         'Winner reuse already removes the factual winning shadow. The new causal cache '
         'also saves detached pre-token memory/arrivals/context and RNG, then evaluates '
         'only each losing route\'s remaining suffix. Same complete future loss returns '
         'and every race alternative remain. This changes execution work, not the '
         'conditional-credit objective or sixteen-token detachment boundary.'),
        ('p','Theory118\'s36 contracts cover private/shared p4/L8 U1/U2/U4, empty and '
         'nonempty memory, every causal snapshot and ALL losing suffixes against '
         'full-prefix shadows, EVERY gradient against old batched AND independent '
         'sequential enumeration, primal/private-state/RNG identity and exact pending '
         'gradient/Adam recovery. Theory119 adds productionT16 double EVERY-gradient '
         'equivalence and actual T4 complete-work admission for all three arms.'),
        ('small',f"Contracts {d['cache_contract']['wall_s']:.3f}s/{d['cache_contract']['max_rss_kb']}KiB; "
         f"resource campaign {r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. Snapshot tensors and "
         'RNG are charged as bytes; Python metadata, graph memory, traffic/RNG/energy '
         'remain separate. Total campaign fitting work is unknown, not zero. '
         'Original winner-reuse tight-coordinate failures and precision audit are retained.')])
    wallrows=[]
    for f in r['families']:
        med=f['t16_median_learning_wall_s']
        wallrows.append([f['family'],*[f'{med[k]:.6f}' for k in ('full','winner','cached')],
            f"{med['cached']/med['winner']:.3f}",
            str(next(x['snapshot_tensor_bytes'] for x in f['production_double_activity'] if x['arm']=='cached'))])
    pages.append([('h1','Appendix B. Cached-prefix prototype: CPU wall regression prevents promotion'),
        ('table',(['Family','Full seconds','Winner seconds','Cached seconds','Cached/winner','Double cache bytes'],wallrows,[22,29,31,31,28,33])),
        ('p','T16 FLOAT32 learning-step wall, three fresh serial replicates per arm with '
         'rotated order, SAME inputs/noise/model/Adam/warmup. The table reports medians; '
         'all individual times are saved. Timing includes accumulation, one equal '
         'gradient-vector diagnostic extraction, normalization, clipping and Adam; '
         'it excludes initialization/DEV and is not an energy measurement. Cache-byte '
         'column is the separate T16 DOUBLE admission, not peak float32 memory.'),
        ('p','The grouped cache is slower despite fewer counted T4 operations. '
         'AtT16 the original/reuse/cache shadow events are8192/4096/2176, while '
         'lanes are512/256/256. Cached suffix grouping expands sixteen batched '
         'shadow token iterations to136 smaller iterations. Repeated stacking, '
         'Python/operator launch overhead and smaller batches are plausible causes, '
         'not yet isolated by a profiler. Fewer event simulations do not guarantee '
         'lower wall, traffic or energy.'),
        ('p','Every float32 replicate\'s factual logits/private state/endRNG matches '
         'exactly across arms, global gradient error stays within3e-5 and actual '
         'Adam displacement error within.005. The production double comparison '
         'matches EVERY gradient to both older programs. These newly fixed global '
         'budgets do not replace Theory116\'s failed coordinate gate. No quality '
         'benchmark was trained or selected.'),
        ('p','Decision: retain exact winner reuse in the separately owned AWS10M '
         'matrix; do not promote this grouped cache. A compact executor for '
         'variable-length live shadow lanes is a future contract requiring '
         'per-lane chronology/noise identity. It must demonstrate actual wall '
         'and resource benefit before another long fit.'),
        ('small','Theory118/119. All six T4 audited updates, eighteen T16 timed updates '
         'and six double gradient fits are paid diagnostic work. Inference and '
         'counterfactual coverage remain unchanged. Negative wall findings stay '
         'beside positive arithmetic savings; ten-million-character quality remains pending.')])
    rows=[]
    for variant,seed,v in d['depth_rows']:
        w=v['work'];rows.append([f"D{v['args']['depth']} "+variant,str(seed),f"{v['final_fit_diagnostic']['nll']:.6f}",
            f"{v['final']['nll']:.6f}",f"{100*v['final']['accuracy']:.3f}",
            f"{w['whole_fit_unit_special_flops_estimate']/1e9:.3f}",
            f"{w['fit_unit_special_flops_per_target_estimate']/1e6:.3f}",
            f"{w['inference_unit_special_flops_per_target_estimate']/1e6:.3f}"])
    pages.append([('h1','Appendix B. Deep fitting diagnosis: looser clipping fails its prediction'),
        ('table',(['Depth/credit/cap','Seed','FIT32 NLL','DEV NLL','DEV %','Whole fit GF','Fit MF/target','Infer MF/target'],rows,[27,10,22,22,19,24,25,25])),
        ('p','Same p16/H2/pool2 fine20 temporal packets,984 FIT/192 subject-disjoint '
         'DEV gestures, eight passes/7872 fitting presentations/496 Adam updates, '
         'U16/lr.003 except marked lr.006. Factorized/all-race cap1, clip4 changes only the cap. '
         'All rows use the SAME whole-fit and per-target denominators; work '
         'includes all candidates/shadows/backward/normalization/clipping/Adam, '
         'first/last-window estimates. No official test or measured energy.'),
        ('p','Correction beside historical §408: clip4 WORSENS the saved FIT32 '
         'NLL in BOTH seeds; development accuracy/NLL change in mixed directions. '
         'The simple fitting-repair prediction fails on this diagnostic. '
         'FIT32 is the first32 fitting gestures at DEV-selected weights, '
         'not whole-training loss or a common final epoch. Newly completed '
         'D4 lr.006 also worsens FIT32 to.960, and D2 clip4 to.683 versus.576. '
         'These rule against the tested larger-step repairs, not every '
         'functional-step or optimizer-history explanation.'),
        ('p','All-race credit gives a positive seed8 result: DEV accuracy4.6875 '
         'points higher and NLL.072620 lower, with better FIT32. Seed7 fails '
         'to improve. Thus the old categorical statement that routing credit '
         'cannot explain deep fitting is too strong. Sampled8 is a distinct '
         'noisy estimator: with21events D4 has168 races, each sampled '
         'contribution scaled21; D6 scales31.5. Full replay costs roughly '
         '111 times factorized fitting here; useful occasional quality is '
         'not an efficient-learning advantage.'),
        ('small','Three independent audits: completed evidence/protocol, optimizer '
         'mechanics, and architecture/transport. Theory122 records the '
         'ranked hypotheses, primary-paper analogies and discriminating tests. '
         'Small elementary transport decay does not establish good complete '
         'recurrent/message Jacobian conditioning; large norms do not establish useful features.')])
    r=d['growth'];row=r['rows'][0]
    gainrows=[]
    for k,(old,new) in enumerate(zip(row['legacy_d6_layer_gains'],row['layer_gains'])):
        gainrows.append([str(k+1),f'{old:.9f}',f'{new:.9f}',
            'inherited D2' if k<2 else 'inherited D4' if k<4 else 'new D6'])
    pages.append([('h1','Appendix B. Progressive depth growth: preserve the actual parent computation'),
        ('table',(['D6 layer','Legacy gain','Preserved gain','Construction origin'],gainrows,[24,40,40,70])),
        ('p','Concrete bug: residual unit.gain is a Python float outside state_dict. '
         'Direct D2->D4 initializes the old layers at.353553 and new layers at.25. '
         'Legacy D4->D6 resets ALL old layers to.25, reducing the oldest two '
         'nonlinear amplitudes29.29% before fitting. Tensor-only checkpoint '
         'copying cannot preserve this part of the computation. Existing direct '
         'D2->D4 completed results remain valid for their original variant.'),
        ('p','The new sibling reconstructs ordinary, legacy-grown and new parents '
         'from source-checked lineage, then preserves their ACTUAL gain vector. '
         'Existing historical resets are reconstructed as historical resets. '
         'Every checkpoint/result binds ancestor bytes and records gains, '
         'including interrupted snapshots. Unknown construction, missing/cyclic '
         'lineage, changed sources and contradictory state/metadata refuse admission.'),
        ('p','Twelve completed contracts: direct D2->D4 bitwise old/new logits, '
         'EVERY gradient, persistent state and caller RNG; heterogeneous gains '
         'and old tensors retained at D6; artifact reload through D8; actual '
         'activated batched factory and interruption annotation; historical '
         'reset interpretation; invalid lineage rejection and parent/source '
         'immutability. Tiny synthetic numerical cases, no fitted quality.'),
        ('p','Closed new sigmoid gates remain at bias-20, a plasticity concern '
         'requiring actual gate/branch/update measurements. Channel and clock '
         'paths remain live. Fresh Adam, extra passes, positive race delays '
         'and shifted parent RNG remain confounds. The fix preserves gains; '
         'it does not claim exact parent-function preservation or useful new '
         'nonlinear features. A matched shallow continuation is still required.'),
        ('small',f"Theory121;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. Original "
         'growth driver/queues/results frozen. Pending external legacy D6 '
         'queue is affected; use the new sibling with a unique tag after '
         'its owner checks parent artifacts. No temporal/sparse mechanism '
         'removed; total contract work/traffic/energy unknown, not zero.')])
    r=d['probe'];rows=[]
    for c in r['cases']:
        caps={o['clip']:o for o in c['outcomes'] if o['lr_scale']==1.}
        label=('D2 fine' if c['depth']==2 else 'D4 coarse')+'/'+('fresh' if c['snapshot']=='fresh_first_step' else str(c['fit_offset']))
        a,b=caps[1.],caps[4.]
        rows.append([label,f"{c['preclip_norm']:.4f}",f"{a['actual_update_norm']:.5f}",
            f"{b['actual_update_norm']:.5f}",f"{a['descent_cosine']:.4f}/{b['descent_cosine']:.4f}",
            f"{a['anchor_loss_after']-a['anchor_loss_before']:+.5f}",
            f"{b['anchor_loss_after']-b['anchor_loss_before']:+.5f}"])
    pages.append([('h1','Appendix B. Actual Adam updates: clipping scale and moment history differ'),
        ('table',(['Protocol/FIT offset','Raw norm','Cap1 step','Cap4 step','Surrogate cosine1/4','Anchor delta1','Anchor delta4'],rows,[33,21,24,24,27,23,22])),
        ('p','Within each row, SAME frozen parameters, normalized actual driver '
         'gradient, Adam moments and FIT inputs. Step columns are full parameter '
         'displacement L2; anchor delta is NLL on fixed independent FIT96..111, '
         'not DEV/test. Three trained windows FIT0..15/16..31/32..47 and one '
         'fresh first step per protocol; six discarded cap/LR forks per window. '
         'Quarter LR and no-cap outcomes plus every per-block/vector are saved.'),
        ('p','Fresh Adam nearly cancels constant gradient rescaling: cap1/4 '
         'step norm differs.019% in D2/.190% in D4 despite substantial gradient '
         'shrinkage. With trained moments, cap4 changes direction and increases '
         'D4 step norm1.36-1.47x, rather than fourfold. Three D4 frozen FIT '
         'losses and anchors improve more at cap4. This changes CURRENT gradient '
         'weight against cap1-trained moments; it is not a history trained '
         'under cap4. The eight-pass cap4 failures remain valid.'),
        ('p','Protocols differ: D2 fine20 seed6/eight-pass native teacher versus '
         'D4 coarse4 seed7/four-pass factorized pilot. No causal cross-depth '
         'comparison. D2 preprocessing hash reproduces; D4 ORIGINAL statistic '
         'hash differs from recomputation on this host, explicitly recorded. '
         'D4 uses controlled CURRENT FIT inputs at genuine paired online '
         'weights/moments. Exact original-input replay remains an artifact gap. '
         'Its frozen original batched-kernel bytes are archived and loaded.'),
        ('small',f"Theory120/122;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB, five checks. "
         'Adam fresh-step formula and constant-history scaling pass. '
         'Directional cosine uses the declared SURROGATE, not exact hard '
         'branch derivatives; finite loss is observed. Four failed attempts '
         'remain archived. Total diagnostic FLOPs/traffic/energy unknown, '
         'not zero; no cap schedule or benchmark improvement established.')])
    r=d['plasticity'];rows=[]
    for arm in r['rows']:
        blocks={b['block']:b for b in arm['parameter_blocks']}
        for layer in arm['before']['layers'][2:]:
            idx=layer['depth'];g=blocks[f'layer{idx}/units/gate'];o=blocks[f'layer{idx}/units/output']
            rows.append([str(int(arm['initial_gate_bias'])),str(idx+1),f"{layer['gate']['mean']:.3g}",
                f"{100*layer['candidate_delta']['nonzero_fraction']:.3f}",
                f"{g['actual_FP32_displacement_l2']:.3g}",f"{o['actual_FP32_displacement_l2']:.3g}",
                f"{100*g['active_gradient_below_eps_fraction']:.2f}"])
    pages.append([('h1','Appendix B. Added message branches: float32 silence and Adam epsilon'),
        ('table',(['New bias','Layer','Mean gate','Visible delta %','Gate step L2','Output step L2','Gate active<eps %'],rows,[20,14,25,28,28,28,31])),
        ('p','Prespecified -20/-8/-4/0 gate-only interventions at identical grown '
         'p16/H2/pool2 D4 from a saved trained D2 parent. SAME sixteen FIT '
         'gestures0..15, exact fine transform, fixed common race noise, one '
         'fresh actual normalized clip1 Adam.003 update per arm. Visible delta '
         'is the fraction of candidate FP32 proposals differing from incoming '
         'content. Steps are actual stored-parameter displacement norms.'),
        ('p','Legacy -20 gates average roughly2.2e-9. About99.7% of added '
         'candidate message contributions round away when added to incoming '
         'float32 values. ALL active gate/output gradients are below Adam eps '
         'after clipping; mean gate updates receive only about.007% of the '
         'unattenuated fresh sign step. Their aggregate displacement is about '
         '1.4-1.5e-5, versus.097 at -4. This directly supports a practical '
         'nonlinear-message plasticity problem in this growth construction.'),
        ('p','Added input maps remain live: -20 raw gradient norms.046/.139 '
         'and Adam displacement about.096 per layer, via persistent memory '
         'key/timing paths. Channel mixes and clocks also remain active. '
         'Therefore added depth is not entirely frozen, and its old accuracy '
         'gain cannot be attributed exclusively to either old layers or '
         'new channels without lesion/continuation controls.'),
        ('p','At -4 all measured candidate contributions are visible and '
         'gate/output steps nearly escape epsilon attenuation. This is a '
         'GATE-ONLY counterpart to the other host\'s pending §410; its full '
         'scratch initialization/transport/fit protocol differs. All four '
         'one-step FIT losses decrease, including -20. No DEV/test, tuning '
         'or benchmark advantage, and no inference-cost gain established.'),
        ('small',f"Theory123/122;{r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. All four "
         'instrumented logits/EVERY gradient bitwise nest plain computation; '
         'fresh-step formulas, parent/source/RNG/kernel immutability pass. '
         'Four independent updates/64 target exposures plus verification '
         'work; total FLOPs/traffic/energy unknown, not zero. Reuse the '
         'owner\'s initialization comparisons before proposing another large fit.')])
    return pages
