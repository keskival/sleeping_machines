"""Completed cross-host DVS evidence; shared units and historical scope retained."""
import hashlib
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
DIAG='diagnostics/'
FILES=[
    'aws_coarse_native_20261002T212600Z_analysis.json',
    'aws_coarse_native_replication_20261002T212900Z_analysis.json',
    'aws_full_coarse_20261002T213100Z_analysis.json',
    'aws_coarse_readout_probe_20261002T213700Z.json',
    'aws_quadratic_native_20261002T214600Z_analysis.json',
    'local_dvs_producer_held_decoder_20261002T215100Z.json',
]


def load(read):
    data=[read(DIAG+name) for name in FILES]
    for summary in data:
        for row in summary['rows']:
            name=row.get('result',row.get('native'))
            if name is None:continue
            digest=row.get('result_sha256')
            if digest and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Changed cross-host DVS parent: '+name)
            parent=read(str(Path(name).relative_to('experiments/results')))
            if 'accuracy' in row:
                if abs(parent['final']['accuracy']-row['accuracy'])>1e-12 or abs(parent['final']['nll']-row['nll'])>1e-12:
                    raise ValueError('Cross-host summary differs from completed result')
    return data


def quality_work(row,label):
    return [label,f"{100*row['accuracy']:.2f}",f"{row['nll']:.4f}",
        *[f"{row[k]:.6f}" for k in ('whole_fit_gflops','fit_mflops_per_target','inference_mflops_per_target')]]


def pages(data):
    screen,replication,full,probe,quadratic,held=data
    labels={'fine':'20 bins/.05','coarse_fastclock':'4 bins/.05','coarse_matchedclock':'4 bins/.25'}
    columns=['Native packet/clock','Accuracy %','Dev NLL','Whole fit GF est.','Fit MF / target','Infer MF / prefix']
    widths=[43,23,23,29,28,27]
    result=[]
    rows=[quality_work(r,f"s{r['seed']} "+labels[r['variant']]) for summary in (screen,replication) for r in summary['rows']]
    result.append([('h1','Appendix B. AWS coarse temporal screens: two seeds pass'),
        ('table',(columns,rows,widths)),
        ('p','Same integrated p16/L2/H2/pool2 core, 256 fitting gestures, 192 subject-disjoint '
         'development gestures, four passes, 1,024 fitting presentations and U16. Coalescing '
         'five 50ms count packets into each 250ms packet preserves total causal counts and '
         'the 1s query but removes timing within each quarter second. Learned temporal '
         'computation, races, separate keys/values, sparse addressed writes and counterfactual '
         'credit remain. The two coarse arms differ only in clock initialization.'),
        ('p','Both coarse arms pass the prespecified screen in seeds6 and7: at least .02 NLL '
         'improvement, at most 1 percentage point accuracy decline and at most .50 fitting '
         'work ratio. Work ratio .239978 means 76.002% less counted fitting work. Events '
         '21,504 to 5,120; key scores 172,032 to 40,960; writes 86,016 to 20,480. Eight '
         'available receivers and four selected writes per event in every arm.'),
        ('p','Seed6 one-thread AWS job walls are 42.846/14.031/14.046s, under the authorized '
         'three-slot scheduler. These are observations, not universal hardware speedups. '
         'The subsequent full-data matrix is complete on the next page; the earlier pending '
         'replication and full-data statuses are superseded by completed evidence.'),
        ('small','Exploratory reused development set; no independent confirmation or control '
         'advantage. All passes, candidate values, backward/Adam/clip are charged in consistent '
         'arithmetic plus unit-special estimates. Whole-job wall includes preprocessing/evaluation; '
         'NumPy preprocessing FLOPs, traffic and energy remain unknown. Both completed analysis '
         'files and the verbatim pre-integration AWS appendix are retained in version control.')])
    rows=[quality_work(r,f"s{r['seed']} "+labels[r['variant']]) for r in full['rows']]
    result.append([('h1','Appendix B. Full coarse matrix: positive means, failed all-seed gates'),
        ('table',(columns,rows,widths)),
        ('p','Nine completed fits: same984 fitting/192 development gestures, eight passes, '
         '7,872 fitting presentations per run. Mean fine accuracy/NLL 60.764%/1.083660; '
         '4-bin/.05 64.062%/1.011241; 4-bin/.25 65.972%/.956220. Coarse fitting work is '
         '4.218015 versus 17.573520 GFLOPs per fit: ratio .240021. Every per-target column '
         'uses the same denominator across models; no whole-fit/per-target unit mixing.'),
        ('p','Both frozen all-seed gates FAIL. Seed8 fine 68.75%/.898591 has lower NLL '
         'than 4-bin/.05 60.417%/1.028562 and 4-bin/.25 69.271%/.920744. Diagnostic '
         'crossed seed/development-user intervals include zero. Preserve positive mean '
         'quality and work savings alongside this failure; no unchanged extension follows.'),
        ('small','Original stronger local references remain valid: native65.10%/.963161 and '
         'clock66.15%/1.041987. They have different fitting implementations and work totals '
         'and are not replaced by the weaker matched fine arm. Same8 receivers, four writes '
         'per event, 21 versus5 events per query. Saved full-coarse analysis retains every '
         'curve, activity ledger, result digest, wall/RSS and uncertainty scope. No official test.')])
    controls=[[f"Raw RBF {r['bins']} bins",f"{100*r['accuracy']:.2f}",f"{r['nll']:.4f}",
        'Unknown','Unknown','Unknown'] for r in full['strong_controls']]
    result.append([('h1','Appendix B. Strong coarse controls and frozen readout evidence'),
        ('table',(columns,controls,widths)),
        ('table',(['Frozen encoder/access','Mean accuracy %','Mean dev NLL','Nomination gate'],
            [['Initial/query32','61.632','1.036019','Diagnostic'],
             ['Initial/all resident state','68.750','.886478','Diagnostic'],
             ['Trained/query32','69.097','.856456','Pass'],
             ['Trained/all resident state','72.743','.829107','FAIL']],[57,36,35,45])),
        ('p','Raw controls use984 fitting/192 development examples and fitting-user GroupKFold '
         'selection. The4-bin RBF result77.604%/.686661 is a strong completed reference under '
         'that protocol; the earlier local RBF73.44%/.706478 uses different selection and '
         'remains visible. Solver FLOPs and total fit/inference work are unknown, not zero.'),
        ('p','All three matched-clock native encoders and their initial reservoirs are frozen. '
         'Query RBF probes improve native mean65.972%/.956220 by3.125 points/.099764 NLL; '
         'individual NLL gains .056455/.163228/.079610 pass the context nomination gate. '
         'All resident reads add128 memory values, eight ages and eight occupancy flags. '
         'Their extra mean .027349 NLL gain misses the .05 resident gate.'),
        ('p','These are information/readout diagnostics. Dense resident access does not '
         'demonstrate sparse dormant-value retrieval. Conditional fitting-only decoder '
         'selection shares label-trained encoders; it is not producer-cross-fitted validation. '
         'The next integrated quadratic head fails, as recorded on the next page.'),
        ('small','Original native fits4.218015GF each, six1,176-prefix replays, 72 fold fits '
         'and12 refits retained. Probe53.605s/516,444KiB; state/winner/probability contracts '
         'pass. Replay solver/materialization/traffic/energy totals remain unknown. '
         'Full per-seed readout outcomes and costs are preserved in AWS_COARSE_READOUT_FINDINGS_20261002.md.')])
    result.append([('h1','Appendix B. Integrated quadratic head: contracts pass, quality fails'),
        ('table',(columns,[quality_work(r,r['variant']) for r in quadratic['rows']],widths)),
        ('p','Same4-bin/.25-clock native p16/L2/H2/pool2, 256 fitting/192 development '
         'gestures, seed6, four passes and1,024 fitting presentations. A standard local '
         'degree2 query residual starts at zero: initial logits/RNG and every old gradient '
         'nest the affine model exactly. Six numerical/deep-gradient/recovery/accounting '
         'contracts and24fit/8dev/two-pass learning smoke pass before this screen.'),
        ('p','The quality gate FAILS: NLL worsens .166618 and accuracy declines .521 '
         'percentage point. Fitting ratio1.083617 and inference ratio1.458782 satisfy '
         'resource admission but do not rescue prediction quality. There is no unchanged '
         'seed7/full-data/epoch extension and no replacement of the leading result.'),
        ('p','The5808 added head weights bring parameters from15,523 to21,331. Eight '
         'available receivers, candidate keys/values and four selected writes per event '
         'are unchanged. Every one of the five sequential readout calls is charged. '
         'Physical time, hard races, persistent addressed state, separate keys/values and '
         'counterfactual learning remain; positive frozen RBF information alone does not '
         'prove this coupled encoder/head optimizer can exploit it.'),
        ('small','Full-coarse seed8 failure and initial-reservoir/readout controls remain '
         'beside this negative result. The AWS team owns the separately frozen affine versus '
         'polynomial convex-fit diagnostic. No pending cell is reported as evidence; solver '
         'work, preprocessing FLOPs, traffic and energy remain separately scoped.')])
    quality=[];work=[]
    for r in held['rows']:
        for rule in ('conditional_cv','producer_held'):
            q=r['development'][rule];s=r['selections'][rule]
            quality.append([str(r['seed']),rule.replace('_',' '),f"{s['nominal_C_for984']:.1f}",
                f"{100*q['accuracy']:.2f}",f"{q['nll']:.6f}"])
        work.append([str(r['seed']),*[f"{r[k]:.6f}" for k in ('known_encoder_fit_gflops_estimate',
            'known_feature_replay_gflops_estimate','known_validation_replay_gflops_estimate')],'Unknown'])
    result.append([('h1','Appendix B. Producer-held selection: both rules agree'),
        ('table',(['Seed','Selection rule','C at984','Accuracy %','Dev NLL'],quality,[18,55,27,36,37])),
        ('table',(['Seed','Original core fit GF','Feature replay GF','Port-check replay GF','Solver/grid GF'],work,[18,40,40,40,35])),
        ('p','Use the two saved256-fit native pilot encoders after four fixed passes, '
         'without development checkpoint selection. Conditional three-fold decoder CV '
         'uses producer-seen fitting labels. Producer-held selection fits on those256 '
         'examples and scores728 fitting examples whose labels the producer never saw. '
         'Both rules choose the strongest nominal C.1 in both seeds; no selection '
         'disagreement or causal selection failure is demonstrated here.'),
        ('p','Three mean-L2 coefficients are invariant across fold sizes: lambda=1/(984C), '
         'solver C=1/(n lambda). The selected affine heads are then refitted on all984 '
         'fitting examples. Final54.17%/1.222593 and53.65%/1.177975 remain weak. Because '
         'the new heads see984 labels versus the original pilots256, this is neither '
         'an equal-data benchmark nor evidence of practical advantage.'),
        ('p','Four contracts include a constructed feature-selection confidence counterexample, '
         'sample-size invariant regularization, reproduction of fixed-pass native probabilities '
         'and folding scaled affine coefficients into the original native head. Every '
         'non-head parameter is bitwise preserved; batched and serial native probabilities '
         'match. Parameter count and the inference architecture remain unchanged.'),
        ('small','Theory96. The mathematical counterexample establishes a possible failure, '
         'not its occurrence in these pilots. Original encoder fits, feature replay and '
         'head-port verification replay are separate costs; solver/transformation/traffic/ '
         'energy work remains unknown. Same reused192 development examples; no official test '
         'or new main model. No guessed global regularization fit is admitted from this result.')])
    return result
