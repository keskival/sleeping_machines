"""Empirical total trained native batch variance and actual warm-Adam forks."""
import hashlib
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'report/conditional_content_evidence.py'))
RESULT='diagnostics/local_trained_content_batch_variance_20261003T052200Z.json'
def load(read):
    d=dict(prior=BASE['load'](read),variance=read(RESULT));r=d['variance']
    if r['status']!='completed' or r['contracts_passed']!=5:raise ValueError('Completed trained-state contracts required')
    for name,digest in r['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed trained content source '+name)
    for name in ('parent','snapshot','vectors'):
        if hashlib.sha256((ROOT/r[name]).read_bytes()).hexdigest()!=r[name+'_sha256']:raise ValueError('Changed trained-state '+name)
    return d
def pages(d):
    pages=BASE['pages'](d['prior']);r=d['variance'];rows=[]
    for label,item in r['estimates'].items():
        rows.append([label,f"{item['factual']['sample_covariance_trace']:.9g}",f"{item['joint']['sample_covariance_trace']:.9g}",f"{item['joint_to_factual_variance_ratio']:.6f}",f"{item['leave_one_pair_out_ratio_range'][0]:.4f}..{item['leave_one_pair_out_ratio_range'][1]:.4f}"])
    dec=[]
    for label,x in r['decomposition'].items():
        dec.append([label,f"{x['content_trace']:.6f}",f"{x['choice_trace']:.6f}",f"{x['twice_content_choice_cross_trace']:.6f}",f"{x['total_trace']:.6f}"])
    pages.append([('h1','Appendix B. Trained batch credit: empirical variance and real Adam moments'),
        ('p','Recover the previous choice-only native p4/D4/H2/pool2 '
         'fit with its EXACT twelve steps,48 presentations, FIT0..23, '
         'normalization, sites/noise and clip1 Adam.003. Every saved '
         'per-step factual/conditional loss, final FIT/anchor prediction '
         'and parameter-group movement agrees. Reconstructed weights '
         'AND actual Adam history are archived and hashed. This is '
         'reconstruction, not another benchmark or seed.'),
        ('table',(['Quantity','Factual content trace','Conditional content trace','Joint/factual','Leave-one-pair-out'],rows,[28,39,39,28,40])),
        ('table',(['Credit','Content trace','Choice trace','2x cross trace','Total raw trace'],dec,[34,35,35,35,35])),
        ('p',f"Here content variance falls {100*(1-r['decomposition']['joint']['content_trace']/r['decomposition']['factual']['content_trace']):.2f}%, but choice credit alone is {100*r['decomposition']['factual']['choice_trace']/r['decomposition']['factual']['total_trace']:.2f}% of total raw trace. Total raw variance falls only {100*(1-r['estimates']['raw']['joint_to_factual_variance_ratio']):.3f}%, while clipped variance rises and real-Adam variance falls only {100*(1-r['estimates']['update']['joint_to_factual_variance_ratio']):.3f}%. Exposure helps a small term in this restricted sampled-one-site teacher."),
        ('p','SAME trained state and original FIT0..3, B4,16 paired '
         'prespecified fresh-noise/uniform-site draws. Extract factual '
         'content C, conditional live content Cbar and the SAME R-scaled '
         'choice A. Shared race noise across episodes is retained. Raw '
         'vectors are actual float32 C+A versus Cbar+A. Full parameter '
         'covariance is measured; content/choice cross terms are signed. '
         'Zero gradients and disconnected None gradients remain distinct.'),
        ('p','Each pair forks the SAME recovered warm moments through '
         'actual clip1 Adam;32 updates are discarded. Raw/clipped/update '
         'vectors and parameter-group statistics are retained. This '
         'tests the nonlinear optimizer, rather than equating its update '
         'with a raw gradient or fresh sign step. Sample traces and '
         'leave-one-pair-out ranges are descriptive;16 draws do not '
         'provide guaranteed intervals or exact expected-risk gradients.'),
        ('p','Contracts compare EVERY parameter in double with isolated '
         'native objectives, then float32 gradients and real updates; '
         'serialized next-update recovery, causal fixed inference, source/'
         'data/state/moment/RNG preservation and operator coverage pass. '
         'No episode-noise independence assumption is inserted into '
         'the actual batch. Earlier per-episode conditional-mean theorem '
         'and its batch-covariance limitation remain explicit.'),
        ('small',f"Theory132; {r['wall_s']:.3f}s/{r['max_rss_kb']}KiB. "
         'One tiny restricted trained state, not all depths or the AWS '
         'language failure. No heldout nomination, convergence guarantee, '
         'larger fit or superiority claim.')])
    rows=[];ratio=r['work'][1]['actual_step_unit_special_flops']/r['work'][0]['actual_step_unit_special_flops'];cost=[]
    for item in r['work']:
        rows.append([item['arm'],f"{item['actual_step_unit_special_flops']/1e9:.9f}",f"{item['fit_unit_special_flops_per_target']/1e6:.6f}",f"{item['inference_unit_special_flops_per_target']/1e6:.6f}"])
    for label,x in r['estimates'].items():cost.append([label,f"{x['joint_to_factual_variance_ratio']:.6f}",f"{ratio:.6f}",f"{ratio*x['joint_to_factual_variance_ratio']:.6f}"])
    pred=[]
    for arm in ('factual','joint'):
        for split in ('fit','anchor'):
            changes=[x[arm]['predictions'][split+'_nll']-r['baseline'][split+'_nll'] for x in r['draws']]
            pred.append([arm,split,f"{r['baseline'][split+'_nll']:.6f}",f"{sum(changes)/len(changes):+.6f}",f"{sum(x<0 for x in changes)}/16",f"{min(changes):+.5f}..{max(changes):+.5f}"])
    pages.append([('h1','Appendix B. Trained content forks: paid work and fixed FIT predictions'),
        ('table',(['Credit','Isolated step GF','Fit MF/presentation','Infer MF/target'],rows,[44,42,44,44])),
        ('table',(['Variance quantity','Joint/factual trace','Step work ratio','Variance x work ratio'],cost,[44,42,44,44])),
        ('table',(['Credit','Split','Initial NLL','Mean delta NLL','Improved draws','Delta range'],pred,[27,22,27,30,29,39])),
        ('p','Only TWO isolated actual fitting windows, four presentations '
         'each, are charged in the work table: normalized forward/loss, '
         'all forced branches, live backward where applicable, clipping '
         'and restored historical Adam. Native inference uses the SAME '
         'twelve targets. One special-function evaluation counts as one. '
         'This is not the whole ensemble campaign or a new whole fit. '
         'The prior page retains complete48-presentation fits and saved '
         'full-data references with consistent units/denominators.'),
        ('p','Same16 available private receivers,8 selected state updates '
         'and16 key scores per event;21 events gives168 selected/336 '
         'scored per target. Inference structure and capacity are identical. '
         'The live losing-message derivative changes learning work, not '
         'available state or candidate discovery support. No inference '
         'saving follows from this estimator.'),
        ('p','All32 discarded updates are evaluated on fixed FIT0..3 and '
         'FIT24..31 anchors under common inference noise314159;384 '
         'prediction-target evaluations plus baseline12. Adjacent FIT '
         'anchors are not IID, DEV or test. Every outcome is saved; none '
         'selects a configuration. Mean loss changes from one warm update '
         'are local effects, not generalization or sustained learning.'),
        ('p','Reconstruction48 presentations/12 updates; ensemble64 '
         'factual targets,128 full shadow lanes/2688 events and48 '
         'component reverse calls; extra two isolated paid updates and '
         'two recovery forks, plus admissions/evaluation. Total campaign '
         'FLOPs/traffic/energy remain unknown, not zero. Variance-times-work '
         'is a declared optimization heuristic, not a convergence theorem '
         'for clipping, Adam or nonlinear sparse temporal models.'),
        ('small','Preserve the tiny paired-fit negative result and these '
         'trained-state measurements together. No long content fit is '
         'admitted. Priority remains completed AWS corrected full replay10M '
         'and matched teachers/controls; other-host live-gate and calibrated '
         'reset-segment language runs use distinct protocols.')])
    return pages
