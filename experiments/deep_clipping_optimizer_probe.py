"""Actual-moment clipping forks on frozen native models; no benchmark refit."""
import argparse
import copy
import json
import importlib.util
from pathlib import Path
import resource
from types import SimpleNamespace
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_native_benchmark as N
import dvs_clock_calibrated_benchmark as C
import dvs_batched_le_benchmark as BL
import aws_coarse_native as K

PARENTS = [
    ('fine_d2_teacher', 'experiments/results/dvs_native/local_dvs_clock_full_20261002T153000Z.json'),
    ('coarse_d4_factorized', 'experiments/results/dvs_native/aws_deep_replay_20261002T234200Z_d4_factorized_pilot_s7.json'),
]


def flat(values):
    return torch.cat([v.detach().double().reshape(-1) for v in values])


def group(name):
    bits = name.split('.')
    if bits[0] in ('units', 'queries', 'channel_mix'): return bits[0] + '.' + bits[1]
    return bits[0]


def logits(model, rows, seed, factorized):
    model.eval()
    if factorized: return BL.batched_logits(model, rows, seed)
    return torch.stack([N.predict(model, row, seed, False)[0] for row in rows])


def score(model, rows, seed, factorized):
    with torch.no_grad():
        z = logits(model, rows, seed, factorized)
        return z, float(F.cross_entropy(z, torch.tensor([r['target'] for r in rows])))


def gradient(model, optimizer, rows, a, epoch, factorized):
    captured = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = optimizer.step
    def clip(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        captured.append([None if p.grad is None else p.grad.detach().clone() for p in parameters])
        return original_clip(parameters, max_norm, **kwargs)
    torch.nn.utils.clip_grad_norm_ = clip
    optimizer.step = lambda: None
    rng = torch.get_rng_state().clone()
    try:
        stats = (BL.train_window if factorized else N.train_window)(model, optimizer, rows, a, epoch, False)
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        optimizer.step = original_step
    assert torch.equal(rng, torch.get_rng_state()), 'caller RNG changed'
    assert len(captured) == 1, 'exactly one post-normalization clip required'
    raw = captured[0]
    for p, g in zip(model.parameters(), raw): p.grad = None if g is None else g.clone()
    assert stats['targets'] == len(rows)
    assert all(g is None or bool(torch.isfinite(g).all()) for g in raw)
    return raw, stats


def restore(a, weights, moments):
    model = C.make_model(a)
    model.load_state_dict(copy.deepcopy(weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    if moments is not None: optimizer.load_state_dict(copy.deepcopy(moments))
    return model, optimizer


def history_contract():
    reference = torch.tensor([.2, -.3, .7], dtype=torch.float64)
    models = [torch.nn.Parameter(reference.clone()) for _ in range(2)]
    optimizers = [torch.optim.Adam([p], lr=.003) for p in models]
    maximum = 0.
    for t in range(12):
        g = torch.tensor([.1 + .02 * t, -.3 + .01 * t, .15 * (-1)**t], dtype=torch.float64)
        for p, o, c in zip(models, optimizers, (1., .125)):
            p.grad = c * g; o.step()
        maximum = max(maximum, float((models[0] - models[1]).detach().abs().max()))
    assert maximum < 1e-7
    return dict(steps=12,constant_scale=.125,maximum_parameter_difference=maximum,
        scope='constant entire gradient-history scaling; epsilon retained, not current-only rescaling')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True); args = parser.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (args.tag + '.json')
    assert not out.exists(); torch.set_num_threads(1)
    started = time.perf_counter(); vectors = {}; cases = []; checks = []; preprocessing=[]
    # Resolve BOTH protocols before expensive gradient forks; never guess a
    # checkpoint's preprocessing from its architecture name.
    for label,path in PARENTS:
        r=json.loads((ROOT/path).read_text())
        ck=torch.load(ROOT/path.replace('.json','.progress.pt'),weights_only=False,map_location='cpu')
        settings=SimpleNamespace(**r['args'])
        _,_,info=(K.load if label.endswith('factorized') else N.load)(settings)
        differences={key:[ck['data'].get(key),info.get(key)] for key in ck['data'].keys()|info.keys()
                     if ck['data'].get(key)!=info.get(key)}
        if differences:
            # Keep this detected protocol gap explicit. This is a controlled
            # current-input fork, not exact replay of the historical input bytes.
            assert label=='coarse_d4_factorized' and differences=={'transform_sha256':[
                'a0496fe9be7efb322d649b18432195e628654d577c929ea81d7ab8cdd1ec4f06',
                '3fefa1805603bb1fb40b82c53e633a4600089e63f02efa71542772f1bc98480a']}, differences
        preprocessing.append(dict(label=label,original_data=ck['data'],current_data=info,differences=differences,
            exact_historical_transform_reproduced=not bool(differences),
            scope='Source/data artifact hashes identical; coarse statistic-byte hash differs. No original-input equivalence claim.'))
    history = history_contract(); checks.append('constant-history double Adam scaling contract')
    for label, path in PARENTS:
        result = json.loads((ROOT / path).read_text()); assert result['status'] == 'completed'
        ckpath = ROOT / path.replace('.json', '.progress.pt')
        checkpoint = torch.load(ckpath, weights_only=False, map_location='cpu')
        a = SimpleNamespace(**result['args']); factorized = label.endswith('factorized')
        for name, digest in checkpoint['source_sha256'].items():
            if N.sha(ROOT / name) != digest:
                archived = ROOT / 'experiments/archive/frozen_sources' / digest / Path(name).name
                assert archived.is_file() and N.sha(archived) == digest, 'unresolved historical source ' + name
        if factorized:
            kernel='sleeping_machines/batched_episodes.py'
            digest=checkpoint['source_sha256'][kernel]
            archived=ROOT/'experiments/archive/frozen_sources'/digest/'batched_episodes.py'
            if N.sha(ROOT/kernel)!=digest:
                spec=importlib.util.spec_from_file_location('sleeping_machines._frozen_clip_probe_episodes',archived)
                historical=importlib.util.module_from_spec(spec); spec.loader.exec_module(historical)
                BL.batched_logits=historical.batched_logits
        fit, _, info = (K.load if factorized else N.load)(a)
        assert info==next(p['current_data'] for p in preprocessing if p['label']==label)
        epoch = min(checkpoint['cursor']['epoch'], a.epochs)
        seed = 100000 + a.seed + 10000 * epoch
        anchor = fit[96:112]; assert len(anchor) == 16
        snapshots = [('online_actual_moments', checkpoint['online_model'], checkpoint['optimizer'], [0,16,32])]
        fresh = C.make_model(a)
        snapshots.append(('fresh_first_step', copy.deepcopy(fresh.state_dict()), None, [0]))
        for snapshot, weights, moments, offsets in snapshots:
            for offset in offsets:
                rows = fit[offset:offset+16]; assert len(rows) == 16
                model, optimizer = restore(a, weights, moments)
                before = {n:p.detach().clone() for n,p in model.named_parameters()}
                initial_moments = copy.deepcopy(optimizer.state_dict())
                raw, stats = gradient(model, optimizer, rows, a, epoch, factorized)
                assert all(torch.equal(before[n], p) for n,p in model.named_parameters())
                # A no-op optimizer interception must preserve actual moment tensors.
                for key, state in initial_moments['state'].items():
                    for n, value in state.items():
                        current = optimizer.state_dict()['state'][key][n]
                        assert torch.equal(value, current) if torch.is_tensor(value) else value == current
                z0, l0 = score(model, rows, seed, factorized)
                za0, la0 = score(model, anchor, seed, factorized)
                gvec = flat([torch.zeros_like(p) if g is None else g for p,g in zip(model.parameters(),raw)])
                norm = float(gvec.norm()); assert norm > 0
                prefix = f'{label}_{snapshot}_{offset}'; vectors[prefix+'_gradient'] = gvec.numpy()
                outcomes = []
                for cap in (1., 4., None):
                    for lr_scale in (1., .25):
                        fork, opt = restore(a, weights, moments)
                        for p,g in zip(fork.parameters(),raw): p.grad = None if g is None else g.clone()
                        if cap is not None: torch.nn.utils.clip_grad_norm_(fork.parameters(), cap, error_if_nonfinite=True)
                        clipped = flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in fork.parameters()])
                        for param_group in opt.param_groups: param_group['lr'] *= lr_scale
                        opt.step()
                        delta = flat([p-before[n] for n,p in fork.named_parameters()]); assert bool(torch.isfinite(delta).all())
                        blocks = {}
                        for (name,p), g in zip(fork.named_parameters(),raw):
                            item = blocks.setdefault(group(name),dict(gradient_squared_norm=0.,update_squared_norm=0.,weight_squared_norm=0.))
                            item['gradient_squared_norm'] += float((torch.zeros_like(p) if g is None else g).double().square().sum())
                            item['update_squared_norm'] += float((p.detach()-before[name]).double().square().sum())
                            item['weight_squared_norm'] += float(before[name].double().square().sum())
                        z1,l1 = score(fork,rows,seed,factorized); za1,la1 = score(fork,anchor,seed,factorized)
                        kl = float(F.kl_div(za1.double().log_softmax(-1),za0.double().softmax(-1),reduction='batchmean'))
                        if moments is None:
                            k = min(1., cap/(norm+1e-6)) if cap is not None else 1.
                            eta = a.lr*lr_scale
                            expected = -eta*gvec/(gvec.abs()+1e-8/k)
                            # Packet-calibrated frequencies can exceed 10: storage ULPs
                            # are larger than a universal 2e-7 absolute tolerance.
                            # Bound arithmetic/storage rounding coordinate by coordinate.
                            wvec = flat(before.values())
                            bound = 4*torch.finfo(next(fork.parameters()).dtype).eps*wvec.abs().clamp_min(1.)
                            assert bool(((delta-expected).abs() <= bound).all())
                        vector_key = prefix+f'_cap{cap}_lr{lr_scale}_delta'; vectors[vector_key]=delta.numpy()
                        outcomes.append(dict(clip=cap,lr_scale=lr_scale,postclip_norm=float(clipped.norm()),
                            clip_factor=float(clipped.norm())/norm,actual_update_norm=float(delta.norm()),
                            gradient_update_dot=float(gvec@delta),
                            descent_cosine=float(-(gvec@delta)/(gvec.norm()*delta.norm()).clamp_min(1e-300)),
                            fitting_loss_before=l0,fitting_loss_after=l1,
                            anchor_loss_before=la0,anchor_loss_after=la1,anchor_prediction_kl=kl,
                            anchor_argmax_changes=int((za0.argmax(-1)!=za1.argmax(-1)).sum()),
                            parameter_blocks=blocks,update_vector_key=vector_key))
                cases.append(dict(parent=path,parent_sha256=N.sha(ROOT/path),checkpoint=str(ckpath.relative_to(ROOT)),
                    checkpoint_sha256=N.sha(ckpath),label=label,snapshot=snapshot,
                    cursor=checkpoint['cursor'],depth=a.depth,fit_offset=offset,targets=16,
                    anchor_fit_indices=list(range(96,112)),training_noise_seed=seed,
                    preclip_norm=norm,nonempty_Adam_states=len(initial_moments['state']),
                    raw_gradient_vector_key=prefix+'_gradient',outcomes=outcomes,
                    scope='Within-checkpoint alternative one-step forks, discarded; no depth-matched causal comparison'))
        checks.append(label+': actual normalized driver gradient, paired online moments, six finite forks and caller RNG verified')
        checks.append(label+': fresh actual-gradient Adam formula matches all six finite forks')
    names = ['experiments/deep_clipping_optimizer_probe.py','experiments/theory/120_deep_clipping_optimizer_probe.md',
        'experiments/dvs_batched_le_benchmark.py','sleeping_machines/batched_episodes.py','experiments/aws_coarse_native.py']
    sources={**N.sources(),**C.sources(),**{n:N.sha(ROOT/n) for n in names}}
    for label,path in PARENTS:
        parent=json.loads((ROOT/path).read_text())
        for name,digest in parent['source_sha256'].items():
            if N.sha(ROOT/name)!=digest:
                archived=ROOT/'experiments/archive/frozen_sources'/digest/Path(name).name
                sources[str(archived.relative_to(ROOT))]=digest
    artifact=out.with_suffix('.vectors.npz'); assert not artifact.exists(); np.savez_compressed(artifact,**vectors)
    report=dict(status='completed',args=vars(args),contracts_passed=len(checks),contracts=checks,
        constant_history_contract=history,cases=cases,preprocessing=preprocessing,source_sha256=sources,
        vectors=dict(path=str(artifact.relative_to(ROOT)),sha256=N.sha(artifact)),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Eight gradients and48 discarded Adam forks; genuine stored moments or explicitly fresh. FIT-only; different saved protocols. '
              'D4 uses explicitly recomputed current-host FIT transform; its statistic-byte hash differs from AWS. '
              'No refit, DEV/test tuning, historical clipping-frequency estimate, whole-training-gradient or supremacy claim. '
              'Total diagnostic FLOPs/traffic/energy unknown, not zero.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',contracts=len(checks),wall_s=report['wall_s'])))


if __name__=='__main__': main()
