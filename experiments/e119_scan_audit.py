"""Scientific equivalence and CPU cost experiment for the E119 event scan."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from e117_serial_event_shd import segmented_memory as doubling, batch, load_items
from e119_linear_event_scan import segmented_memory as linear
from e118_race_carrier_shd import RaceNet, evaluate

torch.set_num_threads(1)

def contracts():
    torch.manual_seed(119)
    rows = []
    for n in (1, 2, 3, 17, 128):
        x = torch.randn(n, 4, dtype=torch.float64, requires_grad=True)
        t = (torch.arange(n, dtype=torch.float64)*.017).requires_grad_()
        c = (1+torch.rand(n, dtype=torch.float64)).requires_grad_()
        tau = torch.tensor([.02, .13, .8], dtype=torch.float64, requires_grad=True)
        keys = torch.arange(n) % 3
        v = torch.randn(n, 3, 4, dtype=torch.float64)
        q = torch.randn(n, 3, dtype=torch.float64)
        outputs, grads, work = [], [], []
        for fn in (doubling, linear):
            m, mass, w = fn(x, t, c, keys, tau)
            outputs.append(torch.cat((m.flatten(), mass.flatten())))
            raw = torch.autograd.grad((m*v).sum()+(mass*q).sum(), (x,t,c,tau), allow_unused=True)
            grads.append(tuple(torch.zeros_like(p) if g is None else g for p,g in zip((x,t,c,tau),raw)))
            work.append(w)
        error = float((outputs[0]-outputs[1]).detach().abs().max())
        ge = [float((u-v).abs().max()) for u,v in zip(*grads)]
        assert error < 1e-10 and max(ge) < 1e-9, (error, ge)
        assert work[1] < 2*n
        rows.append({'events':n, 'forward_max_error':error, 'gradient_max_errors_x_t_c_tau':ge,
                     'doubling_compositions':work[0], 'linear_compositions':work[1]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--checkpoint', required=True)
    a = p.parse_args()
    out = Path('experiments/results/e119')/(a.tag+'.json')
    out.parent.mkdir(exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    checks = contracts()
    saved = torch.load(a.checkpoint, weights_only=False, map_location='cpu')
    nets = {kind: RaceNet(depth=saved['args']['depth'], memory_backend=kind) for kind in ('doubling','linear')}
    for net in nets.values(): net.load_state_dict(saved['state_dict'])
    items = load_items(40,.01,256,'val_spk',7)
    inputs = batch(items[:4])
    equivalence, times = {}, {kind: {'inference_s':[], 'training_step_s':[]} for kind in nets}
    logits, gradients, winner_traces = [], [], []
    for kind, net in nets.items():
        net.train()
        net.zero_grad(set_to_none=True)
        z, _, _, tr = net(*inputs[:4],4,trace=True)
        F.cross_entropy(z,inputs[-1]).backward()
        logits.append(z.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]))
        winner_traces.append(torch.stack([x['winner'] for x in tr]))
    equivalence['logit_max_error'] = float((logits[0]-logits[1]).abs().max())
    equivalence['parameter_gradient_relative_l2'] = float((gradients[0]-gradients[1]).norm()/gradients[0].norm())
    equivalence['winner_disagreements'] = int((winner_traces[0]!=winner_traces[1]).sum())
    assert equivalence['logit_max_error'] < 1e-4
    assert equivalence['parameter_gradient_relative_l2'] < 1e-3
    assert equivalence['winner_disagreements'] == 0
    # Alternating order, warmup excluded, identical inputs/checkpoint; no updates.
    for repeat in range(9):
        for kind in (('doubling','linear') if repeat%2 else ('linear','doubling')):
            net = nets[kind]
            net.eval()
            start = time.perf_counter()
            with torch.no_grad(): net(*inputs[:4],4)
            elapsed = time.perf_counter()-start
            net.train()
            net.zero_grad(set_to_none=True)
            start = time.perf_counter()
            z = net(*inputs[:4],4)[0]
            F.cross_entropy(z,inputs[-1]).backward()
            training = time.perf_counter()-start
            if repeat:
                times[kind]['inference_s'].append(elapsed)
                times[kind]['training_step_s'].append(training)
    evaluations = {}
    for kind, net in nets.items():
        start = time.perf_counter()
        evaluations[kind] = evaluate(net,items,4)
        evaluations[kind]['wall_s'] = time.perf_counter()-start
    equivalence['dev_prediction_disagreements'] = sum(u!=v for u,v in zip(evaluations['doubling']['predictions'],evaluations['linear']['predictions']))
    medians = {k:{metric:float(np.median(v)) for metric,v in values.items()} for k,values in times.items()}
    result = {'status':'completed','args':vars(a),'contracts':checks,'equivalence':equivalence,
              'timings':times,'median_timings':medians, 'dev':evaluations,
              'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'source_sha256':{name:hashlib.sha256(Path('experiments',name).read_bytes()).hexdigest() for name in
                               ('e119_scan_audit.py','e119_linear_event_scan.py','e118_race_carrier_shd.py','e117_serial_event_shd.py')},
              'hardware':{'platform':platform.platform(),'torch':torch.__version__,'threads':torch.get_num_threads()},
              'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'energy_joules':None,'energy_note':'RAPL energy_uj unreadable; wall time and work are not joules'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'equivalence':equivalence,'median_timings':medians,
                      'dev':{k:{f:v[f] for f in ('accuracy','scan_compositions','wall_s')} for k,v in evaluations.items()}}),flush=True)

if __name__ == '__main__': main()
