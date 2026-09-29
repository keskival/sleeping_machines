"""Guarded larger SHD run with linear-work event memory and resumable state.

Same winner-only E118 architecture. Fit/dev partitions and causal packets are
unchanged. The epoch budget and cosine schedule are fixed before the run;
no official test access. Best-dev selection is explicitly development only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from e117_serial_event_shd import batch, load_items
from e118_race_carrier_shd import RaceNet, calibrate, evaluate

torch.set_num_threads(1)
OUT = Path('experiments/results/e119')


def atomic_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--limit', type=int, default=1024)
    p.add_argument('--eval-limit', type=int, default=256)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--resume', help='E119 checkpoint with optimizer/RNG state; new output tag required')
    a = p.parse_args()
    if Path(a.tag).name != a.tag or min(a.limit,a.eval_limit,a.epochs,a.bs)<1:
        raise ValueError('Invalid tag or budget')
    OUT.mkdir(exist_ok=True)
    output = OUT/(a.tag+'.json')
    if output.exists(): raise FileExistsError(output)
    started = time.perf_counter()
    torch.manual_seed(a.seed)
    net = RaceNet(depth=a.depth,memory_backend='linear')
    fit = load_items(40,.01,a.limit,'fit_spk',a.seed)
    dev = load_items(40,.01,a.eval_limit,'val_spk',a.seed+1)
    optimizer = torch.optim.Adam(net.parameters(),lr=a.lr)
    rng = np.random.default_rng(a.seed+2)
    start_epoch, previous_curve, resume_sha = 0, [], None
    if a.resume:
        saved = torch.load(a.resume,weights_only=False,map_location='cpu')
        for key in ('limit','eval_limit','depth','epochs','bs','lr','seed'):
            if saved['args'][key] != getattr(a,key): raise ValueError('Resume protocol differs: '+key)
        net.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        start_epoch = saved['epoch']
        previous_curve = saved['curve']
        normalization = saved['readout_calibration']
        resume_sha = hashlib.sha256(Path(a.resume).read_bytes()).hexdigest()
        if start_epoch >= a.epochs: raise ValueError('Checkpoint already completed this schedule')
    else:
        normalization = calibrate(net,fit,a.bs)
    result = {'status':'running','args':vars(a),'readout_calibration':normalization,
              'parameters':sum(p.numel() for p in net.parameters()),
              'protocol':'SHD train partition only; speakers 3/6 held out; terminal classification; no augmentation; no test access',
              'schedule':'epochwise cosine .003 to .0003 (scaled by --lr); fixed in advance',
              'fit_ids':[x[4] for x in fit], 'dev_ids':[x[4] for x in dev],
              'fit_labels':[x[3] for x in fit], 'dev_labels':[x[3] for x in dev],
              'curve':previous_curve, 'resume_sha256':resume_sha,
              'source_sha256':{f:hashlib.sha256(Path('experiments',f).read_bytes()).hexdigest() for f in
                               ('e119_train_shd.py','e119_linear_event_scan.py','e118_race_carrier_shd.py','e117_serial_event_shd.py',
                                '../sleeping_machines/shared_event.py','../sleeping_machines/event_memory.py')},
              'hardware':{'platform':platform.platform(),'torch':torch.__version__,'threads':torch.get_num_threads()},
              'energy_joules':None}
    result['initial'] = {'fit':evaluate(net,fit,a.bs),'dev':evaluate(net,dev,a.bs)}
    atomic_json(output,result)
    best = max((r['dev']['accuracy'] for r in previous_curve),default=-1.)
    for epoch in range(start_epoch+1,a.epochs+1):
        lr = a.lr*(.1+.9*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1,1)))/2)
        for group in optimizer.param_groups: group['lr'] = lr
        order = rng.permutation(len(fit))
        net.train()
        train_nll, steps, work, value_work = 0., 0, 0, 0
        route_grads = np.zeros(a.depth)
        epoch_start = time.perf_counter()
        for start in range(0,len(fit),a.bs):
            items = [fit[i] for i in order[start:start+a.bs]]
            inputs = batch(items)
            optimizer.zero_grad(set_to_none=True)
            logits,_,stats,_ = net(*inputs[:4],len(items))
            loss = F.cross_entropy(logits,inputs[-1])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for j,layer in enumerate(net.layers):
                if layer.route.grad is not None: route_grads[j] += float(layer.route.grad.norm())
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            train_nll += float(loss.detach())*len(items)
            steps += 1
            work += sum(x['scan_compositions'] for x in stats['layers'])
            value_work += sum(x['value_evaluations'] for x in stats['layers'])
        training_s = time.perf_counter()-epoch_start
        row = {'epoch':epoch,'lr':lr,'training_online_nll':train_nll/len(fit),
               'fit':evaluate(net,fit,a.bs),'dev':evaluate(net,dev,a.bs),
               'route_grad_norm':(route_grads/steps).tolist(),'training_s':training_s,
               'training_scan_compositions':work,'training_value_evaluations':value_work,
               'invocation_wall_s':time.perf_counter()-started}
        result['curve'].append(row)
        result['wall_s'] = time.perf_counter()-started
        result['max_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        checkpoint = {'args':vars(a),'epoch':epoch,'state_dict':net.state_dict(),
                      'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,
                      'torch_rng':torch.get_rng_state(),'readout_calibration':normalization,
                      'curve':result['curve'],'source_sha256':result['source_sha256']}
        dest = OUT/(a.tag+'.pt')
        temp = dest.with_suffix('.pt.tmp')
        torch.save(checkpoint,temp)
        temp.replace(dest)
        if row['dev']['accuracy'] > best:
            best = row['dev']['accuracy']
            torch.save(checkpoint,OUT/(a.tag+'_best_dev.pt'))
        atomic_json(output,result)
        print(json.dumps({k:row[k] for k in ('epoch','lr','training_online_nll','training_s','invocation_wall_s')} | {
            'fit_accuracy':row['fit']['accuracy'],'dev_accuracy':row['dev']['accuracy'],
            'fit_nll':row['fit']['nll'],'dev_nll':row['dev']['nll']}),flush=True)
    result['status'] = 'completed'
    result['best_dev_epoch'] = max(result['curve'],key=lambda r:r['dev']['accuracy'])['epoch']
    atomic_json(output,result)
    print('completed',output,flush=True)

if __name__ == '__main__': main()
