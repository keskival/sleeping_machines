"""Matched compact-query SHD learning intervention; one guarded run per queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e137_compact_scattering_model import CompactScatteringClassifier
from e136_scattering_shd import evaluate
from e117_serial_event_shd import batch, load_items
from e134_value_phase import SOURCE


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True)
    ap.add_argument('--angles',choices=('learned','frozen'),required=True)
    ap.add_argument('--rank',type=int,default=16); ap.add_argument('--depth',type=int,default=12)
    ap.add_argument('--limit',type=int,default=1024); ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--resume-result',type=Path)
    ap.add_argument('--lr',type=float,default=.0003); a=ap.parse_args()
    out=Path('experiments/results/e137')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists() or min(a.rank,a.depth,a.limit,a.epochs)<1:
        raise ValueError('Invalid output/settings')
    out.parent.mkdir(exist_ok=True); torch.set_num_threads(1); started=time.perf_counter()
    model=CompactScatteringClassifier(a.depth,a.rank,a.angles)
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    initial_state_sha=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    fit=load_items(40,.01,a.limit,'fit_spk',6); held=load_items(40,.01,512,'val_spk',7)
    model.calibrate(fit[:128],batch)
    calibration_sha=hashlib.sha256(model.center.numpy().tobytes()+model.scale.numpy().tobytes()).hexdigest()
    opt=torch.optim.Adam(named.values(),lr=a.lr); rng=np.random.default_rng(8)
    sources=[Path(__file__),Path('experiments/e137_compact_scattering_model.py'),
       Path('experiments/e136_scattering_shd.py'),Path('experiments/e136_scattering_shd_model.py'),
       Path('experiments/e136_event_scattering.py'),Path('experiments/e117_serial_event_shd.py'),
       Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py')]
    arguments={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    record={'status':'running','args':arguments,'initial_state_sha256':initial_state_sha,
       'calibration_sha256':calibration_sha,
       'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
       'parent_key_checkpoint_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
       'train_parameters':sum(p.numel() for p in named.values()),
       'head_parameters':sum(p.numel() for p in model.head.parameters()),
       'frozen_key_parameters':sum(p.numel() for p in model.key.parameters()),
       'fit_ids':[r[4] for r in fit],'held_ids':[r[4] for r in held],
       'protocol':'Identical E136 1024 fitting/512 held training-speaker utterances, seed 6, three unaugmented passes, batch four, fresh Adam .0003, RNG 8; completed-utterance CE only. Official test untouched.',
       'key_provenance':'Inherited supervised E122 4096-fit eight-layer checkpoint; keys frozen and reused cyclically through 12 value exchanges. Not from scratch.',
       'query':'Same normalized packet/logcount and all retained states as E136. Rank-16 bank/channel/class factorized state readout; packet head unchanged in form. First 128 fit-only calibration examples.',
       'intervention':'Update versus freeze only angle weights/biases; embedding and identical-capacity readout learn in both. Same initialization, data, order and optimizer budget. Frozen-angle control retains input-conditioned exchange angles.',
       'head_forward_macs_per_query':model.head.forward_macs_per_query(),
       'head_contraction_boundary':'MAC counts only query contractions; excludes key/exchange scans, normalization, backward, optimizer and memory traffic. No joule measurement.',
       'hardware':{'platform':platform.platform(),'torch':torch.__version__,'threads':1,'device':'cpu'},
       'guards':{k:os.environ.get(k) for k in ('MEM_CAP_KB','MEM_CAP_RSS_KB','MIN_AVAIL_MB','JOB_TIMEOUT_S')},
       'energy_joules':None,'curve':[]}
    def persist():
        record['wall_s']=time.perf_counter()-started
        record['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary=out.with_suffix('.json.tmp'); temporary.write_text(json.dumps(record,indent=2)+'\n'); temporary.replace(out)
    completed_epochs=0
    if a.resume_result is not None:
        prior=json.loads(a.resume_result.read_text())
        completed_epochs=len(prior['curve'])
        if not 0<completed_epochs<a.epochs: raise ValueError('No incomplete epoch budget to resume')
        for key in ('angles','rank','depth','limit','epochs','lr'):
            assert prior['args'][key]==arguments[key], key
        for key in ('initial_state_sha256','calibration_sha256','fit_ids','held_ids','parent_key_checkpoint_sha256'):
            assert prior[key]==record[key], key
        for name,digest in prior['source_sha256'].items():
            if name != str(Path(__file__)):
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
        checkpoint=a.resume_result.with_suffix('.pt')
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
        model.load_state_dict(saved['state_dict']); opt.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng']
        record['initial']=prior['initial']; record['curve']=prior['curve']
        record['resume']={'completed_epochs':completed_epochs,'prior_wall_s':prior['wall_s'],
            'prior_result':str(a.resume_result),'prior_result_sha256':hashlib.sha256(a.resume_result.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'prior_source_sha256':prior['source_sha256'],
            'boundary':'Resume completed epoch checkpoint, including optimizer and next-epoch RNG. Prior result is unchanged; any uncheckpointed later work is excluded.'}
    else:
        record['initial']={'fit':evaluate(model,fit),'held':evaluate(model,held)}
    persist()
    print(json.dumps({'initial_fit':record['initial']['fit']['accuracy'],'initial_held':record['initial']['held']['accuracy']}),flush=True)
    for epoch in range(completed_epochs+1,a.epochs+1):
        model.train(); order=rng.permutation(len(fit)); grads=np.zeros(a.depth); nll=0.; steps=0; clipped=0
        for pos in range(0,len(fit),4):
            rows=[fit[j] for j in order[pos:pos+4]]; data=batch(rows); opt.zero_grad(set_to_none=True)
            z,_,_,_=model(*data[:4],len(rows)); loss=F.cross_entropy(z,data[-1])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite class loss')
            loss.backward()
            if model.angle_weight.grad is not None:
                grads+=model.angle_weight.grad.flatten(1).norm(dim=1).detach().numpy()
            norm=torch.nn.utils.clip_grad_norm_(named.values(),1.,error_if_nonfinite=True)
            clipped+=int(norm>1); opt.step(); nll+=float(loss.detach())*len(rows); steps+=1
            if steps%64==0 or pos+4>=len(fit):
                record['progress']={'epoch':epoch,'completed_examples':min(pos+4,len(fit)),'online_nll':nll/min(pos+4,len(fit))}; persist()
                temporary=out.with_suffix('.progress.pt.tmp')
                torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                    'rng':rng.bit_generator.state,'epoch':epoch,'order':order,'next_start':min(pos+4,len(fit)),
                    'accumulators':{'nll':nll,'steps':steps,'gradients':grads,'clipped':clipped},'record':record},temporary)
                temporary.replace(out.with_suffix('.progress.pt')); print(json.dumps(record['progress']),flush=True)
        assert all(torch.equal(p,initial[n]) for n,p in model.named_parameters() if n.startswith('key.'))
        if a.angles=='frozen':
            assert torch.equal(model.angle_weight,initial['angle_weight']) and torch.equal(model.angle_bias,initial['angle_bias'])
        row={'epoch':epoch,'online_nll':nll/len(fit),'angle_layer_gradient_norms':(grads/steps).tolist(),
             'clipped_steps':clipped,'fit':evaluate(model,fit),'held':evaluate(model,held),
             'train_parameter_l2_changes':{n:float((p.detach()-initial[n]).norm()) for n,p in named.items()}}
        for split in ('fit','held'):
            assert row[split]['winner_counts']==record['initial'][split]['winner_counts']
            assert row[split]['mean_added_delay_ms']==record['initial'][split]['mean_added_delay_ms']
        record['curve'].append(row); persist()
        torch.save({'state_dict':model.state_dict(),'args':arguments,'optimizer':opt.state_dict(),'rng':rng.bit_generator.state},out.with_suffix('.pt'))
        print(json.dumps({'epoch':epoch,'fit':row['fit']['accuracy'],'held':row['held']['accuracy'],'gradients':row['angle_layer_gradient_norms']}),flush=True)
    record.update(status='completed',final=record['curve'][-1],
       training_forward_head_macs=len(fit)*a.epochs*model.head.forward_macs_per_query(),
       scope='Single-seed inherited-key learning intervention with matched compact decoder; measures angular adaptation at this budget, not official-test accuracy, universal depth trainability or energy supremacy.')
    persist()


if __name__=='__main__':
    main()
