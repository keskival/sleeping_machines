"""One guarded, completed-query learning screen of observable event memory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e136_scattering_shd_model import ScatteringClassifier
from e117_serial_event_shd import batch,load_items
from e134_value_phase import SOURCE


@torch.no_grad()
def evaluate(model,items):
    model.eval();correct=0;nll=0.;predictions=[];packets=0
    winners=np.zeros((model.depth,3),dtype=np.int64);delay=0.
    for start in range(0,len(items),4):
        rows=items[start:start+4];data=batch(rows)
        z,_,st,_=model(*data[:4],len(rows));p=z.argmax(-1)
        correct+=int((p==data[-1]).sum());nll+=float(F.cross_entropy(z,data[-1],reduction='sum'))
        predictions.extend(p.tolist());packets+=st['packets'];delay+=st['mean_added_delay_ms']*st['packets']
        for j,layer in enumerate(st['layers']):winners[j]+=layer['winner_counts']
    return {'correct':correct,'n':len(items),'accuracy':correct/len(items),'nll':nll/len(items),
            'predictions':predictions,'packets':packets,'winner_counts':winners.tolist(),
            'mean_added_delay_ms':delay/packets}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--readout',choices=('state','packets'),required=True)
    ap.add_argument('--limit',type=int,default=1024);ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--lr',type=float,default=.0003);ap.add_argument('--depth',type=int,default=12)
    a=ap.parse_args();out=Path('experiments/results/e136')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists() or min(a.limit,a.epochs,a.depth)<1:raise ValueError('Invalid output/settings')
    torch.set_num_threads(1);started=time.perf_counter()
    model=ScatteringClassifier(a.depth,a.readout)
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().clone() for n,p in named.items()}
    frozen={n:p.detach().clone() for n,p in model.key.named_parameters()}
    fit=load_items(40,.01,a.limit,'fit_spk',6);held=load_items(40,.01,512,'val_spk',7)
    model.calibrate(fit[:128],batch)
    opt=torch.optim.Adam(named.values(),lr=a.lr);rng=np.random.default_rng(8)
    sources=[Path(__file__),Path('experiments/e136_scattering_shd_model.py'),Path('experiments/e136_event_scattering.py'),
             Path('experiments/e117_serial_event_shd.py'),Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py')]
    record={'status':'running','args':vars(a),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
      'parent_key_checkpoint_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'train_parameters':sum(p.numel() for p in named.values()),'frozen_key_parameters':sum(p.numel() for p in frozen.values()),
      'fit_ids':[r[4] for r in fit],'held_ids':[r[4] for r in held],
      'protocol':'Fit-only speakers; 512 held training-file speakers 3/6. Completed-utterance class CE only, no early class loss, no official test.',
      'key_provenance':'Inherited E122 4096-fit eight-layer checkpoint; frozen key layers are reused through twelve value exchanges. Not from scratch or a matched parent continuation.',
      'value_encoding':'Embedding times four fixed time features, multiplied by sqrt(packet count); addressed orthogonal state exchange. Readout uses outgoing sum/sqrt(total count), log count and optionally ALL final layer states.',
      'calibration':'First 128 fitting utterances only; per-coordinate mean/std, std floor .1; frozen thereafter',
      'augmentation':'None: first isolate learnability and the supervised memory boundary',
      'comparison':'State versus emitted-packet-only query; same nominal parameters/data/optimizer, different active readout capacity and initial logits.',
      'hardware':{'platform':platform.platform(),'torch':torch.__version__,'threads':1,'device':'cpu'},
      'energy_joules':None,'curve':[]}
    def persist():
        record['wall_s']=time.perf_counter()-started;record['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary=out.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out)
    record['initial']={'fit':evaluate(model,fit),'held':evaluate(model,held)};persist()
    print(json.dumps({'initial_fit':record['initial']['fit']['accuracy'],'initial_held':record['initial']['held']['accuracy']}),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();order=rng.permutation(len(fit));grads=np.zeros(a.depth);nll=0.;steps=0;updates=0;keymaps=0;clipped=0
        for pos in range(0,len(fit),4):
            rows=[fit[j] for j in order[pos:pos+4]];data=batch(rows);opt.zero_grad(set_to_none=True)
            z,_,stats,_=model(*data[:4],len(rows));loss=F.cross_entropy(z,data[-1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite class loss')
            loss.backward();grads+=model.angle_weight.grad.flatten(1).norm(dim=1).detach().numpy()
            norm=torch.nn.utils.clip_grad_norm_(named.values(),1.,error_if_nonfinite=True);clipped+=int(norm>1);opt.step()
            nll+=float(loss.detach())*len(rows);steps+=1
            updates+=sum(r['winning_state_updates'] for r in stats['layers'])
            keymaps+=sum(r['value_evaluations'] for r in stats['layers'])
            if steps%64==0 or pos+4>=len(fit):
                record['progress']={'epoch':epoch,'completed_examples':min(pos+4,len(fit)),'online_nll':nll/min(pos+4,len(fit))};persist()
                temporary=out.with_suffix('.progress.pt.tmp');torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                  'rng':rng.bit_generator.state,'epoch':epoch,'order':order,'next_start':min(pos+4,len(fit)),
                  'accumulators':{'nll':nll,'steps':steps,'gradients':grads,'updates':updates,'keymaps':keymaps},'record':record},temporary)
                temporary.replace(out.with_suffix('.progress.pt'));print(json.dumps(record['progress']),flush=True)
        assert all(torch.equal(p,frozen[n]) for n,p in model.key.named_parameters())
        row={'epoch':epoch,'online_nll':nll/len(fit),'angle_layer_gradient_norms':(grads/steps).tolist(),
          'winning_state_updates':updates,'frozen_key_value_map_evaluations':keymaps,'clipped_steps':clipped,
          'fit':evaluate(model,fit),'held':evaluate(model,held),
          'train_parameter_l2_changes':{n:float((p.detach()-initial[n]).norm()) for n,p in named.items()},
          'frozen_key_parameters_exact':True}
        for split in ('fit','held'):
            assert row[split]['winner_counts']==record['initial'][split]['winner_counts']
            assert row[split]['mean_added_delay_ms']==record['initial'][split]['mean_added_delay_ms']
        record['curve'].append(row);persist()
        torch.save({'state_dict':model.state_dict(),'args':vars(a),'optimizer':opt.state_dict(),'rng':rng.bit_generator.state},out.with_suffix('.pt'))
        print(json.dumps({'epoch':epoch,'fit':row['fit']['accuracy'],'held':row['held']['accuracy'],'gradients':row['angle_layer_gradient_norms']}),flush=True)
    record.update(status='completed',final=record['curve'][-1],
      scope='Exploratory interface/learnability screen; pretrained keys, small unaugmented fit, larger terminal head. Gradient and fitting progress are separate from generalization and a matched-resource accuracy claim.')
    persist()


if __name__=='__main__':main()
