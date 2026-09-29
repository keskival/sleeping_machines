"""Content-retrieval full-value SHD phase matched to E134 fixed keys."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e134_value_phase import build,SOURCE,CACHE,hashes
from e135_content_memory import install
from e122_shd_continuation import augment
from e117_serial_event_shd import batch,load_items
from e118_race_carrier_shd import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--key-mode',choices=('fixed',),default='fixed')
    ap.add_argument('--limit',type=int,default=4096);ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--bs',type=int,default=4);ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--checkpoint-every',type=int,default=128);ap.add_argument('--resume-progress',action='store_true')
    a=ap.parse_args();out=Path('experiments/results/e135')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or min(a.limit,a.epochs,a.bs,a.checkpoint_every)<1 or a.lr<=0:
        raise ValueError('Invalid settings')
    progress_path=out.with_suffix('.progress.pt')
    if out.exists() and not a.resume_progress:raise FileExistsError(out)
    if a.resume_progress and (not out.exists() or not progress_path.exists()):raise ValueError('Missing progress')
    torch.set_num_threads(1)
    contract=json.loads(Path('experiments/results/e135/content_contract_20260929.json').read_text())
    assert contract['status']=='completed'
    assert contract['source_sha256']['experiments/e135_content_memory.py']==hashlib.sha256(Path('experiments/e135_content_memory.py').read_bytes()).hexdigest()
    model,_,config=build(a.key_mode);install(model)
    config['content_memory']={'pairs':4,'strength':.5,'row_bound':.5,'compressed_state':True}
    named=dict(model.named_parameters());names=[n for n,p in named.items() if p.requires_grad]
    initial_state={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=torch.optim.Adam([named[n] for n in names],lr=a.lr)
    saved=torch.load(SOURCE,weights_only=False,map_location='cpu')
    order_rng=np.random.default_rng(8);order_rng.bit_generator.state=saved['order_rng']
    aug_rng=np.random.default_rng(128);aug_rng.bit_generator.state=saved['augmentation_rng']
    fit=load_items(40,.01,a.limit,'fit_spk',6);held=load_items(40,.01,512,'val_spk',7)
    source_hashes=hashes(__file__)
    source_hashes['experiments/e135_content_memory.py']=hashlib.sha256(Path('experiments/e135_content_memory.py').read_bytes()).hexdigest()
    cached=json.loads(CACHE.read_text())
    assert cached['status']=='completed'
    assert cached['checkpoint_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name in ('sleeping_machines/shared_event.py','sleeping_machines/event_memory.py',
                 'experiments/e117_serial_event_shd.py','experiments/e118_race_carrier_shd.py'):
        assert cached['source_sha256'][name]==source_hashes[name],name
    splits={'fit':fit,'dev_original':held[:256],'dev_additional':held[256:]}
    for name,items in splits.items():assert [r[4] for r in items]==cached[name+'_ids'],name
    # Zero context columns, unchanged weights and eval semantics nest the parent.
    # The separate/coupled initial forward equality is checked by E134 contracts.
    # Reuse the same completed initial evaluation for BOTH arms.
    start=time.perf_counter();elapsed_before=0.
    if a.resume_progress:
        progress=torch.load(progress_path,weights_only=False,map_location='cpu');record=progress['record']
        assert record['status']=='running' and record['source_sha256']==source_hashes
        assert all(record['args'][k]==v for k,v in vars(a).items() if k!='resume_progress')
        model.load_state_dict(progress['state_dict']);opt.load_state_dict(progress['optimizer'])
        order_rng.bit_generator.state=progress['order_rng'];aug_rng.bit_generator.state=progress['augmentation_rng']
        torch.set_rng_state(progress['torch_rng']);elapsed_before=record['wall_s']
    else:
        progress=None
        record={'status':'running','args':vars(a),'config':config,'initial':cached['initial'],
                'train_names':names,'train_parameters':sum(named[n].numel() for n in names),
                'total_parameters':sum(p.numel() for p in named.values()),
                'source_sha256':source_hashes,'checkpoint_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'cache_provenance':{'path':str(CACHE),'sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),
                   'boundary':'Reuse initial quality evaluation; zero-query exact temporal-mean nesting audited in E135. Expanded state work charged separately.'},
                'ids':{n:[r[4] for r in items] for n,items in splits.items()},'curve':[],
                'protocol':'SHD training speakers only, speakers 3/6 reserved; 4096 fit, 512 held; no official test access',
                'optimizer':'Fresh Adam for both changed value-gradient programs; no inherited coupled policy momentum',
                'credit':'Ordinary realized value/time path gradient; no local loser surrogate in either arm; policy tensors frozen',
                'augmentation':'Same inherited RNG; band shift -2..2 and exp(time-scale Uniform[-.15,.15])',
                'hardware':{'platform':platform.platform(),'torch':torch.__version__,'device':'cpu','threads':1},
                'energy_joules':None,'contract_sha256':hashlib.sha256(Path('experiments/results/e135/content_contract_20260929.json').read_bytes()).hexdigest()}

    def persist():
        record['wall_s']=elapsed_before+time.perf_counter()-start
        record['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(out)

    persist();print(json.dumps({'key_mode':a.key_mode,'train_parameters':record['train_parameters']}),flush=True)
    for epoch in range(progress['epoch'] if progress else 1,a.epochs+1):
        resuming=progress is not None and progress['epoch']==epoch
        order=progress['order'] if resuming else order_rng.permutation(len(fit))
        acc=progress['acc'] if resuming else {'nll':0.,'steps':0,'packets':0,'values':0,'scans':0,'key_scans':0,
              'key_values':0,'global_scans':0,'gradient_norms':np.zeros(8),'clipped_steps':0,
              'content_calls':0,'content_work':{},'query_gradients':np.zeros(8),'key_gradients':np.zeros(8)}
        model.train();first=progress['next_start'] if resuming else 0
        for pos in range(first,len(order),a.bs):
            rows=[augment(fit[j],aug_rng) for j in order[pos:pos+a.bs]];x=batch(rows)
            opt.zero_grad(set_to_none=True);scores,_,stats,_=model(*x[:4],len(rows))
            loss=F.cross_entropy(scores,x[-1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SHD loss')
            loss.backward()
            for j,layer in enumerate(model.layers):
                acc['gradient_norms'][j]+=float(torch.cat([p.grad.flatten() for p in layer.parameters()
                       if p.requires_grad and p.grad is not None]).norm())
                acc['query_gradients'][j]+=float(layer.memory.query.grad.norm())
                acc['key_gradients'][j]+=float(layer.memory.key.grad.norm())
                for call in layer.memory.pop_calls():
                    acc['content_calls']+=1
                    for field in ('scan_scalar_multiply_adds','plain_scan_scalar_multiply_adds',
                                  'projection_multiply_adds','retrieval_multiply_adds','materialized_state_scalars'):
                        acc['content_work'][field]=acc['content_work'].get(field,0)+call[field]
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            acc['clipped_steps']+=int(norm>1);opt.step()
            acc['nll']+=float(loss.detach())*len(rows);acc['steps']+=1;acc['packets']+=stats['packets']
            for s in stats['layers']:
                for target,field in (('values','value_evaluations'),('scans','scan_compositions'),
                    ('key_scans','key_scan_compositions'),('key_values','key_value_evaluations'),
                    ('global_scans','global_scan_compositions')):acc[target]+=s.get(field,0)
            next_pos=min(pos+a.bs,len(order))
            if acc['steps']%a.checkpoint_every==0 or next_pos==len(order):
                record['progress']={'epoch':epoch,'completed_examples':next_pos,'online_nll':acc['nll']/next_pos}
                persist();state={'record':record,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
                  'order_rng':order_rng.bit_generator.state,'augmentation_rng':aug_rng.bit_generator.state,
                  'torch_rng':torch.get_rng_state(),'epoch':epoch,'next_start':next_pos,'order':order,'acc':acc}
                temp=progress_path.with_suffix('.pt.tmp');torch.save(state,temp);temp.replace(progress_path)
                print(json.dumps(record['progress']),flush=True)
        row={n:v for n,v in acc.items() if n not in ('gradient_norms','nll','query_gradients','key_gradients')}
        row.update(epoch=epoch,online_nll=acc['nll']/len(fit),
           value_layer_gradient_norms=(acc['gradient_norms']/acc['steps']).tolist(),
           query_gradient_norms=(acc['query_gradients']/acc['steps']).tolist(),
           content_key_gradient_norms=(acc['key_gradients']/acc['steps']).tolist(),
           parameter_l2_changes={n:float((named[n].detach()-initial_state[n]).norm()) for n in names},
           frozen_parameters_exact=all(torch.equal(p,initial_state[n]) for n,p in named.items() if not p.requires_grad))
        assert row['frozen_parameters_exact']
        row.update({n:evaluate(model,items,a.bs) for n,items in splits.items()})
        for layer in model.layers:layer.memory.pop_calls()
        payload_bound=float(model.embedding.weight.detach().abs().max())+1.
        ly=model.layers[0].memory.input_lipschitz_bound(payload_bound)
        gains=[ly*(2 if j in model.global_context_layers else 1) for j in range(8)]
        assert all(layer.alpha*gains[j]<1 for j,layer in enumerate(model.layers))
        row['conditional_state_transport_bounds']=[math.prod(1-layer.alpha*gains[j] for j,layer in enumerate(model.layers)),
                                                  math.prod(1+layer.alpha*gains[j] for j,layer in enumerate(model.layers))]
        row['clean_key_schedules_preserved']={n:row[n]['winner_counts']==record['initial'][n]['winner_counts'] and
             row[n]['mean_added_delay_ms']==record['initial'][n]['mean_added_delay_ms'] for n in splits}
        if a.key_mode=='fixed':assert all(row['clean_key_schedules_preserved'].values())
        record['curve'].append(row);persist();progress=None
        checkpoint={'state_dict':model.state_dict(),'config':config,'args':vars(a),'optimizer':opt.state_dict(),
                    'order_rng':order_rng.bit_generator.state,'augmentation_rng':aug_rng.bit_generator.state}
        temp=out.with_suffix('.pt.tmp');torch.save(checkpoint,temp);temp.replace(out.with_suffix('.pt'))
        print(json.dumps({'epoch':epoch,'fit':row['fit']['accuracy'],
             'held_correct':row['dev_original']['correct']+row['dev_additional']['correct'],
             'value_gradients':row['value_layer_gradient_norms'],'wall_s':record['wall_s']}),flush=True)
    record.update(status='completed',final=record['curve'][-1],
                  scope='One seed; E134 fixed-key full-value mask plus 2048 query/key parameters, same parent, examples, augmentation and fresh optimizer budget. Content retrieval has additional measured partial forward work. Initial quality cached through audited nesting. No energy or official-test claim.')
    persist()


if __name__=='__main__':main()
