"""Observer-conditioned D12 with live pre-normalized features and verified rates.

Old parameters retain their optimizer moments and clipping ownership. A common
fitting-only replay calibrates the old/new learning-rate scale before training.
New identity-block output maps get a fresh optimizer group. Both arms consume
identical fitting examples, transformation/order RNG and one encoder pass.
New depth adds real work and 36ms initial output delay; class queries are untimed.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from e139_fine_packet_model import augment_marked,load_marked
from e150_single_state_shd import forward_query,evaluate
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.depth_growth import grow_event_encoder
from sleeping_machines.observer_conditioned_depth import grow_observer_conditioned


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--run',default='experiments/results/e152/nuisance_state_n6144_s6_e2_20260930.json')
    p.add_argument('--depth',type=int,required=True);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--limit',type=int,default=6144);p.add_argument('--bs',type=int,default=4)
    p.add_argument('--lr',type=float,default=.000325);p.add_argument('--seed',type=int,default=6)
    p.add_argument('--calibrate',action='store_true');a=p.parse_args()
    out=Path('experiments/results/e163')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists() or min(a.limit,a.epochs,a.bs)<1:
        raise ValueError('Unique output and positive settings required')
    reference=json.loads(Path(a.run).read_text())
    if reference['status']!='completed':raise ValueError('Completed starting model required')
    selected=min(reference['curve'],key=lambda r:(-r['dev']['correct'],r['dev']['nll']))
    checkpoint=Path(a.run).with_name(Path(a.run).stem+f".epoch{selected['epoch']}.pt")
    for path,digest in reference['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed inherited source')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['result']['final']['epoch']!=selected['epoch']:raise ValueError('Wrong selected checkpoint')
    cfg=saved['encoder_config'];window=cfg['window']
    if a.depth<cfg['depth']:raise ValueError('This protocol does not remove trained layers')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    base=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
    base.load_state_dict(saved['encoder_state_dict'])
    growth=dict(norm_learning_rate_scale=1.)
    if a.depth>cfg['depth']:encoder,growth=grow_observer_conditioned(base,a.depth)
    else:encoder=copy.deepcopy(base)
    old_names=[name for name,_ in base.named_parameters()];named=dict(encoder.named_parameters())
    old_params=[named[name] for name in old_names]
    old_set=set(old_names);new_params=[param for name,param in named.items() if name not in old_set]
    new_norm=[param for name,param in named.items() if name not in old_set and '.norm.' in name]
    norm_ids={id(q) for q in new_norm};new_maps=[q for q in new_params if id(q) not in norm_ids]
    opt=torch.optim.Adam(old_params,lr=a.lr);opt.load_state_dict(saved['optimizer'])
    original_old_rate=opt.param_groups[0]['lr']
    step_factor=1.;calibrations=[];new_rate=a.lr
    for path in (Path('experiments/results/e156/adam_step_audit_20260930.json'),
                 Path('experiments/results/e162/observer_depth_replay_20260930.json')):
        row=json.loads(path.read_text())
        if row['status']!='completed' or row['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Matching completed first-step calibration required')
        calibrations.append(row)
    step_factor=calibrations[0]['selected_factor']
    new_rate=calibrations[1]['selected_new_learning_rate']
    if new_rate is None:raise ValueError('No admissible new-depth first update')
    for group in opt.param_groups:group['lr']*=step_factor
    old_rate=opt.param_groups[0]['lr']
    if new_maps:opt.add_param_group(dict(params=new_maps,lr=new_rate))
    if new_norm:opt.add_param_group(dict(params=new_norm,
        lr=new_rate))
    expected_rates=[g['lr'] for g in opt.param_groups]
    for group in opt.param_groups:group['initial_lr']=group['lr']
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,
        lr_lambda=lambda epoch:.1+.9*.5*(1+math.cos(math.pi*min(epoch,a.epochs)/a.epochs)))
    actual_rates=[g['lr'] for g in opt.param_groups]
    if actual_rates!=expected_rates:raise ValueError('Scheduler changed calibrated rates')
    print(json.dumps({'verified_initial_optimizer_rates':actual_rates}),flush=True)
    fit=load_marked(a.limit,'fit_spk',a.seed);held=load_marked(8 if a.calibrate else 512,'val_spk',a.seed+1)
    if len(fit)!=a.limit or set(r[8] for r in fit)&set(r[8] for r in held):raise ValueError('Invalid split')
    order_rng=np.random.default_rng(a.seed+2);order_rng.bit_generator.state=saved['order_rng']
    aug_rng=np.random.default_rng(a.seed+122);aug_rng.bit_generator.state=saved['augmentation_rng']
    initial=evaluate(encoder,held,a.bs,window)
    baseline=evaluate(base,held,a.bs,window)
    if initial!=baseline:raise ValueError('Growth changed initial class function')
    if not a.calibrate and initial['correct']!=selected['dev']['correct']:raise ValueError('Wrong initial score')
    paths=[Path(__file__),Path('sleeping_machines/depth_growth.py'),Path('sleeping_machines/observer_conditioned_depth.py'),Path('sleeping_machines/event_state.py'),
        Path('sleeping_machines/event_memory.py'),Path('sleeping_machines/rotating_memory.py'),
        Path('experiments/e150_single_state_shd.py'),Path('experiments/e143_event_state_shd.py'),
        Path('experiments/e139_fine_packet_model.py')]
    result=dict(status='running',args=vars(a),curve=[],initial=dict(dev=initial),
        selected_starting_epoch=selected['epoch'],starting_old_learning_rate=old_rate,
        original_old_learning_rate=original_old_rate,common_step_factor=step_factor,new_output_learning_rate=new_rate,
        growth_geometry=growth,verified_initial_optimizer_rates=actual_rates,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        fit_absolute_ids=[r[8] for r in fit],dev_absolute_ids=[r[8] for r in held],
        deployed_parameters=sum(q.numel() for q in encoder.parameters()),new_parameters=sum(q.numel() for q in new_params),
        initial_added_clock_s=.006*(a.depth-cfg['depth']),starting_prefix_depth=cfg['depth'],
        architecture=f'One {a.depth}-block modal event encoder; old blocks/head retained; new blocks normalize state features before observer-conditioned zero output maps',
        optimizer_protocol='Retained old Adam moments/parameter order; fitting-only old-step calibration plus actual new-depth replay; old norm cap1 unchanged; new parameters own fresh groups and norm cap1',
        protocol='Matched inherited private speaker3/6 continuation, same unique fit/held data and augmentation/order RNG; greater depth costs more work; no official test',
        work_scope='Partial new-encoder training-forward maps/scan/clocks only; not complete FLOPs, memory traffic or joules',
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1),energy_joules=None)
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    del base,saved
    persist();print(json.dumps({'initial_dev':initial['accuracy'],'old_lr':old_rate,'depth':a.depth,
        'new_parameters':result['new_parameters']}),flush=True)
    for epoch in range(1,a.epochs+1):
        encoder.train();order=order_rng.permutation(len(fit));train_start=time.perf_counter()
        acc=dict(nll=0.,steps=0,clipped=0,source_events=0,new_packets=0,
            source_table_projection_macs=0,source_transported_scalars=0,source_payload_sum_scalars=0,
            event_projection_macs=0,gate_macs=0,state_compositions=0,clock_candidates=0,observer_weight_fold_macs=0,
            layer_gradient_norm=[0.]*a.depth)
        for start in range(0,len(fit),a.bs):
            rows=[augment_marked(fit[j],aug_rng) for j in order[start:start+a.bs]]
            opt.zero_grad(set_to_none=True);score,_,y,new=forward_query(encoder,rows,window)
            loss=F.cross_entropy(score,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for i,layer in enumerate(encoder.layers):
                acc['layer_gradient_norm'][i]+=float(torch.sqrt(sum(
                    (q.grad.square().sum() for q in layer.parameters() if q.grad is not None),score.new_tensor(0.))))
            old_norm=torch.nn.utils.clip_grad_norm_(old_params,1.,error_if_nonfinite=True)
            new_norm=(torch.nn.utils.clip_grad_norm_(new_params,1.,error_if_nonfinite=True) if new_params else 0.)
            acc['clipped']+=int(old_norm>1);acc['new_clipped']=acc.get('new_clipped',0)+int(new_norm>1);opt.step()
            acc['nll']+=float(loss.detach())*len(rows);acc['steps']+=1
            acc['source_events']+=new[0]['source_events'];acc['new_packets']+=new[0]['emitted_vectors']
            for name in ('source_table_projection_macs','source_transported_scalars','source_payload_sum_scalars'):
                acc[name]+=new[0][name]
            acc['event_projection_macs']+=sum(r.get('input_projection_macs',0)+r['output_projection_macs'] for r in new)
            acc['gate_macs']+=sum(r['nonlinear_gate_macs'] for r in new)
            acc['state_compositions']+=sum(r['state_scan_compositions'] for r in new)
            acc['clock_candidates']+=sum(r['clock_candidates'] for r in new)
            acc['observer_weight_fold_macs']+=sum(r.get('observer_weight_fold_macs',0) for r in new)
            if acc['steps']%128==0:
                result['progress']=dict(epoch=epoch,completed_examples=start+len(rows),online_nll=acc['nll']/(start+len(rows)))
                persist();print(json.dumps(result['progress']),flush=True)
        acc.update(online_nll=acc['nll']/len(fit),training_wall_s=time.perf_counter()-train_start)
        row=dict(epoch=epoch,training=acc,fit=evaluate(encoder,fit,a.bs,window),
            dev=evaluate(encoder,held,a.bs,window),lr=opt.param_groups[0]['lr'])
        all_layers=encoder.layers
        encoder.layers=torch.nn.ModuleList(list(all_layers)[:cfg['depth']])
        row['prefix_dev']=evaluate(encoder,held,a.bs,window)
        encoder.layers=all_layers
        row['new_output_parameter_norms']=[float(layer.output.parametrizations.weight.original.detach().norm()) for layer in encoder.layers[cfg['depth']:]]
        row['new_normalization_gain_norms']=[float(layer.norm.weight.detach().norm()) for layer in encoder.layers[cfg['depth']:]]
        row['depth_vs_its_prefix']=dict(
            full_only_correct=sum(p==y and q!=y for p,q,y in zip(row['dev']['predictions'],row['prefix_dev']['predictions'],row['dev']['labels'])),
            prefix_only_correct=sum(p!=y and q==y for p,q,y in zip(row['dev']['predictions'],row['prefix_dev']['predictions'],row['dev']['labels'])),
            changed_predictions=sum(p!=q for p,q in zip(row['dev']['predictions'],row['prefix_dev']['predictions'])))
        result['curve'].append(row);result['final']=row;scheduler.step();persist()
        payload=dict(encoder_state_dict=encoder.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            args=vars(a),encoder_config=cfg,actual_depth=a.depth,result=result,order_rng=order_rng.bit_generator.state,
            augmentation_rng=aug_rng.bit_generator.state,torch_rng=torch.get_rng_state())
        torch.save(payload,out.with_suffix('.pt'))
        torch.save(payload,out.with_name(out.stem+f'.epoch{epoch}.pt'))
        print(json.dumps({'epoch':epoch,'fit':row['fit']['accuracy'],'dev':row['dev']['accuracy'],
            'dev_nll':row['dev']['nll'],'training_wall_s':acc['training_wall_s'],
            'rss_kb':result['max_rss_kb']}),flush=True)
    result['status']='completed';persist()


if __name__=='__main__':main()
