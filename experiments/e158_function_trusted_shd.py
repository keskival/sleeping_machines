"""Function-trusted D12 training with fitting-only finite parameter counterfactuals.

Every candidate update is replayed in the real event network; accepted rays
decrease observed current/paired fitting CE and stay within a probability budget.
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
from sleeping_machines.head_conditioned_depth import grow_head_conditioned


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--run',default='experiments/results/e152/nuisance_state_n6144_s6_e2_20260930.json')
    p.add_argument('--depth',type=int,required=True);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--limit',type=int,default=6144);p.add_argument('--bs',type=int,default=4)
    p.add_argument('--lr',type=float,default=.000325);p.add_argument('--seed',type=int,default=6)
    p.add_argument('--calibrate',action='store_true');p.add_argument('--kl-budget',type=float,default=.02);a=p.parse_args()
    out=Path('experiments/results/e158')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists() or min(a.limit,a.epochs,a.bs)<1:
        raise ValueError('Unique output and positive settings required')
    if a.limit<=a.bs or a.kl_budget<=0:raise ValueError('Independent fitting anchors and positive KL budget required')
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
    if a.depth>cfg['depth']:encoder,growth=grow_head_conditioned(base,a.depth)
    else:encoder=copy.deepcopy(base)
    old_names=[name for name,_ in base.named_parameters()];named=dict(encoder.named_parameters())
    old_params=[named[name] for name in old_names]
    old_set=set(old_names);new_params=[param for name,param in named.items() if name not in old_set]
    new_norm=[param for name,param in named.items() if name not in old_set and '.norm.' in name]
    norm_ids={id(q) for q in new_norm};new_maps=[q for q in new_params if id(q) not in norm_ids]
    opt=torch.optim.Adam(old_params,lr=a.lr);opt.load_state_dict(saved['optimizer'])
    original_old_rate=opt.param_groups[0]['lr']
    step_factor=1.;calibrations=[]
    if not a.calibrate:
        for path in (Path('experiments/results/e156/adam_step_audit_20260930.json'),
                     Path('experiments/results/e157/conditioned_d12_audit_20260930.json')):
            row=json.loads(path.read_text())
            if row['status']!='completed' or row['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError('Matching completed first-step calibration required')
            calibrations.append(row)
        common=set(r['factor'] for r in calibrations[0]['rows'] if r['accepted'])
        common &= set(r['factor'] for r in calibrations[1]['rows'] if r['accepted'])
        if not common:raise ValueError('No common admissible first-update scale')
        step_factor=max(common)
    for group in opt.param_groups:group['lr']*=step_factor
    old_rate=opt.param_groups[0]['lr']
    if new_maps:opt.add_param_group(dict(params=new_maps,lr=a.lr*step_factor))
    if new_norm:opt.add_param_group(dict(params=new_norm,
        lr=a.lr*step_factor*growth['norm_learning_rate_scale']))
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
    anchor_rng=np.random.default_rng(a.seed+158)
    initial=evaluate(encoder,held,a.bs,window)
    baseline=evaluate(base,held,a.bs,window)
    if initial!=baseline:raise ValueError('Growth changed initial class function')
    if not a.calibrate and initial['correct']!=selected['dev']['correct']:raise ValueError('Wrong initial score')
    paths=[Path(__file__),Path('sleeping_machines/depth_growth.py'),Path('sleeping_machines/head_conditioned_depth.py'),Path('sleeping_machines/event_state.py'),
        Path('sleeping_machines/event_memory.py'),Path('sleeping_machines/rotating_memory.py'),
        Path('experiments/e150_single_state_shd.py'),Path('experiments/e143_event_state_shd.py'),
        Path('experiments/e139_fine_packet_model.py')]
    result=dict(status='running',args=vars(a),curve=[],initial=dict(dev=initial),
        selected_starting_epoch=selected['epoch'],starting_old_learning_rate=old_rate,
        original_old_learning_rate=original_old_rate,common_step_factor=step_factor,
        growth_geometry=growth,verified_initial_optimizer_rates=actual_rates,
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        fit_absolute_ids=[r[8] for r in fit],dev_absolute_ids=[r[8] for r in held],
        deployed_parameters=sum(q.numel() for q in encoder.parameters()),new_parameters=sum(q.numel() for q in new_params),
        initial_added_clock_s=.006*(a.depth-cfg['depth']),
        architecture=f'One {a.depth}-block modal event encoder; old blocks/head retained; new blocks identity with live output teachers',
        optimizer_protocol='Retained moments and owned clipping; common initial calibration; each proposed Adam ray replayed at up to six scales, requiring current-batch and paired fitting/anchor CE descent with mean KL <= declared budget; rejected ray leaves parameters unchanged, moments advance once',
        protocol='Matched inherited private speaker3/6 continuation, same unique fit/held data and augmentation/order RNG; greater depth costs more work; no official test',
        work_scope='Gradient-forward maps/scan/clocks partial counts; anchor and replay views counted separately and their maps/scan work must also be charged; excludes complete backward, optimizer, physical traffic and joules',
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
            event_projection_macs=0,gate_macs=0,state_compositions=0,clock_candidates=0,
            layer_gradient_norm=[0.]*a.depth,accepted_updates=0,rejected_updates=0,
            replay_attempts=0,accepted_factor_sum=0.,reference_forward_views=0,replay_forward_views=0,
            extra_source_events=0,extra_packets=0,extra_source_table_projection_macs=0,
            extra_source_transported_scalars=0,extra_source_payload_sum_scalars=0,
            extra_event_projection_macs=0,extra_gate_macs=0,extra_state_compositions=0,extra_clock_candidates=0,
            maximum_accepted_mean_kl=0.,accepted_ce_change_sum=0.)
        def count_extra(rows):
            acc['extra_source_events']+=rows[0]['source_events'];acc['extra_packets']+=rows[0]['emitted_vectors']
            for field in ('source_table_projection_macs','source_transported_scalars','source_payload_sum_scalars'):
                acc['extra_'+field]+=rows[0][field]
            acc['extra_event_projection_macs']+=sum(r.get('input_projection_macs',0)+r['output_projection_macs'] for r in rows)
            acc['extra_gate_macs']+=sum(r['nonlinear_gate_macs'] for r in rows)
            acc['extra_state_compositions']+=sum(r['state_scan_compositions'] for r in rows)
            acc['extra_clock_candidates']+=sum(r['clock_candidates'] for r in rows)
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
            acc['clipped']+=int(old_norm>1);acc['new_clipped']=acc.get('new_clipped',0)+int(new_norm>1)
            # A separate fitting-only draw measures interference without using held labels.
            candidates=np.flatnonzero(~np.isin(np.arange(len(fit)),order[start:start+a.bs]))
            anchor_ids=anchor_rng.choice(candidates,size=min(a.bs,len(candidates)),replace=False)
            anchors=[augment_marked(fit[j],anchor_rng) for j in anchor_ids]
            encoder.eval()
            with torch.no_grad():
                reference_score,_,reference_y,reference_work=forward_query(encoder,rows+anchors,window)
                reference_batch_ce=float(F.cross_entropy(reference_score[:len(rows)],y))
                reference_ce=float(F.cross_entropy(reference_score,reference_y))
                reference_p=reference_score.softmax(-1)
                reference_logp=reference_score.log_softmax(-1)
            acc['reference_forward_views']+=len(rows)+len(anchors);count_extra(reference_work)
            params=list(encoder.parameters());previous=[q.detach().clone() for q in params]
            opt.step()
            delta=[q.detach()-old for q,old in zip(params,previous)]
            accepted=False
            with torch.no_grad():
                for factor in (1.,.25,.0625,.015625,.00390625,.0009765625):
                    for q,old,change in zip(params,previous,delta):q.copy_(old+factor*change)
                    candidate_score,_,candidate_y,candidate_work=forward_query(encoder,rows+anchors,window)
                    if not torch.isfinite(candidate_score).all():continue
                    if not torch.equal(candidate_y,reference_y):raise ValueError('Changed label order')
                    mean_kl=float((reference_p*(reference_logp-candidate_score.log_softmax(-1))).sum(-1).mean())
                    candidate_batch_ce=float(F.cross_entropy(candidate_score[:len(rows)],y))
                    candidate_ce=float(F.cross_entropy(candidate_score,reference_y))
                    acc['replay_attempts']+=1;acc['replay_forward_views']+=len(rows)+len(anchors)
                    count_extra(candidate_work)
                    if candidate_batch_ce<=reference_batch_ce and candidate_ce<=reference_ce and mean_kl<=a.kl_budget:
                        accepted=True;acc['accepted_updates']+=1;acc['accepted_factor_sum']+=factor
                        acc['maximum_accepted_mean_kl']=max(acc['maximum_accepted_mean_kl'],mean_kl)
                        acc['accepted_ce_change_sum']+=candidate_ce-reference_ce
                        break
                if not accepted:
                    for q,old in zip(params,previous):q.copy_(old)
                    acc['rejected_updates']+=1
            # Adam teacher statistics advance once per fitting batch; only the accepted parameter ray moves.
            encoder.train()
            del previous,delta,reference_score,reference_p,reference_logp,candidate_score
            acc['nll']+=float(loss.detach())*len(rows);acc['steps']+=1
            acc['source_events']+=new[0]['source_events'];acc['new_packets']+=new[0]['emitted_vectors']
            for name in ('source_table_projection_macs','source_transported_scalars','source_payload_sum_scalars'):
                acc[name]+=new[0][name]
            acc['event_projection_macs']+=sum(r.get('input_projection_macs',0)+r['output_projection_macs'] for r in new)
            acc['gate_macs']+=sum(r['nonlinear_gate_macs'] for r in new)
            acc['state_compositions']+=sum(r['state_scan_compositions'] for r in new)
            acc['clock_candidates']+=sum(r['clock_candidates'] for r in new)
            if acc['steps']%128==0:
                result['progress']=dict(epoch=epoch,completed_examples=start+len(rows),online_nll=acc['nll']/(start+len(rows)))
                persist();print(json.dumps(result['progress']),flush=True)
        acc.update(online_nll=acc['nll']/len(fit),training_wall_s=time.perf_counter()-train_start)
        row=dict(epoch=epoch,training=acc,fit=evaluate(encoder,fit,a.bs,window),
            dev=evaluate(encoder,held,a.bs,window),lr=opt.param_groups[0]['lr'])
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
