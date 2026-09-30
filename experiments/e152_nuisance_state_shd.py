"""Paired-view, nuisance-aware absorption into one six-block event encoder.

Fit-only feature projection followed by convex head conditioning supplies a
declared initialization. Paired clean/augmented fitting views and a stronger declared whitened penalty
correct the initialization objective. Subsequent training uses utterance labels and only
the single encoder. The checkpoint inherits E143's learned features and their
training lineage; this is neither a from-scratch result nor an energy claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from e139_fine_packet_model import augment_marked, load_marked, marked_batch
from e143_event_state_shd import event_batch
from sleeping_machines.event_state import CoalescedEventStateEncoder
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.readout_absorption import project_component, optimize_head, absorption_certificate
from sleeping_machines.nuisance_readout import paired_view_metric, paired_logit_risk


def forward_query(encoder,rows,window):
    packed=event_batch(rows,window)
    scores,payload,_,stats=encoder(*packed,len(rows))
    counts,receivers=packed[5],packed[4]
    mass=payload.new_zeros(len(rows)).index_add(0,receivers,counts)
    query=payload.new_zeros((len(rows),payload.shape[1])).index_add(
        0,receivers,payload*counts[:,None])/mass[:,None]
    labels=torch.tensor([r[3] for r in rows],dtype=torch.long)
    return scores,query,labels,stats


def metrics(scores,labels):
    pred=scores.argmax(-1)
    return dict(n=len(labels),correct=int((pred==labels).sum()),
        accuracy=float((pred==labels).float().mean()),
        nll=float(F.cross_entropy(scores,labels)),predictions=pred.tolist(),labels=labels.tolist())


@torch.no_grad()
def cache_queries(core,encoder,items,bs,window):
    encoder.eval();features,parents,residuals,labels=[],[],[],[]
    raw_sources,old_packets,new_packets=0,0,0
    for start in range(0,len(items),bs):
        rows=items[start:start+bs]
        score,query,y,new=forward_query(encoder,rows,window)
        packed=marked_batch(rows)
        parent,_,old,_=core(*packed[:4],len(rows))
        features.append(query);parents.append(parent);residuals.append(score);labels.append(y)
        raw_sources+=new[0]['source_events'];old_packets+=old['packets']
        new_packets+=new[0]['emitted_vectors']
    return dict(features=torch.cat(features),parent=torch.cat(parents),residual=torch.cat(residuals),
        labels=torch.cat(labels),work=dict(source_events=raw_sources,old_packets=old_packets,new_packets=new_packets))


@torch.no_grad()
def evaluate(encoder,items,bs,window):
    encoder.eval();scores,labels=[],[]
    for start in range(0,len(items),bs):
        score,_,y,_=forward_query(encoder,items[start:start+bs],window)
        if not torch.isfinite(score).all():raise FloatingPointError('Nonfinite logits')
        scores.append(score);labels.append(y)
    return metrics(torch.cat(scores),torch.cat(labels))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--checkpoint',default='experiments/results/e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.pt')
    p.add_argument('--limit',type=int,default=6144);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--bs',type=int,default=4);p.add_argument('--lr',type=float,default=.000325)
    p.add_argument('--seed',type=int,default=6);p.add_argument('--calibrate',action='store_true')
    a=p.parse_args();out=Path('experiments/results/e152')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists() or min(a.limit,a.epochs,a.bs)<1:
        raise ValueError('Unique output and positive settings required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);started=time.perf_counter()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    reference=json.loads(Path(a.checkpoint).with_suffix('.json').read_text())
    if reference['status']!='completed' or saved['result']['final']['epoch']!=3:
        raise ValueError('Expected completed E143 pass-three checkpoint')
    for source,digest in reference['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed inherited source: '+source)
    cfg=reference['args'];window=cfg['window']
    parent=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    core=SharedEventModel(depth=8,memory_backend='linear')
    core.load_state_dict(parent['state_dict']);core.requires_grad_(False);core.eval()
    encoder=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cfg['depth'])
    encoder.load_state_dict(saved['encoder_state_dict'])
    fit=load_marked(a.limit,'fit_spk',a.seed)
    held=load_marked(8 if a.calibrate else 512,'val_spk',a.seed+1)
    if len(fit)!=a.limit or set(r[8] for r in fit)&set(r[8] for r in held):
        raise ValueError('Unavailable or overlapping split')
    order_rng=np.random.default_rng(a.seed+2);order_rng.bit_generator.state=saved['order_rng']
    aug_rng=np.random.default_rng(a.seed+122);aug_rng.bit_generator.state=saved['augmentation_rng']
    paths=[Path(__file__),Path('sleeping_machines/readout_absorption.py'),Path('sleeping_machines/nuisance_readout.py'),Path('sleeping_machines/event_state.py'),
        Path('sleeping_machines/event_memory.py'),Path('sleeping_machines/rotating_memory.py'),
        Path('sleeping_machines/shared_event.py'),Path('experiments/e143_event_state_shd.py'),
        Path('experiments/e139_fine_packet_model.py')]
    result=dict(status='running',args=vars(a),curve=[],
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        inherited_parent_sha256=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest(),
        fit_absolute_ids=[r[8] for r in fit],dev_absolute_ids=[r[8] for r in held],
        deployed_parameters=sum(q.numel() for q in encoder.parameters()),
        removed_parent_parameters=sum(q.numel() for q in core.parameters()),
        architecture='One D6 width128 modal event encoder, source lookup, nonlinear gated residuals, winning clocks and one folded affine completed-query head',
        protocol='Private train-file speaker3/6 development; inherited E143 features; two fitting-only clean/augmented views; parent-logit projection and nuisance-regularized supervised/teacher convex head; subsequent ordinary CE trains only the single encoder',
        work_scope='Training-forward new-encoder maps/scan/clock partial counts only. Initial feature/parent cache work separate; inherited training, backward, optimizer, physical traffic and joules excluded',
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1),energy_joules=None)
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    persist()
    cached=cache_queries(core,encoder,fit,a.bs,window)
    dev_cache=cache_queries(core,encoder,held,a.bs,window)
    result['teacher']=dict(fit=metrics(cached['parent']+cached['residual'],cached['labels']),
        dev=metrics(dev_cache['parent']+dev_cache['residual'],dev_cache['labels']))
    if not a.calibrate and result['teacher']['dev']['correct']!=408:raise ValueError('Unexpected inherited score')
    result['initial_cache_work']=dict(fit_clean=cached['work'],dev=dev_cache['work'])
    import copy
    replay_order=copy.deepcopy(order_rng)
    replay_aug=copy.deepcopy(aug_rng)
    aug_order=replay_order.permutation(len(fit))
    # Keep only the small cached queries; never duplicate the complete raw dataset.
    aug_parts=[]
    for start in range(0,len(fit),a.bs):
        views=[augment_marked(fit[j],replay_aug) for j in aug_order[start:start+a.bs]]
        aug_parts.append(cache_queries(core,encoder,views,a.bs,window))
    inverse=torch.from_numpy(np.argsort(aug_order))
    aug_cache={name:torch.cat([part[name] for part in aug_parts])[inverse]
        for name in ('features','parent','residual','labels')}
    aug_cache['work']={name:sum(part['work'][name] for part in aug_parts)
        for name in ('source_events','old_packets','new_packets')}
    del aug_parts,views
    result['initial_cache_work']['fit_augmented']=aug_cache['work']
    clean_cache=cached
    clean_w,clean_b,clean_mean,clean_metric,_=project_component(cached['features'],cached['parent'])
    clean_weight,clean_bias,_=optimize_head(cached['features'],cached['parent']+cached['residual'],cached['labels'],
        encoder.head.weight+clean_w.T.to(encoder.head.weight),
        encoder.head.bias+clean_b.to(encoder.head.bias),clean_mean,clean_metric)
    def affine(h,w,b):return h.double()@w.double().T+b.double()
    clean_score=affine(cached['features'],clean_weight,clean_bias)
    aug_score=affine(aug_cache['features'],clean_weight,clean_bias)
    result['matched_clean_head']=dict(fit=metrics(clean_score,cached['labels']),
        augmented_fit=metrics(aug_score,cached['labels']),
        dev=metrics(affine(dev_cache['features'],clean_weight,clean_bias),dev_cache['labels']),
        nuisance=paired_logit_risk(clean_score,aug_score,cached['labels']))
    if not a.calibrate and result['matched_clean_head']['dev']['correct']!=374:
        raise ValueError('Unexpected matched clean-head reconstruction')
    cached={name:torch.cat((cached[name],aug_cache[name])) for name in ('features','parent','residual','labels')}
    result['head_fitting_views']=dict(unique_utterances=len(fit),views_per_utterance=2,
        total_views=len(cached['labels']),augmentation='Same first-pass draws as E150; fitting only; training RNG replayed unchanged')
    with torch.no_grad():
        w,b,mean,metric,projection=project_component(cached['features'],cached['parent'])
        encoder.head.weight.add_(w.T.to(encoder.head.weight))
        encoder.head.bias.add_(b.to(encoder.head.bias))
        folded_dev=encoder.head(dev_cache['features'])
        result['projected_initial']=dict(dev=metrics(folded_dev,dev_cache['labels']),projection=projection,
            dev_certificate=absorption_certificate(dev_cache['parent']+dev_cache['residual'],folded_dev))
    mean,metric,nuisance_geometry=paired_view_metric(clean_cache['features'],aug_cache['features'])
    result['nuisance_geometry']=nuisance_geometry
    weight,bias,head_fit=optimize_head(cached['features'],cached['parent']+cached['residual'],cached['labels'],
        encoder.head.weight,encoder.head.bias,mean,metric,ridge=1e-4)
    with torch.no_grad():
        encoder.head.weight.copy_(weight.to(encoder.head.weight));encoder.head.bias.copy_(bias.to(encoder.head.bias))
        result['conditioned_initial']=dict(head_fit=head_fit,
            fit=metrics(encoder.head(cached['features']),cached['labels']),
            dev=metrics(encoder.head(dev_cache['features']),dev_cache['labels']))
    result['conditioned_initial']['nuisance']=paired_logit_risk(
        encoder.head(clean_cache['features']).detach(),encoder.head(aug_cache['features']).detach(),clean_cache['labels'])
    result['conditioned_initial']['clean_fit']=metrics(encoder.head(clean_cache['features']).detach(),clean_cache['labels'])
    result['conditioned_initial']['augmented_fit']=metrics(encoder.head(aug_cache['features']).detach(),aug_cache['labels'])
    del clean_cache,aug_cache,clean_score,aug_score
    # Parent and fitting feature cache are never called by the learner or its deployed forward pass.
    del core,parent,cached,dev_cache
    opt=torch.optim.Adam(encoder.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr/10)
    result['initialization_wall_s']=time.perf_counter()-started;persist()
    initial=dict(epoch=0,fit=result['conditioned_initial']['clean_fit'],
        dev=result['conditioned_initial']['dev'],lr=None,
        training=dict(scope='Two-view convex readout initialization; no new encoder update'))
    result['curve'].append(initial);result['final']=initial;persist()
    torch.save(dict(encoder_state_dict=encoder.state_dict(),optimizer=opt.state_dict(),
        scheduler=scheduler.state_dict(),args=vars(a),encoder_config=cfg,result=result,
        order_rng=order_rng.bit_generator.state,augmentation_rng=aug_rng.bit_generator.state,
        torch_rng=torch.get_rng_state()),out.with_name(out.stem+'.epoch0.pt'))
    print(json.dumps({'projected_dev':result['projected_initial']['dev']['accuracy'],
        'conditioned_fit':result['conditioned_initial']['fit']['accuracy'],
        'conditioned_dev':result['conditioned_initial']['dev']['accuracy'],
        'parameters':result['deployed_parameters']}),flush=True)
    for epoch in range(1,a.epochs+1):
        encoder.train();order=order_rng.permutation(len(fit));train_start=time.perf_counter()
        acc=dict(nll=0.,steps=0,clipped=0,source_events=0,new_packets=0,
            source_table_projection_macs=0,source_transported_scalars=0,source_payload_sum_scalars=0,
            event_projection_macs=0,gate_macs=0,state_compositions=0,clock_candidates=0,
            layer_gradient_norm=[0.]*cfg['depth'])
        for start in range(0,len(fit),a.bs):
            rows=[augment_marked(fit[j],aug_rng) for j in order[start:start+a.bs]]
            opt.zero_grad(set_to_none=True);score,_,y,new=forward_query(encoder,rows,window)
            loss=F.cross_entropy(score,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for i,layer in enumerate(encoder.layers):
                acc['layer_gradient_norm'][i]+=float(torch.sqrt(sum(
                    (q.grad.square().sum() for q in layer.parameters() if q.grad is not None),score.new_tensor(0.))))
            norm=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.,error_if_nonfinite=True)
            acc['clipped']+=int(norm>1);opt.step()
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
            args=vars(a),encoder_config=cfg,result=result,order_rng=order_rng.bit_generator.state,
            augmentation_rng=aug_rng.bit_generator.state,torch_rng=torch.get_rng_state())
        torch.save(payload,out.with_suffix('.pt'))
        torch.save(payload,out.with_name(out.stem+f'.epoch{epoch}.pt'))
        print(json.dumps({'epoch':epoch,'fit':row['fit']['accuracy'],'dev':row['dev']['accuracy'],
            'dev_nll':row['dev']['nll'],'training_wall_s':acc['training_wall_s'],
            'rss_kb':result['max_rss_kb']}),flush=True)
    result['status']='completed';persist()


if __name__=='__main__':main()
