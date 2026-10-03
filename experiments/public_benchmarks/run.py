"""Guarded public archive screen/final fit using the existing native event core.

Run only in unique run_safe queues. Screen never opens the official TEST file.
"""
import argparse,copy,json,math,platform,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.public_benchmarks.data import EXPECTED,load_train,read_ts,sha
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.compiled_episodes import compiled_logits
from sleeping_machines.batched_episodes import batched_logits
from race_language_screen import capture


def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def source_hashes():
    names=['experiments/public_benchmarks/run.py','experiments/public_benchmarks/data.py',
           'sleeping_machines/addressed_event_heads.py','sleeping_machines/compiled_episodes.py',
           'sleeping_machines/batched_episodes.py','sleeping_machines/fast_native_core.py',
           'sleeping_machines/parallel_stream_language.py','sleeping_machines/operation_audit.py',
           'experiments/race_language_screen.py']
    return {n:sha(ROOT/n) for n in names}


def encode(rows,center,scale,horizon):
    output=[]
    for row in rows:
        x=(np.asarray(row['values'],np.float32)-center)/scale
        events=[((k+1)/(horizon+1),np.r_[value,0.].astype(np.float32)) for k,value in enumerate(x)]
        events.append((max(1.,(len(x)+1)/(horizon+1)),np.r_[np.zeros(len(center)),1.].astype(np.float32)))
        output.append(dict(target=row['target'],identity=row['identity'],events=events))
    return output


@torch.no_grad()
def evaluate(model,rows,batch):
    probabilities=[];loss=0.;correct=0
    for start in range(0,len(rows),batch):
        group=rows[start:start+batch];model.eval();logits=compiled_logits(model,group,314159,route_credit=None)
        y=torch.tensor([r['target'] for r in group]);loss+=float(F.cross_entropy(logits,y,reduction='sum'))
        correct+=int((logits.argmax(-1)==y).sum());probabilities.extend(logits.softmax(-1).tolist())
    return dict(targets=len(rows),accuracy=correct/len(rows),nll=loss/len(rows),probabilities=probabilities,
                identities=[r['identity'] for r in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--dataset',choices=EXPECTED,required=True)
    p.add_argument('--stage',choices=['screen','final'],default='screen');p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--payload',type=int,default=16);p.add_argument('--depth',type=int,default=2);p.add_argument('--pool',type=int,default=2)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=6);p.add_argument('--selection');p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);started=time.perf_counter();sources=source_hashes()
    directory=ROOT/'experiments/results/public_benchmarks';directory.mkdir(exist_ok=True)
    out=directory/(a.tag+'.json');checkpoint=directory/(a.tag+'.pt');assert not out.exists(),'Immutable completed tag'
    manifest=json.loads((ROOT/'experiments/public_benchmarks/data_manifest.json').read_text())
    entry=next(r for r in manifest['datasets'] if r['dataset']==a.dataset)
    train_path=ROOT/'data/public_benchmarks/raw'/f'{a.dataset}_TRAIN.ts';assert sha(train_path)==entry['train_sha256']
    raw,mapping,_=load_train(a.dataset)
    selection=None
    if a.stage=='final':
        if not a.selection:raise ValueError('Final requires a completed DEV-selected screen')
        selection=json.loads((ROOT/a.selection).read_text());assert selection['status']=='completed' and selection['args']['stage']=='screen'
        for key in ['dataset','payload','depth','pool','heads','lr','batch']:
            assert selection['args'][key]==vars(a)[key],key
        assert a.epochs==selection['best_epoch'],'Final epoch count must be selected without TEST'
        fit_raw=raw;dev_raw=[]
    else:
        fit_raw=[raw[i] for i in entry['fit_indices']];dev_raw=[raw[i] for i in entry['dev_indices']]
    values=np.concatenate([np.asarray(r['values']) for r in fit_raw],axis=0)
    center=values.mean(0).astype(np.float32);scale=np.maximum(values.std(0),1e-6).astype(np.float32)
    horizon=max(len(r['values']) for r in fit_raw);fit=encode(fit_raw,center,scale,horizon);dev=encode(dev_raw,center,scale,horizon)
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed+100)
    model=fast_class(AddressedEventHeads)(sources=1,content_dim=EXPECTED[a.dataset][2]+1,classes=len(mapping),
                              payload=a.payload,depth=a.depth,heads=a.heads,pool=a.pool)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr);first=0;curve=[];best=None;traced=[];presentations=0;prior_wall=0.
    configuration={k:v for k,v in vars(a).items() if k!='resume'}
    if a.resume:
        saved=torch.load(checkpoint,weights_only=False);assert saved['args']==configuration and saved['source_sha256']==sources
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
        first=saved['epoch'];curve=saved['curve'];best=saved['best'];traced=saved['traced'];presentations=saved['presentations'];prior_wall=saved['wall_s']
    elif checkpoint.exists():raise ValueError('Existing checkpoint requires --resume')
    for epoch in range(first,a.epochs):
        order=rng.permutation(len(fit));losses=[]
        for start in range(0,len(fit),a.batch):
            group=[fit[i] for i in order[start:start+a.batch]];seed=100000+a.seed*1000+epoch*100000+start
            box={}
            def step(logits_fn=compiled_logits):
                model.train();optimizer.zero_grad(set_to_none=True)
                z=logits_fn(model,group,seed,route_credit='linear');y=torch.tensor([r['target'] for r in group])
                loss=F.cross_entropy(z,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();box['loss']=float(loss.detach())
            if start==0:
                work=capture(lambda:step(batched_logits));traced.append(dict(targets=len(group),events=sum(len(r['events']) for r in group),work=work))
            else:step()
            losses.append(box['loss']);presentations+=len(group)
        score=evaluate(model,dev,a.batch) if dev else None
        row=dict(epoch=epoch+1,mean_batch_train_nll=float(np.mean(losses)),dev=score)
        curve.append(row)
        if score and (best is None or score['nll']<best['dev']['nll']):
            best=dict(epoch=epoch+1,dev=score,weights=copy.deepcopy(model.state_dict()))
        saved=dict(args=configuration,source_sha256=sources,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                   epoch=epoch+1,curve=curve,best=best,traced=traced,presentations=presentations,wall_s=prior_wall+time.perf_counter()-started)
        temp=checkpoint.with_suffix('.tmp');torch.save(saved,temp);temp.replace(checkpoint)
        print(json.dumps(dict(epoch=epoch+1,dev_accuracy=score['accuracy'] if score else None,wall_s=saved['wall_s'])),flush=True)
    test=None
    if a.stage=='final':
        test_path=ROOT/'data/public_benchmarks/raw'/f'{a.dataset}_TEST.ts';assert sha(test_path)==entry['test_sha256']
        test_raw,_=read_ts(test_path);assert len(test_raw)==EXPECTED[a.dataset][1]
        for i,r in enumerate(test_raw):r.update(target=mapping[r['label']],identity=f'TEST:{i}')
        test=evaluate(model,encode(test_raw,center,scale,horizon),a.batch)
    fit_work=sum(t['work']['arithmetic_flops']+t['work']['special_function_evaluations'] for t in traced)
    fit_traced=sum(t['targets'] for t in traced);per=fit_work/fit_traced
    result=dict(status='completed',args=vars(a),source_sha256=sources,data_manifest_sha256=sha(ROOT/'experiments/public_benchmarks/data_manifest.json'),
        selection_parent_sha256=sha(ROOT/a.selection) if a.selection else None,parameters=sum(q.numel() for q in model.parameters()),
        best_epoch=best['epoch'] if best else a.epochs,development=best['dev'] if best else None,test=test,
        fitting_presentations=presentations,curve=curve,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
        work=dict(fit_flops_per_presented_target_estimate=per,whole_fit_flops_estimate=per*presentations,
                  traced_fitting_targets=fit_traced,scope='One full eager optimizer window per epoch extrapolated per target. Variable/padded lengths vary; shape-mean estimate, not exact whole fit. DEV/test/packing/compilation/traffic/energy separate.'),
        capacity=dict(receivers=a.depth*a.heads*a.pool,scored_keys_per_event=a.depth*a.heads*a.pool,selected_writes_per_event=a.depth*a.heads),
        protocol=dict(split='Official archive TRAIN/TEST; fixed stratified20% of TRAIN for DEV; full TRAIN refit only after configuration/epoch selection.',
            encoding='Synchronous real-valued channels, FIT-only centering/scaling, ordinal timestamps normalized by FIT maximum length, explicit terminal query. No claim of original physical timestamps.',
            credit='Native factorized temporal races plus linear local-expectation categorical message credit; whole episode BPTT, persistent private state reset per series.',
            test_access='None in screen; once after final refit.',inference_work='Pending trained-state trace; no inference advantage claimed.'),
        hardware=dict(platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),
        wall_s=prior_wall+time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic(out,result)

if __name__=='__main__':main()
