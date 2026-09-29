"""Evidence-free shared event language learning; bounded depth diagnostic."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from e120_shared_tasks import Example, prefix, text_slice
from e120_shared_bench import inputs, loss_for, calibrate, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--depth', type=int, required=True)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--fit', type=int, default=8192)
    ap.add_argument('--dev', type=int, default=1024)
    ap.add_argument('--context', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=.003)
    ap.add_argument('--seed', type=int, default=6)
    a = ap.parse_args()
    if Path(a.tag).name != a.tag or min(a.fit,a.dev,a.context,a.depth,a.epochs,a.bs) < 1:
        raise ValueError('Invalid settings')
    if a.fit+a.context > 90000000 or a.dev+a.context > 5000000:
        raise ValueError('Requested budget crosses a reserved split')
    out = Path('experiments/results/e133')/(a.tag+'.json')
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)

    def examples(offset, n):
        raw = text_slice(offset, n+a.context)
        return [Example(prefix(raw[i-a.context:i]), int(raw[i]), str(offset+i))
                for i in range(a.context, a.context+n)]

    fit, dev = examples(0, a.fit), examples(90000000, a.dev)
    assert all(r.evidence is None for r in fit+dev)
    config = dict(bands=27,classes=27,groups=1,readout='last',dim=a.dim,
                  depth=a.depth,memory_backend='linear',evidence_count=0,
                  cf_credit=True,value_backward='full')
    model = SharedEventModel(**config)
    assert model.evidence_count == 0 and model.phase_memory is None
    calibration = calibrate(model, fit, a.bs)
    initial_state = {n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(),lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=a.lr/10)
    order_rng = np.random.default_rng(a.seed+10)
    sources = [Path(__file__),Path('experiments/e120_shared_bench.py'),
               Path('experiments/e120_shared_tasks.py'),*Path('sleeping_machines').glob('*.py')]
    data_hash = hashlib.sha256()
    with Path('data/text8/text8').open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            data_hash.update(chunk)

    def measured(rows):
        r = evaluate(model,rows,a.bs)
        r['bpc'] = r['nll']/math.log(2)
        if rows is fit:
            r.pop('predictions',None)
        return r

    record = dict(status='running',args=vars(a),config=config,parameters=sum(p.numel() for p in model.parameters()),
                  calibration=calibration,protocol={'dataset':'text8','token_unit':'character',
                  'training_targets':[a.context,a.context+a.fit],
                  'validation_targets':[90000000+a.context,90000000+a.context+a.dev],
                  'context':a.context,'expert_modules':[],'official_test_read':False,
                  'query_execution':'Every target replays its observed context; no persistent scheduler',
                  'gradient_rule':'Current deterministic hard-race surrogate; E132 stochastic joint credit is not used'},
                  data_sha256=data_hash.hexdigest(),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  initial={'fit':measured(fit),'dev':measured(dev)},curve=[],
                  hardware={'platform':platform.platform(),'torch':torch.__version__,'device':'cpu','threads':1},
                  energy_joules=None)

    def persist():
        temp=out.with_suffix('.json.tmp')
        temp.write_text(json.dumps(record,indent=2)+'\n')
        temp.replace(out)

    persist()
    print(json.dumps({'depth':a.depth,'initial_dev_bpc':record['initial']['dev']['bpc'],'parameters':record['parameters']}),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train()
        total,packets,scans,values,steps=0.,0,0,0,0
        grad={n:np.zeros(a.depth) for n in ('value','route','log_tau')}
        order=order_rng.permutation(len(fit))
        for i in range(0,len(order),a.bs):
            rows=[fit[j] for j in order[i:i+a.bs]]
            scores,_,stats,_=model(**inputs(rows))
            loss=loss_for(scores,rows)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite language loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for j,layer in enumerate(model.layers):
                for name in grad:
                    g=getattr(layer,name).grad
                    if g is not None:
                        grad[name][j]+=float(g.norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=float(loss.detach())*len(rows)
            packets+=stats['packets']
            scans+=sum(s['scan_compositions'] for s in stats['layers'])
            values+=sum(s['value_evaluations'] for s in stats['layers'])
            steps+=1
        scheduler.step()
        row=dict(epoch=epoch,online_nll=total/len(fit),fit=measured(fit),dev=measured(dev),
                 mean_gradient_norm={n:(g/steps).tolist() for n,g in grad.items()},
                 parameter_l2_changes={n:float((p.detach()-initial_state[n]).norm())
                                       for n,p in model.named_parameters()},
                 training_prefix_packets=packets,training_scan_compositions=scans,
                 training_value_evaluations=values,optimizer_steps=steps)
        record['curve'].append(row)
        persist()
        torch.save({'state_dict':model.state_dict(),'config':config,'args':vars(a),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'order_rng':order_rng.bit_generator.state,'epoch':epoch},out.with_suffix('.progress.pt'))
        print(json.dumps({'depth':a.depth,'epoch':epoch,'fit_bpc':row['fit']['bpc'],'dev_bpc':row['dev']['bpc'],
                          'value_gradient_norms':row['mean_gradient_norm']['value']}),flush=True)
    model.eval()
    # Diagnostic only: the same fitted weights see the last observed character.
    # This is not a retrained reference and differs from the training context.
    last_only=[Example(prefix([int(r.prefix.channels[-1])]),r.label,r.identity) for r in dev]
    record.update(final=record['curve'][-1],last_character_only=measured(last_only),
                  diagnostic_note='Frozen input-context deletion, not a retrained baseline; constant-count dependence is projected at fitting calibration.',
                  wall_s=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  status='completed',total_training_presentations=a.fit*a.epochs,
                  scope='Single-seed bounded neural trainability/depth screen. No specialized prediction expert, full test score, matched capacity comparison or scaling/energy claim.')
    torch.save({'state_dict':model.state_dict(),'config':config,'args':vars(a),
                'optimizer':optimizer.state_dict(),'protocol':record['protocol']},out.with_suffix('.pt'))
    persist()
    print(json.dumps({'depth':a.depth,'final_dev_bpc':record['final']['dev']['bpc'],
                      'last_character_only_bpc':record['last_character_only']['bpc'],'wall_s':record['wall_s']}),flush=True)


if __name__=='__main__':
    main()
