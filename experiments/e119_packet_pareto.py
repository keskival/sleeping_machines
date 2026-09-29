"""Frozen-model accuracy/work/latency tradeoff under causal input coalescing.

Only winning payloads propagate inside the network. Coalescing changes the
input representation, so unlike the exact scan optimization it may change
accuracy. All selection here is on development speakers, not official test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from e117_serial_event_shd import load_items
from e118_race_carrier_shd import RaceNet, evaluate


def coalesce(item, factor):
    b,t,c,y,index = item
    if factor == 1: return item
    ticks = np.rint(t/.01).astype(np.int64)
    closing = ((ticks+factor-1)//factor)*factor
    pairs, inv = np.unique(np.stack((closing,b),1),axis=0,return_inverse=True)
    count = np.zeros(len(pairs),dtype=np.float32)
    np.add.at(count,inv,c)
    times = (pairs[:,0]*.01).astype(np.float32)
    assert float(count.sum()) == float(c.sum())
    assert np.all(times[inv]+1e-6 >= t)
    assert np.max(times[inv]-t) <= (factor-1)*.01+1e-6
    return pairs[:,1],times,count,y,index


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--checkpoint',required=True)
    a = p.parse_args()
    out = Path('experiments/results/e119')/(a.tag+'.json')
    if out.exists(): raise FileExistsError(out)
    saved = torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    net = RaceNet(depth=saved['args']['depth'],memory_backend='linear')
    net.load_state_dict(saved['state_dict'])
    items = load_items(40,.01,256,'val_spk',7)
    all_items = {f:[coalesce(x,f) for x in items] for f in (1,2,4,8)}
    # Warm each representation, then alternate measurement order.
    for data in all_items.values(): evaluate(net,data[:4],4)
    rows, timing = {}, {f:[] for f in all_items}
    for repeat in range(3):
        for factor in (list(all_items) if repeat%2 else list(reversed(all_items))):
            start = time.perf_counter()
            value = evaluate(net,all_items[factor],4)
            timing[factor].append(time.perf_counter()-start)
            if factor in rows:
                assert value['predictions'] == rows[factor]['predictions']
            rows[factor] = value
    ref = rows[1]
    for factor,row in rows.items():
        row['packet_window_ms'] = 10*factor
        row['maximum_additional_input_delay_ms'] = 10*(factor-1)
        row['wall_s_repetitions'] = timing[factor]
        row['median_wall_s'] = float(np.median(timing[factor]))
        row['packet_ratio_to_10ms'] = row['packets']/ref['packets']
        row['corrected_vs_10ms'] = sum(p==x[3] and q!=x[3] for p,q,x in zip(row['predictions'],ref['predictions'],items))
        row['lost_vs_10ms'] = sum(p!=x[3] and q==x[3] for p,q,x in zip(row['predictions'],ref['predictions'],items))
    result = {'status':'completed','args':vars(a),'checkpoint_epoch':saved.get('epoch'),
              'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'source_sha256':{f:hashlib.sha256(Path('experiments',f).read_bytes()).hexdigest() for f in
                               ('e119_packet_pareto.py','e119_linear_event_scan.py','e118_race_carrier_shd.py','e117_serial_event_shd.py')},
              'protocol':'Same frozen model, held-out speakers 3/6, 256 development examples, batch4, 3 warm repetitions; no official test access',
              'dev_ids':[x[4] for x in items],'dev_labels':[x[3] for x in items],
              'rows':rows,'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'energy_joules':None,'note':'Only model evaluation is timed; data loading/coalescing excluded. CPU time is not energy.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({f:{k:r[k] for k in ('accuracy','nll','packets','packet_ratio_to_10ms','median_wall_s','corrected_vs_10ms','lost_vs_10ms')} for f,r in rows.items()}),flush=True)

if __name__ == '__main__': main()
