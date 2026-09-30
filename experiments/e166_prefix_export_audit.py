"""Verify the exported single deployment checkpoint on its declared dev split."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from e139_fine_packet_model import load_marked
from e150_single_state_shd import evaluate
from sleeping_machines.event_state import CoalescedEventStateEncoder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=Path('experiments/results/e166')/(args.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(args.tag).name!=args.tag or out.exists():raise ValueError('Unique output required')
    torch.set_num_threads(1);started=time.perf_counter()
    path=Path('experiments/results/e165/selected_prefix_20260930.json')
    reference=json.loads(path.read_text());checkpoint=path.with_suffix('.pt')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if reference['status']!='completed' or digest!=reference['exported_checkpoint_sha256']:
        raise ValueError('Completed unchanged export required')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['encoder_config']
    model=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=saved['actual_depth'])
    model.load_state_dict(saved['encoder_state_dict']);model.eval()
    optimizer=torch.optim.Adam(model.parameters(),lr=.000325);optimizer.load_state_dict(saved['optimizer'])
    items=load_marked(512,'val_spk',7)
    if [r[8] for r in items]!=reference['dev_absolute_ids']:raise ValueError('Changed dev sample')
    score=evaluate(model,items,4,cfg['window'])
    if score!=reference['final']['dev']:raise ValueError('Export changed evaluated classifier')
    result=dict(status='completed',dev=score,checkpoint_sha256=digest,
        optimizer_restored=True,deployed_parameters=sum(p.numel() for p in model.parameters()),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path('experiments/e150_single_state_shd.py'),Path('sleeping_machines/event_state.py'))},
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Restoration of exported six-block weights/old optimizer and exact reproduction of the already selected private-development score; no optimizer updates, new score, or official-test access',
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(correct=score['correct'],nll=score['nll'],
        optimizer_restored=True,parameters=result['deployed_parameters'])),flush=True)


if __name__=='__main__':main()
