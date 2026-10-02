"""Matched deep coarse models: original teacher, factorized control, full replay."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import dvs_batched_reg_benchmark as R
import dvs_batched_le_benchmark as BL

N=A.N
BASE_EVALUATE=N.evaluate
parser=R.parser


def parse(argv=None):
    p=parser();p.add_argument('--credit-mode',choices=['teacher','factorized','replay'],required=True)
    a=p.parse_args(argv);a.route_credit=a.credit_mode=='replay';a.route_races=0;a.terminal_risk='local'
    if a.depth<4 or a.bins!=4 or a.weight_decay or a.input_noise:
        raise ValueError('First matched deep protocol: depth>=4,coarse4,no new regularization')
    return a


def sources():
    files=['experiments/aws_deep_replay.py','experiments/aws_deep_replay_contracts.py','experiments/theory/aws_20261002_deep_replay_protocol.md']
    return {**R.sources(),**{f:N.sha(ROOT/f) for f in files}}


@contextmanager
def activate(a):
    with R.activate(a):
        old=N.sources,N.train_window,N.evaluate
        N.sources=sources
        if a.credit_mode=='teacher':N.train_window,N.evaluate=A.B.train_window,BASE_EVALUATE
        try:yield
        finally:N.sources,N.train_window,N.evaluate=old


def run(a,directory=None):
    directory=Path(directory) if directory else ROOT/'experiments/results/dvs_native'
    before=BL.REPLAYS[0];previous_targets=0
    if a.resume:
        import torch
        previous_targets=torch.load(directory/f'{a.tag}.progress.pt',weights_only=False,map_location='cpu')['result']['activity']['targets']
    with activate(a):result=N.run(a,directory)
    replay_per_target=5*a.depth*a.heads*a.pool if a.credit_mode=='replay' else 0
    executed=BL.REPLAYS[0]-before
    assert executed==(result['activity']['targets']-previous_targets)*replay_per_target
    result['deep_replay_activity']=dict(mode=a.credit_mode,factual_races_per_prefix=5*a.depth*a.heads,
        replay_lanes_this_process=executed,total_fitting_replay_lanes=result['activity']['targets']*replay_per_target,
        replay_events=result['activity']['targets']*replay_per_target*5,
        scope='Executed lane count checked against processed targets; replay arithmetic included in traced forward stage')
    (directory/f'{a.tag}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':run(parse())
