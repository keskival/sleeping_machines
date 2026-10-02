"""Depth8 matched credits with private or depth-shared receiver maps."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_shared_depth_replay as S
import aws_deep_replay as D

BASE_SOURCES=D.sources
BASE_MAKE=S.BASE_MAKE


def parse(argv=None):
    p=D.parser();p.add_argument('--credit-mode',choices=['teacher','factorized','replay'],required=True)
    p.add_argument('--receiver-sharing',choices=['private','depth'],required=True)
    a=p.parse_args(argv);a.route_credit=a.credit_mode=='replay';a.route_races=0;a.terminal_risk='local'
    if a.depth!=8 or a.bins!=4 or a.input_noise or a.weight_decay:
        raise ValueError('Matched depth8/coarse4/no-new-regularization protocol required')
    return a


def make_model(a,fast=True):
    return S.make_model(a,fast) if a.receiver_sharing=='depth' else BASE_MAKE(a,fast)


def sources():
    files=['experiments/aws_depth8_replay.py','experiments/aws_depth8_replay_contracts.py','experiments/theory/aws_20261002_depth8_replay_protocol.md']
    return {**S.sources(),**{f:D.N.sha(ROOT/f) for f in files}}


def run(a,directory=None):
    old=D.BL.make_model,D.sources;D.BL.make_model,D.sources=make_model,sources
    try:return D.run(a,directory)
    finally:D.BL.make_model,D.sources=old


if __name__=='__main__':run(parse())
