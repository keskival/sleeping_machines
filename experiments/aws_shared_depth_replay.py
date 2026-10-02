"""Deep private event state with depth-shared receiver maps and matched credits."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_deep_replay as D

SHARED=('input','output','gate','control','key_read')
BASE_MAKE=D.BL.make_model
BASE_SOURCES=D.sources


def make_model(a,fast=True):
    model=BASE_MAKE(a,fast)
    for head in range(model.heads):
        master=model.units[0][head][0][0]
        for depth in model.units:
            for unit in depth[head][0]:
                for name in SHARED:setattr(unit,name,getattr(master,name))
    return model


def sources():
    files=['experiments/aws_shared_depth_replay.py','experiments/aws_shared_depth_replay_contracts.py','experiments/theory/aws_20261002_shared_depth_replay_protocol.md']
    return {**BASE_SOURCES(),**{f:D.N.sha(ROOT/f) for f in files}}


def parse(argv=None):
    a=D.parse(argv);a.depth_shared_receiver_maps=True;return a


def run(a,directory=None):
    old=D.BL.make_model,D.sources
    D.BL.make_model,D.sources=make_model,sources
    try:return D.run(a,directory)
    finally:D.BL.make_model,D.sources=old


if __name__=='__main__':run(parse())
