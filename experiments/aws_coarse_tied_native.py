"""Coarse packets with private receiver state and shared receiver maps."""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
import dvs_tied_pool_benchmark as T

N=A.N; B=A.B; C=T
parser=A.parser
load=A.load
coalesce=A.coalesce
BASE_LOAD=A.BASE_LOAD


def sources():
    names=['experiments/aws_coarse_tied_native.py','experiments/aws_coarse_tied_contracts.py','experiments/theory/aws_20261002_coarse_tied_protocol.md']
    return {**A.sources(),**T.sources(),**{n:N.sha(ROOT/n) for n in names}}


@contextmanager
def activate():
    old=N.load,N.make_model,N.train_window,N.sources
    N.load,N.make_model,N.train_window,N.sources=load,T.make_model,B.train_window,sources
    try:yield
    finally:N.load,N.make_model,N.train_window,N.sources=old


def run(args,directory=None):
    with activate():return N.run(args,directory)


if __name__=='__main__':run(parser().parse_args())
