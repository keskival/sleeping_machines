"""Integrated coarse native core with a zero-nested quadratic local readout."""
from contextlib import contextmanager
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import aws_coarse_native as A
from sleeping_machines.quadratic_native_readout import QuadraticNativeReadout
N=A.N

def make_model(a,fast=True):
    return replace_head(A.C.make_model(a,fast))

def replace_head(model):
    model.head=QuadraticNativeReadout(model.head)
    return model

def sources():
    files=['sleeping_machines/quadratic_native_readout.py','experiments/aws_quadratic_native.py','experiments/aws_quadratic_native_contracts.py','experiments/theory/aws_20261002_quadratic_native_protocol.md']
    return {**A.sources(),**{f:N.sha(ROOT/f) for f in files}}

def parser():return A.parser()

@contextmanager
def activate():
    with A.activate():
        old=N.make_model,N.sources
        N.make_model,N.sources=make_model,sources
        try:yield
        finally:N.make_model,N.sources=old

def run(a,directory=None):
    if a.terminal_risk!='local':raise ValueError('Only the preregistered local-credit comparison is admitted')
    with activate():return N.run(a,directory)
if __name__=='__main__':run(parser().parse_args())
