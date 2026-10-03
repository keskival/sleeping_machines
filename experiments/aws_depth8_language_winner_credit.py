"""Same full conditional language replay, reusing factual winner returns."""
from pathlib import Path
import hashlib
import aws_depth8_language_credit as B
import aws_language_winner_reuse as W
import causal_language_replay_accumulator as A
ORIGINAL_SOURCES=B.sources
ROOT=Path(__file__).resolve().parents[1]

def sources():
    files=['experiments/aws_depth8_language_winner_credit.py','experiments/aws_language_winner_reuse.py','experiments/aws_language_winner_reuse_contracts.py','experiments/aws_language_winner_driver_contracts.py']
    return {**ORIGINAL_SOURCES(),**{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}

def main():
    old=A.batched_objective,B.sources
    A.batched_objective=W.objective;B.sources=sources
    try:B.main()
    finally:A.batched_objective,B.sources=old
if __name__=='__main__':main()
