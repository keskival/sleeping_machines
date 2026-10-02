"""Balanced prefix-parity examples with identical query suffixes.

Inputs end in a cue, targets are never included in the observed input. For each
noise suffix, all four bit pairs occur; their query-time local count vectors
are equal for orders <= gap+1 (the cue never occurs in the prefix).
"""
import numpy as np


def examples(gap=32, suffixes=8, seed=71):
    if gap<8 or suffixes<1:raise ValueError('At least8 suffix tokens and one paired suffix required')
    rng=np.random.default_rng(seed);rows=[]
    for group in range(suffixes):
        suffix=rng.integers(2,24,size=gap).tolist()
        for left,right in ((0,0),(0,1),(1,0),(1,1)):
            inputs=[24,left,25,right,*suffix,26]
            rows.append(dict(group=group,inputs=inputs,target=left^right,bits=[left,right]))
    return rows
