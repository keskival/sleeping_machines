"""Paired nonlocal labels with identical marginals and query counts.

Inputs contain each bit followed by its complement, so unigram counts also
match across a quartet. Restricted local query counts are controls, not a
bound on arbitrary algorithms inspecting the complete count table or prefix.
"""
import hashlib
import json
import numpy as np
from paired_parity_protocol import examples


def joint_examples(gap=8, groups=8, seed=71):
    rows=examples(gap=gap,suffixes=groups,seed=seed)
    for row in rows:
        left,right=row['bits']
        row['inputs']=[24,left,1-left,25,right,1-right,*row['inputs'][4:]]
    return rows


def data_hash(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_pairs(rows, max_order=8):
    if not 0<=max_order<=8:raise ValueError('Bounded count orders 0..8')
    groups=sorted({row['group'] for row in rows})
    for group in groups:
        quartet=[r for r in rows if r['group']==group]
        if len(quartet)!=4 or sorted(r['target'] for r in quartet)!=[0,0,1,1]:
            raise ValueError('Complete balanced quartet required')
        reference=quartet[0]['inputs']
        for row in quartet:
            x=row['inputs']
            if 26 in x[:-1] or x[-1]!=26 or len(x)!=len(reference):raise ValueError('Unique query cue required')
            if row['target']!=row['bits'][0]^row['bits'][1]:raise ValueError('Invalid joint label')
            if x[-max_order:]!=reference[-max_order:] and max_order:raise ValueError('Unmatched query suffix')
            if not np.array_equal(np.bincount(x,minlength=27),np.bincount(reference,minlength=27)):
                raise ValueError('Unmatched observed symbol totals')
    return dict(groups=len(groups),targets=len(rows),query_suffix_order=max_order,
        identical_symbol_totals=True,balanced_query_suffix_logloss_lower_bound_bits=1.,
        full_prefix_oracle_logloss_bits=0.,
        scope='Bound for predictors restricted to common query suffix/count vectors including root marginals; not arbitrary inspection of prefix or unrelated count addresses.')
