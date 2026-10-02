import numpy as np
import sys
sys.path.insert(0,'experiments')
from paired_parity_protocol import examples
from sleeping_machines.count_carrying_language import eval_stream_counts


def test_query_suffix_and_actual_causal_counts_cannot_distinguish_balanced_targets():
    rows=examples(gap=32,suffixes=4)
    fit=np.random.default_rng(9).integers(0,27,size=1024)
    for group in range(4):
        paired=[r for r in rows if r['group']==group]
        assert sorted(r['target'] for r in paired)==[0,0,1,1]
        assert all(r['inputs'][-33:]==paired[0]['inputs'][-33:] for r in paired)
        counts=[]
        for row in paired:
            inputs=row['inputs'];assert 26 not in inputs[:-1]
            stream=np.asarray([*inputs,row['target']])
            counts.append(eval_stream_counts(fit,stream,8)[:,len(inputs)-1])
            assert row['target']==row['bits'][0]^row['bits'][1]
        for c in counts[1:]:np.testing.assert_array_equal(c,counts[0])
