"""Read-only witness: identical payloads can have unequal addressed write utility."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def objective(scores, before, writes, readout):
    rates = np.exp(scores); p = rates/rates.sum()
    branch_costs = np.array([readout @ before + readout[i]*(writes[i]-before[i])
                             for i in range(len(scores))])
    return p @ branch_costs, branch_costs


def audit():
    scores=np.zeros(2); before=np.zeros(2); writes=np.ones(2); readout=np.array([0.,1.])
    _, losses=objective(scores,before,writes,readout)
    p=np.array([.5,.5]); exact=p*(losses-p@losses)
    finite=np.array([(objective(scores+np.eye(2)[i]*1e-5,before,writes,readout)[0]-
                      objective(scores-np.eye(2)[i]*1e-5,before,writes,readout)[0])/2e-5 for i in range(2)])
    np.testing.assert_allclose(exact,finite,atol=1e-10,rtol=1e-9)
    # Identical values AND proposed memory contents. Only write location differs.
    values=np.zeros((2,1)); proposed_memory=np.ones((2,1))
    assert np.array_equal(values[0],values[1])
    assert np.array_equal(proposed_memory[0],proposed_memory[1])
    value_teacher=np.zeros(2)
    assert np.linalg.norm(exact)>0 and np.linalg.norm(value_teacher)==0
    # A state adjoint at each address gives exact scalar branch utility for
    # this LINEAR future readout, including the write delta relative to old state.
    utility=readout*(writes-before)
    np.testing.assert_array_equal(utility,losses)
    return dict(status='completed',scope='Deterministic one-race addressed-state information-path witness; no fitting or full-model quality claim',
        before_state=before.tolist(),candidate_writes=writes.tolist(),future_readout=readout.tolist(),
        branch_losses=losses.tolist(),true_route_score_gradient=exact.tolist(),finite_difference=finite.tolist(),
        value_only_credit=value_teacher.tolist(),state_adjoint_branch_utilities=utility.tolist(),
        max_abs_error=float(np.max(np.abs(exact-finite))),
        conclusion='Write identity matters even when immediate values and proposed memory contents coincide; concatenating payload and memory content alone cannot expose the address effect',
        limitation='Future adjoints at losing addresses must be available/estimated and charged; nonlinear and stochastic suffixes need further credit')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    path=Path(parser.parse_args().output)
    if path.exists():raise ValueError('Never overwrite an audit')
    result=audit();result['source_sha256']={'experiments/addressed_write_credit_reference.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
