import numpy as np

from experiments.addressed_write_credit_reference import audit, objective


def test_identical_content_requires_addressed_future_credit():
    result=audit()
    assert result['value_only_credit']==[0.,0.]
    np.testing.assert_allclose(result['true_route_score_gradient'],[-.25,.25])
    assert result['max_abs_error']<1e-9


def test_no_write_delta_has_no_route_utility():
    scores=np.array([.2,-.7]); before=np.array([1.,-2.]); readout=np.array([3.,4.])
    _, losses=objective(scores,before,before.copy(),readout)
    np.testing.assert_array_equal(losses,np.full(2,readout@before))
