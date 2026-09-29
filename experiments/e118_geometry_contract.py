"""Finite linear witnesses for THEORY §§171–172; no SHD training."""
import json
from pathlib import Path

import numpy as np


def main():
    out = Path(__file__).parent/"results"/"e118"/"optional_control_geometry.json"
    if out.exists():
        raise FileExistsError(out)
    a = np.array([[.9,0.],[.2,1.]])
    b0, b1 = np.array([[1.],[0.]]),np.array([[0.],[1.]])
    m = np.diag([1.,2.])
    j = np.column_stack((a@b0,b1))
    gramian = a@(b0@b0.T)@a.T + 2*b1@b1.T
    recursion_error = float(np.max(np.abs(gramian-j@m@j.T)))
    direction = np.array([1.,2.])
    support = np.sqrt(direction@gramian@direction)
    control = m@j.T@direction/support
    support_error = float(abs(direction@j@control-support))
    budget_error = float(abs(control@np.linalg.solve(m,control)-1))
    gradient = np.array([1.,-.4])
    curvature = np.diag([.2,1.3])
    penalty = .7
    hessian = j.T@curvature@j+penalty*np.linalg.inv(m)
    response = j.T@gradient
    optimum = -np.linalg.solve(hessian,response)
    reserve = .5*response@np.linalg.solve(hessian,response)
    actual_gain = -(response@optimum+.5*optimum@hessian@optimum)
    reserve_error = float(abs(reserve-actual_gain))
    shared_response = 1.-1.
    actual_shared_gramian = shared_response**2
    falsely_independent = 1.**2+(-1.)**2
    assert max(recursion_error,support_error,budget_error,reserve_error)<1e-12
    result = {"gramian_recursion_error":recursion_error,"support_error":support_error,
              "control_budget_error":budget_error,"quadratic_reserve_error":reserve_error,
              "reserve":float(reserve),"shared_cancelled_gramian":actual_shared_gramian,
              "incorrect_independent_gramian":falsely_independent,
              "equal_trace_directional_supports":[1.,0.],
              "duplicate_route_union_support":1.,"incorrect_summed_support":float(np.sqrt(2)),
              "scope":"Exact finite linear witnesses; not a trained optionality estimator"}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
