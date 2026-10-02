"""Independent finite matrix checks of note91's local geometry, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unused tag required')
    started = time.perf_counter()
    rng = np.random.default_rng(910021)
    gaps = []
    # Direct joint solution and independently eliminated solution, including
    # full-sequence offset columns that change score rows.
    for score_preserving in (True, False):
        for _ in range(8):
            A = rng.normal(size=(7, 3)); B = rng.normal(size=(7, 2))
            if score_preserving:
                B[:2] = 0
            b = rng.normal(size=7); lam = .2; mu = .4
            H = A.T @ A + lam * np.eye(3)
            x0 = np.linalg.solve(H, A.T @ b)
            r = b - A @ x0
            S = B.T @ (np.eye(7) - A @ np.linalg.solve(H, A.T)) @ B + mu * np.eye(2)
            h = B.T @ r
            predicted_gap = .5 * h @ np.linalg.solve(S, h)
            C = np.column_stack((A, B))
            v = np.linalg.solve(C.T @ C + np.diag([lam]*3 + [mu]*2), C.T @ b)
            old = .5 * np.linalg.norm(A @ x0 - b)**2 + lam/2 * np.linalg.norm(x0)**2
            new = .5 * np.linalg.norm(C @ v - b)**2 + .5 * v @ np.diag([lam]*3 + [mu]*2) @ v
            np.testing.assert_allclose(old-new, predicted_gap, rtol=1e-11, atol=1e-12)
            assert predicted_gap >= 0
            gaps.append(float(predicted_gap))
    # Zero projection means exactly no improvement despite new independent columns.
    A = np.array([[1.], [0.], [0.]])
    B = np.array([[0.], [1.], [0.]])
    b = np.array([0., 0., 1.])
    assert float((B.T @ b)[0]) == 0
    np.testing.assert_allclose(np.linalg.lstsq(np.column_stack((A,B)), b, rcond=None)[0], 0)
    # Redundant columns add no unpenalized rank; they can reduce movement penalty.
    old_x = np.linalg.solve(np.array([[2.]]), np.array([1.]))
    joint = np.linalg.solve(np.array([[2.,1.],[1.,2.]]), np.array([1.,1.]))
    old_q = .5*(old_x[0]-1)**2 + .5*old_x[0]**2
    new_q = .5*(joint.sum()-1)**2 + .5*(joint @ joint)
    np.testing.assert_allclose([old_q,new_q],[.25,1/6], atol=1e-12)
    assert np.linalg.matrix_rank(np.array([[1.,1.]])) == 1
    # Exact determinant plus raw-offset saturation factor.
    rho=.3; omega=1.2; age=.7; u=.4; m=np.array([.8,-.2])
    beta=np.pi*np.tanh(u); angle=omega*age+beta
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    z=np.exp(-rho*age)*R @ m; J=np.array([[0.,-1.],[1.,0.]])
    da=-rho*z+omega*J@z; du=np.pi/np.cosh(u)**2*(J@z)
    det=float(np.linalg.det(np.column_stack((da,du))))
    expected=-rho*(z@z)*np.pi/np.cosh(u)**2
    np.testing.assert_allclose(det,expected,atol=1e-12)
    result=dict(status='completed',args=vars(a),contracts_passed=4,
        schur_direct_joint_cases=16,schur_benefit_range=[min(gaps),max(gaps)],
        zero_projection_no_benefit=True,redundant_regularized_costs=[float(old_q),float(new_q)],
        raw_offset_determinant=det,expected_determinant=float(expected),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['experiments/local_credit_geometry_contracts.py','experiments/theory/91_useful_freedom_and_local_credit_geometry.md']},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Finite matrix and rotor identities only; no optimizer/model fit, causal bottleneck proof or advantage claim.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
