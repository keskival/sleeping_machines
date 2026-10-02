"""Analytical joint-score identities, including a hard future-time threshold."""
import numpy as np


def finite_difference(objective,scores):
    eye=np.eye(len(scores));h=1e-5
    return np.asarray([(objective(scores+h*e)-objective(scores-h*e))/(2*h) for e in eye])


def test_linear_time_surrogate_joint_score_matches_exact_objective():
    scores=np.asarray([.3,-.4,.8]);a=np.asarray([1.,4.,2.]);b=np.asarray([.2,.7,1.1])
    rates=np.exp(scores);total=rates.sum();p=rates/total
    exact=p*(a-p@a+(b-2*(p@b))/total)
    objective=lambda s:(np.exp(s)/np.exp(s).sum())@(a+b/np.exp(s).sum())
    np.testing.assert_allclose(exact,finite_difference(objective,scores),rtol=1e-8,atol=1e-9)
    tau,weights=np.polynomial.laguerre.laggauss(32)
    integrated=sum(w*(p*(a+b*t/total)-p*t*(p@(a+b*t/total))) for t,w in zip(tau,weights))
    np.testing.assert_allclose(integrated,exact,rtol=1e-10,atol=1e-11)


def test_jump_loss_has_clock_credit_missing_from_smooth_interiors():
    scores=np.zeros(2);rates=np.exp(scores);total=rates.sum();p=rates/total;t0=1.
    exact=p[1]*np.exp(-total*t0)*(np.asarray([0.,1.])-p-rates*t0)
    objective=lambda s:(np.exp(s)[1]/np.exp(s).sum())*np.exp(-np.exp(s).sum()*t0)
    np.testing.assert_allclose(exact,finite_difference(objective,scores),rtol=1e-8,atol=1e-9)
    omitted_jump=p[1]*np.exp(-total*t0)*(np.asarray([0.,1.])-p)
    assert exact[1]<0<omitted_jump[1]
    assert exact.sum()<0 and abs(omitted_jump.sum())<1e-15
