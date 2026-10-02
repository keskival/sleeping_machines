import itertools
import torch


def test_individual_information_and_gradient_are_zero_but_joint_interaction_has_credit():
    rows=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=torch.float64)
    a,b,noise=rows.T;y=a*b
    assert (y*a).mean()==(y*b).mean()==0
    for second,expected in ((noise,0.),(b,-.5)):
        weight=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        loss=torch.nn.functional.softplus(-y*weight*a*second).mean()
        loss.backward();torch.testing.assert_close(weight.grad,torch.tensor(expected,dtype=torch.float64),rtol=0,atol=0)
    w=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    torch.nn.functional.softplus(-y*(rows[:,:2]@w)).mean().backward()
    torch.testing.assert_close(w.grad,torch.zeros_like(w),rtol=0,atol=0)


def test_independent_relevant_delivery_attenuates_joint_credit_by_product_probability():
    rows=torch.tensor(list(itertools.product((-1.,1.),repeat=4)),dtype=torch.float64)
    a,b,d1,d2=rows.T;y=a*b;r1,r2=.2,.3
    weight=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    loss=0.
    for first,p1 in ((a,r1),(d1,1-r1)):
        for second,p2 in ((b,r2),(d2,1-r2)):
            loss=loss+p1*p2*torch.nn.functional.softplus(-y*weight*first*second).mean()
    loss.backward()
    torch.testing.assert_close(weight.grad,torch.tensor(-r1*r2/2,dtype=torch.float64),rtol=1e-14,atol=1e-14)
