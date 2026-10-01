"""Independent feature-ID rows admitted to a native content/time receiver stack."""
import torch
from torch import nn
from sleeping_machines.clock_feature_event_heads import ClockFeatureEventHeads


class NativeTabularModel(nn.Module):
    def __init__(self,features,classes,payload=8,depth=8,heads=2,pool=2,clock_features=0,clock_allocation='uniform'):
        super().__init__();self.features=features
        self.core=ClockFeatureEventHeads(sources=1,content_dim=features+3,classes=classes,
            payload=payload,depth=depth,heads=heads,pool=pool,clock_features=clock_features,
            clock_allocation=clock_allocation)

    def forward(self,values,order=None,missing=None):
        if values.shape!=(self.features,):raise ValueError('One independent fixed-width row required')
        order=list(range(self.features)) if order is None else list(order)
        if sorted(order)!=list(range(self.features)):raise ValueError('Each feature ID observed exactly once')
        missing=torch.zeros(self.features,dtype=torch.bool,device=values.device) if missing is None else missing
        if missing.shape!=values.shape:raise ValueError('Missingness shape differs')
        state=self.core.new_state()
        # Static data has no physical arrival order. Canonicalize feature IDs;
        # presentation permutations are exactly equivalent, not augmented sequences.
        for slot,i in enumerate(sorted(order)):
            identity=values.new_zeros(self.features);identity[i]=1
            mark=torch.cat((identity,values[i:i+1]*(~missing[i]).to(values.dtype),
                            missing[i:i+1].to(values.dtype),values.new_zeros(1)))
            self.core.consume_event(0,float(slot),mark,state)
        query=values.new_zeros(self.features+3);query[-1]=1
        z,_=self.core.consume_event(0,float(self.features),query,state)
        return z,state


def contracts(features=4,classes=2,clock_features=2):
    with torch.random.fork_rng():
        torch.manual_seed(917);m=NativeTabularModel(features,classes,clock_features=clock_features)
        x=torch.linspace(-.7,.9,features);rng=torch.get_rng_state()
        def predict(training,values,order=None,missing=None):
            torch.set_rng_state(rng);m.train(training)
            return m(values,order,missing)
        with torch.no_grad():
            a,s=predict(False,x);b,_=predict(False,x,list(reversed(range(features))))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            c,_=predict(True,x);torch.testing.assert_close(a,c,rtol=0,atol=0)
            predict(False,x+10);d,_=predict(False,x);torch.testing.assert_close(a,d,rtol=0,atol=0)
            zero=x.clone();zero[0]=0;mask=torch.zeros(features,dtype=torch.bool);mask[0]=True
            e,_=predict(False,zero);f,_=predict(False,zero,missing=mask)
            assert not torch.equal(e,f),'Observed zero and missing feature must differ'
        z,_=predict(True,x);z.square().sum().backward()
        for block in m.core.queries:
            for query in block:assert query.weight.grad is not None and torch.isfinite(query.weight.grad).all()
        assert s.events==features+1 and s.selected_updates==(features+1)*8*2
        return dict(feature_presentation_invariant=True,independent_row_reset=True,
            teacher_equals_inference=True,missing_not_zero=True,all_depths_have_finite_query_credit=True,
            target_not_an_input=True,features_read_once=True)
