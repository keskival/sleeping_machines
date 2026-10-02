from dataclasses import replace
import torch
from sleeping_machines.event_history_control import EventHistoryControl
from experiments.native_event_tasks import episodes


def test_history_prediction_is_target_independent_and_source_local():
    model=EventHistoryControl(8)
    row=episodes('order',4,4,719)[0]
    original,_,_=model.predict_population(row)
    changed=[replace(e,target=None if e.target is None else (e.target+1)%4) for e in row]
    other,_,_=model.predict_population(changed)
    torch.testing.assert_close(original,other,atol=0,rtol=0)
    solo,_,state=model.predict_population([e for e in row if e.source==0])
    query_sources=[e.source for e in row if e.mark[1]]
    torch.testing.assert_close(solo[0],original[query_sources.index(0)],atol=0,rtol=0)
    assert len(state[0])==3


def test_time_translation_preserves_elapsed_input():
    model=EventHistoryControl(8).double();row=episodes('order',4,4,719)[0]
    shifted=[replace(e,time=e.time+100.) for e in row]
    torch.testing.assert_close(model.predict_population(row)[0],model.predict_population(shifted)[0],atol=1e-12,rtol=1e-12)
