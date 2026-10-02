from dataclasses import replace
import torch

from experiments.paired_timing_tasks import paired_timing_episodes
from sleeping_machines.decay_timing_reference import DecayTimingReference


def predictions(row):
    model=DecayTimingReference();out=[]
    for event in row:
        margin=model.consume(event.source,event.time,event.mark)
        if margin is not None:out.append(margin)
    return torch.stack(out)


def test_causal_traces_reproduce_generator_labels():
    for row in paired_timing_episodes(4,64,419):
        labels=torch.tensor([e.target for e in row if e.target is not None])
        assert torch.equal((predictions(row)>0).long(),labels)


def test_labels_never_enter_prediction_and_time_shift_preserves_margins():
    row=paired_timing_episodes(4,8,419)[0]
    modified=[replace(e,target=None if e.target is None else 1-e.target) for e in row]
    torch.testing.assert_close(predictions(row),predictions(modified),atol=0,rtol=0)
    shifted=[replace(e,time=e.time+100.) for e in row]
    torch.testing.assert_close(predictions(row),predictions(shifted),atol=1e-12,rtol=1e-12)


def test_interleaving_other_sources_does_not_change_source_zero_state():
    row=paired_timing_episodes(4,8,419)[0]
    whole=DecayTimingReference();solo=DecayTimingReference()
    for e in row:
        margin=whole.consume(e.source,e.time,e.mark)
        if e.source==0:
            isolated=solo.consume(e.source,e.time,e.mark)
            if margin is not None:torch.testing.assert_close(margin,isolated,atol=0,rtol=0)


def test_reference_arithmetic_has_complete_operator_coverage():
    from sleeping_machines.operation_audit import OperationAudit
    audit=OperationAudit()
    with audit:predictions(paired_timing_episodes(4,8,419)[0])
    work=audit.result()
    assert work['formula_coverage_complete'],work['unsupported_floating_operators']
    assert work['arithmetic_flops']>0 and work['special_function_evaluations']>0
