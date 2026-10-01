"""Read-only credit and frozen-intervention contracts; no optimizer steps."""
import copy
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from diagnose_parallel_head_language import short_intervention,route_credit
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel


def test_detaching_old_cache_removes_old_projection_credit_but_preserves_forward():
    with torch.random.fork_rng():
        torch.manual_seed(151)
        model=ParallelHeadRaceLanguageModel(8,2,2,heads=2)
        tokens=torch.tensor([1,2,1,3,1,2,4,1])
        # Recreate a live graph with exactly the same weights and race draws.
        torch.manual_seed(167);_,live=model.forward_chunk(tokens[:-1])
        rng=torch.get_rng_state();detached=copy.deepcopy(live.detach())
        # The live state was just detached; rebuild it from the same warm RNG.
        torch.manual_seed(167);_,live=model.forward_chunk(tokens[:-1])
        torch.set_rng_state(rng);a=model.consume(tokens[-1],live)
        a.square().sum().backward()
        live_grad=[p.weight.grad.detach().clone() if p.weight.grad is not None else None for p in model.kv_value]
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng);b=model.consume(tokens[-1],detached)
        b.square().sum().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert all(p.weight.grad is None for p in model.kv_value)
        assert all(p.weight.grad is None for p in model.kv_key)
        assert any(g is not None and bool(g.abs().sum()>0) for g in live_grad)


def test_frozen_interventions_and_route_credit_preserve_parameters_and_restore_hooks():
    with torch.random.fork_rng():
        torch.manual_seed(173)
        model=ParallelHeadRaceLanguageModel(8,2,2,heads=2)
        before=copy.deepcopy(model.state_dict())
        tokens=torch.tensor(([1,2,1,3,4,1,2,5]*10)[:73])
        original=model.race
        for mode in ('unaltered','source_off','kv_off','channel_identity'):
            assert short_intervention(model,tokens[:17],mode)['bpc']>0
        result=route_credit(model,tokens,1)
        assert len(result['rows'])==2
        assert model.race==original
        assert all(abs(r['teacher_sum'])<1e-10 for r in result['rows'])
        for name,p in model.state_dict().items():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        assert not model.source_gate._forward_hooks
        assert all(not m._forward_hooks for m in model.channel_mix)
