from tests.test_statistic_race_memory import _pair, _run  # noqa: F401
import numpy as np
import torch



def test_sampled_race_writes_follow_pi_reproduce_and_nest_delivery():
    from sleeping_machines.statistic_race_sampled import SampledStatisticRaceNativeModel
    _, race, tokens = _pair()
    torch.manual_seed(0)
    sampled = SampledStatisticRaceNativeModel(payload=4, depth=2, pool=2, heads=2, orders=2, addresses=8,
                                              key_dim=4).double()
    sampled.load_state_dict(race.state_dict()); sampled.streams, sampled.active = race.streams, race.active
    pi = torch.tensor([.7, .2, .1, 0, 0, 0, 0, 0], dtype=torch.float64)
    wins = np.bincount([sampled.race_winner(pi, e) for e in range(4000)], minlength=8)
    assert abs(wins[0] / 4000 - .7) < .03 and abs(wins[1] / 4000 - .2) < .03 and wins[3:].sum() == 0
    assert sampled.race_winner(pi, 17) == sampled.race_winner(pi, 17)
    first = _run(sampled, tokens)[0]
    assert torch.allclose(first[:1], _run(race, tokens)[0][:1], atol=1e-12)  # identical before any write
    assert torch.allclose(first, _run(sampled, tokens, (16, 16, 8))[0], atol=1e-10)
    changed = tokens.clone(); changed[30:] = (changed[30:] + 1) % 5
    assert torch.allclose(first[:30], _run(sampled, changed)[0][:30], atol=1e-12)
