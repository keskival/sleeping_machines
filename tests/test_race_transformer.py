import sys

import torch

sys.path.insert(0, 'experiments')
from e64_lm_baselines import TfLM  # noqa: E402
from sleeping_machines.race_transformer import RaceTransformer  # noqa: E402


def _net():
    torch.manual_seed(0)
    net = TfLM(32, 2, 64).double().eval()
    return net


def test_expected_delivery_reproduces_the_transformer_exactly():
    net = _net()
    x = torch.randint(0, 27, (1, 40), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref, _ = net(x)
        race = RaceTransformer(net).double()
        out, state = race.forward_stream(x[0])
    torch.testing.assert_close(out, ref[0], rtol=0, atol=1e-10)
    assert state.value_reads == state.key_scores  # expected delivery reads every value


def test_sampled_races_are_unbiased_for_attention():
    net = _net()
    x = torch.randint(0, 27, (12,), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        exact, _ = RaceTransformer(net).double().forward_stream(x)
        g = torch.Generator().manual_seed(3)
        sampled = RaceTransformer(net, mode='sampled', samples=2000, generator=g).double()
        approx, _ = sampled.forward_stream(x)
        few = RaceTransformer(net, mode='sampled', samples=2, generator=g).double()
        _, state = few.forward_stream(torch.randint(0, 27, (60,), generator=torch.Generator().manual_seed(7)))
    assert (approx - exact).abs().max() < .05 * exact.abs().max()
    assert state.value_reads < state.key_scores / 10   # two races read at most two values per head and layer


def test_shortlist_equals_full_when_it_covers_all_keys_and_reads_fewer_otherwise():
    net = _net()
    x = torch.randint(0, 27, (30,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        exact, _ = RaceTransformer(net).double().forward_stream(x)
        full, _ = RaceTransformer(net, mode='shortlist', shortlist=64).double().forward_stream(x)
        short, state = RaceTransformer(net, mode='shortlist', shortlist=4).double().forward_stream(x)
    torch.testing.assert_close(full, exact, rtol=0, atol=1e-10)
    assert state.value_reads < state.key_scores


def test_temporal_decay_nests_at_zero_and_uses_elapsed_physical_time():
    net = _net()
    x = torch.randint(0, 27, (20,), generator=torch.Generator().manual_seed(5))
    times = torch.cumsum(torch.rand(20, generator=torch.Generator().manual_seed(6)) * 3, 0)
    race = RaceTransformer(net).double()
    with torch.no_grad():
        a, _ = race.forward_stream(x)                   # index times
        b, _ = race.forward_stream(x, times.tolist())   # physical times, lambda = 0: identical
        race.temporal_decay.fill_(.5)
        c, _ = race.forward_stream(x, times.tolist())
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(b, c)
    race.temporal_decay.data.fill_(.1)
    out, _ = race.forward_stream(x, times.tolist())
    torch.nn.functional.cross_entropy(out[:-1], x[1:]).backward()
    assert race.temporal_decay.grad.abs().sum() > 0


def test_eval_driver_on_a_saved_random_checkpoint(tmp_path):
    import json
    import subprocess
    net = TfLM(16, 1, 32).eval()
    ck = tmp_path / 'tf.pt'
    torch.save({'args': dict(model='transformer', size=16, layers=1, ctx=32, D=1000), 'state': net.state_dict()}, ck)
    tag = 'pytest_race_transformer_eval_tmp'
    out = __import__('pathlib').Path('experiments/results/race_transformer') / f'{tag}.json'
    out.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, 'experiments/race_transformer_eval.py', '--tag', tag, '--checkpoint', str(ck),
                        '--chars', '200'], check=True, capture_output=True)
        r = json.loads(out.read_text())
        assert r['modes']['expected']['max_logprob_diff'] < 1e-4
        assert abs(r['modes']['expected']['bpc'] - r['reference_bpc']) < 1e-4
        assert r['modes']['sampled_S1']['value_reads_per_token'] < r['modes']['expected']['value_reads_per_token']
    finally:
        out.unlink(missing_ok=True)
