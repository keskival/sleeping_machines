import sys

import torch

sys.path.insert(0, 'experiments')
import joint_event_dense_controls as D  # noqa: E402
import joint_event_language_tasks as J  # noqa: E402


def test_controls_are_causal_and_never_see_the_label():
    row = J.episodes(4, 7)[0]
    for model in (D.TimeGRU(16), D.TimeTransformer(16)):
        model.eval()
        x, t, q, y = D.tensors(row)
        with torch.no_grad():
            full = model(x, t)
            prefix = model(x[:5], t[:5])
            flipped = row[:-1] + [J.Event(0, row[-1].time, row[-1].mark, 1 - row[-1].target)]
            same, _ = D.predict(model, flipped)
        torch.testing.assert_close(full[:5], prefix, rtol=0, atol=1e-5)   # later events never change earlier outputs
        torch.testing.assert_close(same, full[q], rtol=0, atol=0)         # target is not an input


def test_controls_use_elapsed_time():
    row = J.episodes(4, 8)[0]
    for model in (D.TimeGRU(16), D.TimeTransformer(16)):
        model.eval()
        x, t, q, _ = D.tensors(row)
        with torch.no_grad():
            a = model(x, t)[q]; b = model(x, t * 1.7)[q]
        assert not torch.allclose(a, b)


def test_controls_receive_gradients_everywhere():
    row = J.episodes(4, 9)[0]
    for model in (D.TimeGRU(16), D.TimeTransformer(16)):
        z, y = D.predict(model, row)
        torch.nn.functional.cross_entropy(z, y).backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
