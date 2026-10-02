"""Timestamp-aware dense calibration controls for the joint text + event task (THEORY §394).  AWS-only training.

Same episodes, same 32-wide event content and physical timestamps as the native benchmark; the label never
enters inputs.  Two conventional learners with full access to elapsed time:
  gru          continuous-time GRU: before each event the hidden state decays by exp(-dt / tau) with learned
               per-unit tau (GRU-D style), and log(1 + dt) is appended to the input;
  transformer  causal Transformer over the episode's events with sinusoidal encodings of the absolute
               timestamp and of the gap to the previous event; the query event's output is classified.
Calibration passes if a control beats the table bar by >= 20 points (§394).  Fitting work: the same representative-window
estimate and operator audit as the native joint driver (2 FLOPs/MAC, specials separate).
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import joint_event_language_tasks as J  # noqa: E402
from parallel_head_accumulated_language import merge  # noqa: E402
from race_language_screen import capture  # noqa: E402


def tensors(row):
    x = torch.tensor([e.mark for e in row], dtype=torch.float32)
    t = torch.tensor([e.time for e in row], dtype=torch.float32)
    q = [i for i, e in enumerate(row) if e.target is not None]
    return x, t, q, torch.tensor([row[i].target for i in q])


class TimeGRU(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.cell = nn.GRUCell(J.WIDTH + 1, width)
        self.log_tau = nn.Parameter(torch.linspace(-1, 3, width))
        self.out = nn.Linear(width, 2)

    def forward(self, x, t):
        h, prev, outs = x.new_zeros(self.cell.hidden_size), t[0], []
        for i in range(len(x)):
            dt = (t[i] - prev).clamp_min(0.); prev = t[i]
            h = h * torch.exp(-dt / self.log_tau.exp())
            h = self.cell(torch.cat([x[i], torch.log1p(dt)[None]])[None], h[None])[0]
            outs.append(h)
        return self.out(torch.stack(outs))


def time_encoding(v, width):
    freq = torch.exp(torch.linspace(0, -math.log(1000.), width // 2))
    angle = v[:, None] * freq[None]
    return torch.cat([angle.sin(), angle.cos()], -1)


class TimeTransformer(nn.Module):
    def __init__(self, width=64, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(J.WIDTH, width)
        self.time = nn.Linear(width, width, bias=False)
        block = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, 2)

    def forward(self, x, t):
        gap = torch.cat([t.new_zeros(1), (t[1:] - t[:-1]).clamp_min(0)])
        w = self.inp.out_features
        h = self.inp(x) + self.time(time_encoding(t, w) + time_encoding(gap * 10, w))
        mask = torch.triu(torch.full((len(x), len(x)), float('-inf')), 1)
        return self.out(self.encoder(h[None], mask=mask)[0])


def predict(model, row):
    x, t, q, y = tensors(row)
    return model(x, t)[q], y


@torch.no_grad()
def evaluate(model, rows):
    model.eval(); loss = correct = n = 0
    for row in rows:
        z, y = predict(model, row)
        loss += float(F.cross_entropy(z, y, reduction='sum')); correct += int((z.argmax(-1) == y).sum()); n += len(y)
    return dict(n=n, nll=loss / n, accuracy=correct / n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--model', choices=('gru', 'transformer'), required=True)
    p.add_argument('--width', type=int, default=64); p.add_argument('--fit', type=int, default=512)
    p.add_argument('--dev', type=int, default=256); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=6); p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--update-episodes', type=int, default=16)
    p.add_argument('--background', type=int, nargs=2, default=[2, 6], help='[low, high) phase-A events: history length')
    p.add_argument('--trace-windows', type=int, default=2, help='optimizer windows of pass 1 traced for the work estimate')
    a = p.parse_args()
    out = ROOT / 'experiments/results/joint_event_language' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); started = time.perf_counter()
    fit, dev = J.episodes(a.fit, 1301, tuple(a.background)), J.episodes(a.dev, 2301, tuple(a.background))
    model = TimeGRU(a.width) if a.model == 'gru' else TimeTransformer(a.width)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); rng = np.random.default_rng(a.seed + 10)
    ledger = {k: [] for k in ('forward_and_loss', 'backward', 'gradient_normalization', 'gradient_clipping', 'optimizer')}
    result = dict(status='running', args=vars(a), parameters=sum(q.numel() for q in model.parameters()), curve=[],
                  data_sha256=dict(fit=J.data_hash(fit), dev=J.data_hash(dev)),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/aws_joint_event_dense_controls.py', 'experiments/joint_event_dense_controls.py', 'experiments/joint_event_language_tasks.py', 'experiments/native_event_tasks.py', 'sleeping_machines/operation_audit.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, device='cpu', threads=1),
                  scope='Labelled dense timestamp-aware control (Theory §394 calibration); not the research architecture')
    best, best_state = float('inf'), None
    events = traced_events = 0
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(fit)).tolist(); model.train()
        for begin in range(0, len(order), a.update_episodes):
            opt.zero_grad(set_to_none=True); pending = 0
            traced = epoch == 1 and begin // a.update_episodes < a.trace_windows
            run = capture if traced else (lambda f: f())
            for i in order[begin:begin + a.update_episodes]:
                box = {}
                def forward():
                    z, y = predict(model, fit[i]); box.update(loss=F.cross_entropy(z, y, reduction='sum'), n=len(y))
                r1 = run(forward); r2 = run(lambda: box['loss'].backward()); pending += box['n']
                events += len(fit[i])
                if traced:
                    ledger['forward_and_loss'].append(r1); ledger['backward'].append(r2); traced_events += len(fit[i])
            def normalize():
                for q in model.parameters():
                    if q.grad is not None:
                        q.grad.div_(pending)
            for key, step in (('gradient_normalization', normalize),
                              ('gradient_clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)),
                              ('optimizer', opt.step)):
                r = run(step)
                if traced:
                    ledger[key].append(r)
            if traced:
                for k in ledger:
                    ledger[k] = [merge(ledger[k])]
        score = evaluate(model, dev); result['curve'].append(dict(epoch=epoch, dev=score))
        if score['nll'] < best:
            best, best_state = score['nll'], copy.deepcopy(model.state_dict()); result['selected_epoch'] = epoch
        print(json.dumps(dict(epoch=epoch, **score)), flush=True)
    model.load_state_dict(best_state)
    confirmation = J.episodes(1024, 3301, tuple(a.background))
    result['data_sha256']['confirmation'] = J.data_hash(confirmation)
    result['final'] = dict(dev=evaluate(model, dev), confirmation=evaluate(model, confirmation))
    checkpoint = out.with_suffix('.pt')
    torch.save(dict(model=model.state_dict(), args=vars(a), selected_epoch=result['selected_epoch']), checkpoint)
    result['checkpoint'] = dict(path=str(checkpoint.relative_to(ROOT)), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), purpose='Selected fixed predictor; not an optimizer-resume checkpoint')
    model.eval()
    with torch.no_grad():
        inference = merge([capture(lambda row=row: predict(model, row)) for row in dev[:16]])
    train = {k: merge(v) for k, v in ledger.items()}
    scale = events / traced_events
    arith = sum(t['arithmetic_flops'] for t in train.values()) * scale
    special = sum(t['special_function_evaluations'] for t in train.values()) * scale
    result['work'] = dict(total_training_unit_special_flops=arith + special, total_training_arithmetic_flops=arith,
                          fit_mflops_per_query=(arith + special) / (a.fit * a.epochs) / 1e6, traced_training_stages=train,
                          fitting_events=events, traced_events=traced_events,
                          estimate='first pass-1 windows fully traced; traced work per fitting event x fitting events',
                          inference_unit_special_flops_per_query=(inference['arithmetic_flops'] + inference['special_function_evaluations']) / min(16, len(dev)),
                          inference_ledger=inference, inference_queries=min(16, len(dev)))
    result.update(status='completed', wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
