"""Frozen dense control + count receivers (THEORY §386 prediction 1).

Loads a saved E64 checkpoint ({"args", "state"}), reproduces its test-scoring positions and base
log-probabilities exactly (LSTM: stateful 4096 blocks; Transformer: windows of T with T/2 context), and
composes them with addressed context-suffix count receivers of orders 1..K built from the same D fitting
characters (§378 escape-race cascade, base measure at the bottom). Only the 2K escape parameters (discount
D_k, concentration theta_k) are fitted, by L-BFGS on validation text8[90M:90M+V]; network weights stay frozen.
Counts are frozen fit counts (no test adaptation), so the gain measures memorized statistics the network
did not store, not in-context accumulation. Reports base-only, counts-only (uniform base, own fitted escape)
and composed test bpc on identical positions, with responsibility statistics.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import count_reference_scale as S  # noqa: E402
from sleeping_machines.count_carrying_language import compose  # noqa: E402

A = 27


def build_net(args):
    import e64_lm_baselines as E
    if args['model'] == 'lstm':
        return E.LSTMLM(args['size'], 0.0)
    return E.TfLM(args['size'], args['layers'], args['ctx'], 0.0)


def base_logprobs(net, model, seg, ctx, batch=16):
    """target indices t (predicting seg[t]) and log-probs, exactly the positions E64's score() counts."""
    seg = torch.as_tensor(seg.astype(np.int64))
    idx, out = [], []
    net.eval()
    with torch.no_grad():
        if model == 'lstm':
            state = None
            for s0 in range(0, len(seg) - 1, 4096):
                yb = seg[s0 + 1:s0 + 4097]
                xb = seg[s0:s0 + len(yb)][None]
                logits, state = net(xb, state)
                out.append(F.log_softmax(logits[0], -1))
                idx.append(torch.arange(s0 + 1, s0 + 1 + len(yb)))
        else:
            T, half = ctx, ctx // 2
            starts = list(range(0, len(seg) - T - 1, half))
            for o in range(0, len(starts), batch):
                ss = starts[o:o + batch]
                xb = torch.stack([seg[s:s + T] for s in ss])
                lp = F.log_softmax(net(xb)[0], -1)
                for j, s in enumerate(ss):
                    keep = slice(0, T) if (o == 0 and j == 0) else slice(half, T)
                    out.append(lp[j, keep])
                    idx.append(torch.arange(s + 1, s + T + 1)[keep])
    return torch.cat(idx).numpy(), torch.cat(out).float()


def count_vectors(tables, seg, targets, K):
    """(K, len(targets), A) raw counts of the order-k context preceding each target, within the segment."""
    out = np.zeros((K, len(targets), A), np.float32)
    for k in range(1, K + 1):
        codes = S.ctx_codes(seg, k, 0, len(seg))[targets]
        for s in range(0, len(targets), 65536):
            C, _, _ = tables[k].lookup(codes[s:s + 65536])
            out[k - 1, s:s + 65536] = C
    return torch.from_numpy(out)


def nll(log_q, counts, y, raw_d, raw_t):
    lp = compose(log_q, counts, torch.sigmoid(raw_d), F.softplus(raw_t))
    return -lp.gather(1, y[:, None]).sum(), lp


def fit_escape(log_q, counts, y, K):
    raw_d = torch.full((K,), math.log(3.), requires_grad=True)
    raw_t = torch.full((K,), math.log(math.expm1(1.)), requires_grad=True)
    opt = torch.optim.LBFGS([raw_d, raw_t], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = nll(log_q, counts, y, raw_d, raw_t)[0] / len(y)
        loss.backward()
        return loss
    for _ in range(3):
        opt.step(closure)
    return raw_d.detach(), raw_t.detach()


def evaluate(log_q, counts, y, raw_d, raw_t, block=65536):
    tot, rs = 0.0, []
    for s in range(0, len(y), block):
        sl = slice(s, s + block)
        with torch.no_grad():
            loss, lp = nll(log_q[sl], counts[:, sl], y[sl], raw_d, raw_t)
            # responsibility of the base at the bottom of the cascade: d log p / d log q_y
        lq = log_q[sl].clone().requires_grad_(True)
        lpy = compose(lq, counts[:, sl], torch.sigmoid(raw_d), F.softplus(raw_t)).gather(1, y[sl][:, None]).sum()
        g, = torch.autograd.grad(lpy, lq)
        rs.append(g.gather(1, y[sl][:, None])[:, 0].numpy())  # d log p(y)/d log q(y) = r_y
        tot += float(loss)
    r = np.concatenate(rs)
    return dict(bpc=tot / len(y) / math.log(2), mean_responsibility=float(r.mean()),
                frac_resp_lt_05=float((r < .05).mean()), frac_resp_lt_20=float((r < .2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--K', type=int, default=7)
    ap.add_argument('--valid', type=int, default=200_000)
    ap.add_argument('--test', type=int, default=1_000_000)
    ap.add_argument('--fit', type=int, default=None, help='override D (contract tests only)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    torch.set_num_threads(1)
    t0 = time.time()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    args = saved['args']
    D = a.fit or args['D']
    net = build_net(args)
    net.load_state_dict(saved['state'])
    x = S.load()
    valid, test = x[90_000_000:90_000_000 + a.valid], x[95_000_000:95_000_000 + a.test]
    raw, _ = S.build(x[:D], a.K, False)
    rows = {}
    seg_data = {}
    for name, seg in (('valid', valid), ('test', test)):
        t, lq = base_logprobs(net, args['model'], seg, args.get('ctx', 256))
        seg_data[name] = (lq, count_vectors(raw, seg, t, a.K), torch.as_tensor(seg[t].astype(np.int64)))
    lq_v, c_v, y_v = seg_data['valid']
    lq_t, c_t, y_t = seg_data['test']
    uniform_v = torch.full_like(lq_v, -math.log(A))
    uniform_t = torch.full_like(lq_t, -math.log(A))
    rows['base_only_test_bpc'] = float(-lq_t.gather(1, y_t[:, None]).sum()) / len(y_t) / math.log(2)
    d_c, t_c = fit_escape(uniform_v, c_v, y_v, a.K)
    rows['counts_only'] = dict(test=evaluate(uniform_t, c_t, y_t, d_c, t_c),
                               discount=torch.sigmoid(d_c).tolist(), concentration=F.softplus(t_c).tolist())
    d_m, t_m = fit_escape(lq_v, c_v, y_v, a.K)
    rows['composed'] = dict(test=evaluate(lq_t, c_t, y_t, d_m, t_m), valid=evaluate(lq_v, c_v, y_v, d_m, t_m),
                            discount=torch.sigmoid(d_m).tolist(), concentration=F.softplus(t_m).tolist())
    rows['scored_test_targets'] = int(len(y_t))
    result = dict(status='completed', kind='count_composition_frozen', checkpoint=a.checkpoint,
                  checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  checkpoint_args=args, fit_characters=D, orders=a.K, valid=[90_000_000, 90_000_000 + a.valid],
                  test=[95_000_000, 95_000_000 + a.test], results=rows,
                  scope=('Frozen network; 2K escape parameters fitted on validation; frozen fit counts, no test '
                         'adaptation; identical scored positions for all three rows. THEORY §386 prediction 1.'),
                  wall_seconds=time.time() - t0,
                  hardware=dict(device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
