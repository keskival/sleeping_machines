"""Local-expectation route credit with a learned counterfactual critic as control variate (THEORY §403, note 92).

Every race r gets cheap credit from a small critic Q_phi(r, i) predicting the replay loss of alternative i (as an advantage
over the race's mean prediction); k sampled races per episode get the exact first-time-preserving replay correction:

    surrogate = sum_{all r} sum_i pi_{r,i} sg(Q(r,i)) + (R/k) sum_{sampled r} sum_i pi_{r,i} sg(L_r(i) - Q(r,i)).

Unbiased for any critic that is fixed while the window's races are sampled: the critic is cross-fitted in time.
Its parameters change only after the window's model update, by regression on that window's replay targets.
Critic features are detached local race quantities: scores, pi, depth one-hot, event position, |value| and
|value - mean value| per alternative.  Critic evaluation, training and replays are charged (traced windows).
"""
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
import dvs_fork_replay as FR  # noqa: E402
from parallel_head_accumulated_language import merge  # noqa: E402
from race_language_screen import capture  # noqa: E402

CRITIC = {}


def parser():
    p = LE.parser(); p.add_argument('--critic-width', type=int, default=32); p.add_argument('--critic-lr', type=float, default=.003)
    p.add_argument('--fork', action='store_true', help='resume replays at the forced race event (§404; exact, cheaper)')
    p.add_argument('--no-critic', action='store_true', help='plain replay credit (critic output fixed at zero)')
    return p


def features(scores, values, race, races, depth, layers):
    pi = torch.softmax(scores.detach(), 0)
    v = values.detach().norm(dim=-1); dv = (values.detach() - values.detach().mean(0)).norm(dim=-1)
    pos = torch.full_like(pi, race / max(1, races - 1))
    d = torch.zeros(len(pi), layers, dtype=pi.dtype); d[:, depth] = 1
    return torch.cat([scores.detach()[:, None], pi[:, None], v[:, None], dv[:, None], pos[:, None], d], 1).float()


def critic_for(model, a):
    key = id(model)
    if key not in CRITIC:
        torch.manual_seed(a.seed + 99)
        net = nn.Sequential(nn.Linear(5 + model.depth, a.critic_width), nn.GELU(), nn.Linear(a.critic_width, 1))
        CRITIC[key] = (net, torch.optim.Adam(net.parameters(), lr=a.critic_lr), [])
    return CRITIC[key]


def run_with_values(model, row, seed, fork=False):
    """realized episode recording scores and candidate values per race (and event snapshots when forking)."""
    record, vals = [], []
    orig = LE.PATHWISE

    def spy(scores, values):
        vals.append(values); return orig(scores, values)
    LE.PATHWISE = spy
    try:
        if fork:
            loss, races, state, snaps = FR.realized(model, row, seed, record)
        else:
            (loss, races, state), snaps = LE.run(model, row, seed, record=record), None
    finally:
        LE.PATHWISE = orig
    return loss, races, state, record, vals, snaps


def train_window(model, optimizer, rows, a, epoch, trace=False):
    net, copt, buffer = critic_for(model, a)
    optimizer.zero_grad(set_to_none=True); stages = {}; loss_sum = 0.; events = keys = commits = teachers = replays = 0
    def traced(name, fn):
        if trace: stages.setdefault(name, []).append(capture(fn))
        else: fn()
    window_targets = []
    for row in rows:
        seed = 100000 + a.seed + 10000 * epoch
        rng = np.random.default_rng(seed * 7919 + row['index'])
        box = {}
        def forward():
            model.train()
            loss, races, state, scores, vals, snaps = run_with_values(model, row, seed, getattr(a, 'fork', False))
            feats = [features(scores[r], vals[r], r, races, (r // model.heads) % model.depth, model.depth) for r in range(races)]
            with torch.no_grad():
                q = [net(f)[:, 0].double() for f in feats]
                q = [x - x.mean() for x in q]                                   # advantage form
                if getattr(a, 'no_critic', False):
                    q = [torch.zeros_like(x) for x in q]
            route = loss.new_zeros(())
            for r in range(races):                                              # critic credit for every race
                route = route + (torch.softmax(scores[r], 0) * q[r].to(scores[r].dtype)).sum()
            chosen = rng.choice(races, size=min(a.route_samples, races), replace=False)
            with torch.no_grad():
                if snaps is not None:
                    losses = {int(r): torch.tensor([float(FR.forked(model, row, snaps, (int(r), i))[0])
                                                    for i in range(len(scores[int(r)]))], dtype=torch.float64) for r in chosen}
                else:
                    losses = {int(r): torch.tensor([float(LE.run(model, row, seed, force=(int(r), i))[0])
                                                    for i in range(len(scores[int(r)]))], dtype=torch.float64) for r in chosen}
            for r, L in losses.items():                                         # exact replay correction
                adv = L - L.mean()
                route = route + (races / len(chosen)) * (torch.softmax(scores[r], 0) * (adv - q[r]).to(scores[r].dtype)).sum()
                window_targets.append((feats[r], adv.float()))
            box.update(loss=loss, objective=loss + route, state=state, replays=sum(len(L) for L in losses.values()))
        traced('forward_and_loss', forward)
        if not torch.isfinite(box['loss']): raise FloatingPointError('Nonfinite query loss')
        traced('backward', lambda: box['objective'].backward()); loss_sum += float(box['loss'].detach())
        state = box['state']; events += state.events; keys += state.candidate_scores
        commits += state.selected_updates; teachers += state.counterfactual_values; replays += box['replays']
        LE.REPLAYS[0] += box['replays']
    def normalize():
        for parameter in model.parameters():
            if parameter.grad is not None: parameter.grad.div_(len(rows))
    traced('gradient_normalization', normalize)
    traced('gradient_clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    traced('optimizer', optimizer.step)
    def fit_critic():                                                           # cross-fitted: after the model update
        if window_targets:
            x = torch.cat([f for f, _ in window_targets]); y = torch.cat([t for _, t in window_targets])
            copt.zero_grad(); F.mse_loss(net(x)[:, 0], y).backward(); copt.step()
    traced('critic_update', fit_critic)
    return dict(targets=len(rows), loss_sum=loss_sum, events=events, key_scores=keys, selected_updates=commits,
                counterfactual_values=teachers, route_replays=replays,
                stages={name: merge(values) for name, values in stages.items()})


def sources():
    return {**LE.sources(), 'experiments/dvs_critic_le_benchmark.py': N.sha(ROOT / 'experiments/dvs_critic_le_benchmark.py'),
            'experiments/dvs_fork_replay.py': N.sha(ROOT / 'experiments/dvs_fork_replay.py')}


@contextmanager
def activate():
    old = N.make_model, N.parser, N.sources, N.train_window
    N.make_model, N.parser, N.sources, N.train_window = LE.make_model, parser, sources, train_window
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources, N.train_window = old


if __name__ == '__main__':
    args = parser().parse_args()
    with activate():
        N.run(args)
