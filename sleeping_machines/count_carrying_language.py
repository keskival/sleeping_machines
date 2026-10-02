"""Count-carrying receivers over the native addressed temporal core (THEORY §§376-380).

The native stream model is unchanged: its races, addressed receivers, persistent
memory and counterfactual route credit produce a learned predictive q (the base
measure).  Addressed context-suffix receivers of orders 1..K carry sufficient
statistics (next-symbol counts).  Prediction is the escape-race cascade of §378
with Pitman-Yor style learnable discount D_k and concentration theta_k:

    p_0 = q,   p_k(y) = (max(c_ky - D_k, 0) + (theta_k + D_k T_k) p_{k-1}(y)) / (n_k + theta_k)

for every order whose context has been observed (n_k > 0); unseen contexts escape
with probability one.  Count vectors are supplied per stream position and are
causal by construction: leave-one-out counts on the fitting stream (the event's
own transition removed) and prequential counts on an evaluation stream (fitting
counts plus the evaluation prefix strictly before the target).  Counts are not
parameters; only D_k, theta_k and the native weights learn.  forward_chunk returns
normalized log-probabilities, so cross-entropy on its output is exactly -log p(y).
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native_stream_language import NativeStreamLanguageModel

A = 27


def context_codes(tokens, k):
    """code of tokens[p-k+1..p] for every position p (positions p < k-1 get -1)."""
    x = np.asarray(tokens, np.int64)
    code = np.full(len(x), -1, np.int64)
    if k == 0:
        return np.zeros(len(x), np.int64)
    acc = np.zeros(len(x) - k + 1, np.int64)
    for j in range(k):
        acc = acc * A + x[j:len(x) - k + 1 + j]
    code[k - 1:] = acc
    return code


def fit_tables(fit, K):
    """dict per order: context code -> next-symbol count vector, over every transition of the fit stream."""
    x = np.asarray(fit, np.int64)
    tables = []
    for k in range(1, K + 1):
        codes = context_codes(x, k)[:-1]
        nxt = x[1:]
        tab = {}
        for c, y in zip(codes.tolist(), nxt.tolist()):
            if c >= 0:
                tab.setdefault(c, np.zeros(A, np.float64))[y] += 1
        tables.append(tab)
    return tables


def fit_stream_counts(fit, K):
    """(K, len(fit), A) leave-one-out counts: position p excludes its own transition p -> p+1."""
    x = np.asarray(fit, np.int64)
    tables = fit_tables(x, K)
    out = np.zeros((K, len(x), A), np.float32)
    for k in range(1, K + 1):
        codes = context_codes(x, k)
        for p in range(len(x) - 1):
            if codes[p] >= 0:
                v = tables[k - 1][codes[p]].copy()
                v[x[p + 1]] -= 1
                out[k - 1, p] = v
    return out


def eval_stream_counts(fit, stream, K):
    """(K, len(stream), A) prequential counts: fit transitions plus stream transitions strictly before p."""
    tables = [{c: v.copy() for c, v in t.items()} for t in fit_tables(fit, K)]
    x = np.asarray(stream, np.int64)
    out = np.zeros((K, len(x), A), np.float32)
    codes = [context_codes(x, k) for k in range(1, K + 1)]
    for p in range(len(x) - 1):
        for k in range(K):
            c = codes[k][p]
            if c >= 0 and c in tables[k]:
                out[k, p] = tables[k][c]
        for k in range(K):  # observe the transition only after position p has been scored
            c = codes[k][p]
            if c >= 0:
                tables[k].setdefault(c, np.zeros(A, np.float64))[x[p + 1]] += 1
    return out


def compose(log_q, counts, discount, theta):
    """escape-race cascade; log_q (L, A) base log-probs, counts (K, L, A); returns log p (L, A)."""
    p = log_q.exp()
    for k in range(counts.shape[0]):
        c = counts[k]
        n = c.sum(-1, keepdim=True)
        T = (c > 0).to(p.dtype).sum(-1, keepdim=True)
        D, th = discount[k], theta[k]
        mixed = (torch.clamp(c - D, min=0.) + (th + D * T) * p) / (n + th)
        p = torch.where(n > 0, mixed, p)
    return torch.log(p.clamp_min(1e-30))


class CountCarryingNativeModel(NativeStreamLanguageModel):
    def __init__(self, payload=16, depth=8, pool=2, heads=2, orders=4, vocabulary=27):
        super().__init__(payload, depth, pool, heads, vocabulary)
        if vocabulary != A or orders < 1:
            raise ValueError('27-symbol text and at least one count order required')
        self.orders = orders
        self.raw_discount = nn.Parameter(torch.full((orders,), float(np.log(.75 / .25))))  # sigmoid -> .75
        self.raw_theta = nn.Parameter(torch.full((orders,), float(np.log(np.expm1(1.0)))))  # softplus -> 1
        self.streams = {}
        self.active = None

    def register_stream(self, name, counts):
        counts = torch.as_tensor(counts, dtype=self.embedding.weight.dtype)
        if counts.shape[0] != self.orders or counts.shape[2] != A:
            raise ValueError('counts must be (orders, positions, 27)')
        self.streams[name] = counts

    def use_stream(self, name):
        if name not in self.streams:
            raise KeyError(name)
        self.active = name

    def escape_parameters(self):
        return torch.sigmoid(self.raw_discount), F.softplus(self.raw_theta)

    def forward_chunk(self, tokens, state=None):
        state = self.new_state() if state is None else state
        start = state.events
        logits, state = super().forward_chunk(tokens, state)
        counts = self.streams[self.active][:, start:start + logits.shape[0]]
        if counts.shape[1] != logits.shape[0]:
            raise IndexError('stream position beyond registered counts')
        D, th = self.escape_parameters()
        return compose(F.log_softmax(logits, -1), counts, D, th), state
