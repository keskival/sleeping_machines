"""Shared event backbone: addressed temporal state, hard races and local loser credit.

A query consumes only an explicitly supplied observed prefix. This module has
no knowledge of speech, text, markets or synthetic labels. Readout semantics
and event units are declared by configuration; objective functions live in
objectives.py. E118/E119 are the zero-expert, pooled-readout special case.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .event_memory import segmented_memory

class RaceLayer(nn.Module):
    def __init__(self, dim, depth, beta, cf_credit, options=3, memory_backend="doubling"):
        super().__init__()
        self.alpha, self.cf_credit = beta/depth, cf_credit
        self.options, self.dim = options, dim
        if memory_backend == "linear":
            from .event_memory import linear_memory as memory
            self.memory = memory
        elif memory_backend == "doubling":
            self.memory = segmented_memory
        else:
            raise ValueError(memory_backend)
        self.value = nn.Parameter(torch.randn(options, dim, 2*dim+1)*.03)
        self.bias = nn.Parameter(torch.zeros(options, dim))
        self.route = nn.Parameter(torch.randn(options, 2*dim+1)*.1)
        self.route_bias = nn.Parameter(torch.zeros(options))
        self.log_tau = nn.Parameter(torch.logspace(math.log10(.02), math.log10(.8), options).log())

    def forward(self, x, t, count, keys, sequential=False, override=None, trace=False):
        # Learned delays can reorder carriers. Memory sees actual arrival
        # order within each receiver, with the original index breaking ties.
        time_order = torch.argsort(t, stable=True)
        inverse = torch.argsort(time_order)
        mem, mass, work = self.memory(
            x[time_order], t[time_order], count[time_order], keys[time_order],
            self.log_tau.clamp(math.log(.002), math.log(4.)).exp(), sequential)
        mem, mass = mem[inverse], mass[inverse]
        features = torch.cat((x[:, None, :].expand(-1, self.options, -1), mem,
                              (mass/(1+mass))[:, :, None]), -1)
        scores = (features*self.route[None, :, :]).sum(-1) + self.route_bias
        delay = .001 + .010*torch.sigmoid(-scores)
        winner = delay.argmin(-1)
        if override is not None:
            event, choice = override
            winner = winner.clone()
            winner[event] = choice
        w = self.value / self.value.abs().sum(-1, keepdim=True).clamp_min(1)
        rows = torch.arange(len(x))
        if self.training and self.cf_credit or trace:
            alternatives = torch.tanh(torch.einsum("ekf,kdf->ekd", features, w) + self.bias)
            correction = alternatives[rows, winner]
            value_evaluations = len(x)*self.options
        else:
            # Inference and pathwise control evaluate only selected values.
            correction = torch.zeros_like(x)
            for k in range(self.options):
                selected = torch.nonzero(winner == k, as_tuple=True)[0]
                correction = correction.index_copy(0, selected,
                    torch.tanh(F.linear(features[selected, k], w[k], self.bias[k])))
            alternatives = None
            value_evaluations = len(x)
        winning_delay = delay[rows, winner]
        out = x + self.alpha*correction
        tout = t + winning_delay
        if self.training and self.cf_credit:
            p = torch.softmax(-delay/.002, -1)
            zero_forward = p-p.detach()
            # Loser values have no ordinary value-path gradient. They supply
            # only a score direction comparing alternative and winner.
            out = out + self.alpha * (zero_forward[:, :, None] *
                (alternatives-correction[:, None, :]).detach()).sum(1)
            tout = tout + (zero_forward * (delay-winning_delay[:, None]).detach()).sum(1)
        sorted_delay = delay.detach().sort(-1).values
        stats = {"winner_counts": torch.bincount(winner, minlength=self.options).tolist(),
                 "mean_margin_ms": float((sorted_delay[:, 1]-sorted_delay[:, 0]).mean()*1000),
                 "mean_delay_ms": float(winning_delay.detach().mean()*1000),
                 "scan_compositions": work, "value_evaluations": value_evaluations}
        details = None
        if trace:
            details = {"winner": winner, "alternatives": alternatives,
                       "delays": delay, "out": out, "times": tout}
        return out, tout, stats, details


class SharedEventModel(nn.Module):
    def __init__(self, bands=40, dim=32, depth=8, groups=5, beta=1., cf_credit=True,
                 memory_backend="linear", classes=20, readout="mean", continuous_dim=0,
                 evidence_count=0, phase_period=None, phase_seed=6, phase_correction_bound=.25,
                 phase_margin_guard=False):
        super().__init__()
        if bands < 1 or groups < 1 or classes < 1 or dim < 4 or evidence_count < 0 or continuous_dim < 0:
            raise ValueError("Invalid model dimensions")
        if readout not in ("mean", "last"):
            raise ValueError("readout must be mean or last")
        self.readout = readout
        self.classes, self.evidence_count = classes, evidence_count
        if depth < 1 or dim % 4 or bands % groups or not 0 < beta/depth <= 1:
            raise ValueError("Invalid dimensions or residual bound")
        # The depth-1 control uses alpha=1 under the same beta/depth rule.
        # Its conditional lower bound is zero; the strict invertibility
        # certificate applies only when alpha<1, as in the depth-8 model.
        self.bands, self.dim, self.groups = bands, dim, groups
        self.embedding = nn.Embedding(bands, dim//4)
        nn.init.normal_(self.embedding.weight, std=.5)
        self.head = nn.Linear(dim+1, classes)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        self.layers = nn.ModuleList([RaceLayer(dim, depth, beta, cf_credit,
                                             memory_backend=memory_backend) for _ in range(depth)])
        self.register_buffer("time_constants", torch.tensor([.05, .2, .8]))
        self.register_buffer("center", torch.zeros(dim+1))
        self.register_buffer("scale", torch.ones(dim+1))
        self.register_buffer("whitener", torch.eye(dim+1))
        # Extra components initialize after legacy parameters: the zero-evidence
        # model can load existing E118/E119 checkpoints without a changed trunk.
        self.continuous = nn.Linear(continuous_dim, dim, bias=False) if continuous_dim else None
        if evidence_count:
            self.evidence_weights = nn.Parameter(torch.full((evidence_count,), 1/evidence_count))
            self.evidence_gate = nn.Linear(dim+1, evidence_count, bias=False)
            nn.init.zeros_(self.evidence_gate.weight)
        else:
            self.register_parameter("evidence_weights", None)
            self.evidence_gate = None
        self.phase_memory = None
        self.phase_correction_bound = float(phase_correction_bound)
        self.phase_margin_guard = bool(phase_margin_guard)
        if phase_period is not None:
            if evidence_count or phase_correction_bound < 0:
                raise ValueError("Phase clocks currently use their own bounded query readout")
            from .phase_memory import PhaseMemory
            self.phase_memory = PhaseMemory(bands, classes, phase_period, seed=phase_seed)

    def forward(self, b, t, c, ids, size, sequential=False, overrides=None, trace=False,
                continuous=None, evidence=None, expert_mode="combined"):
        if expert_mode not in ("combined", "core", "memory"):
            raise ValueError(expert_mode)
        phi = torch.cat((t.new_ones((len(t), 1)), torch.exp(-t[:, None]/self.time_constants)), -1)
        x = (self.embedding(b)[:, :, None]*phi[:, None, :]).flatten(1)
        if self.continuous is not None:
            if continuous is None or len(continuous) != len(x):
                raise ValueError("Continuous event marks are required")
            x = x + self.continuous(continuous)
        elif continuous is not None:
            raise ValueError("Model has no continuous input adapter")
        original_t = t
        width = self.bands//self.groups
        stats, traces = [], []
        for j, layer in enumerate(self.layers):
            keys = ids*(self.groups+1) + (b + (width//2 if j%2 and self.groups > 1 else 0))//width
            x, t, st, tr = layer(x, t, c, keys, sequential,
                                 (overrides or {}).get(j), trace)
            stats.append(st)
            if trace:
                traces.append(tr)
        mass = x.new_zeros(size).index_add(0, ids, c)
        mean = x.new_zeros((size, self.dim)).index_add(0, ids, x*c[:, None])/mass[:, None]
        if self.readout == "last":
            # Inputs are grouped by example, with chronological order inside it.
            ends = torch.bincount(ids, minlength=size).cumsum(0)-1
            mean = x[ends]
        summary = torch.cat((mean, torch.log1p(mass[:, None])/10), -1)
        features = ((summary-self.center)/self.scale) @ self.whitener
        core_logits = self.head(features)
        logits = core_logits
        if self.evidence_count:
            if evidence is None or evidence.shape != (size, self.evidence_count, self.classes):
                raise ValueError("Expected evidence shaped [queries, experts, outcomes]")
            weights = self.evidence_weights + self.evidence_gate(features)
            memory_logits = torch.einsum("be,bec->bc", weights, evidence)
            logits = core_logits + memory_logits
            if expert_mode == "core":
                logits = core_logits
            elif expert_mode == "memory":
                logits = memory_logits
        elif expert_mode == "memory" and self.phase_memory is None:
            raise ValueError("No evidence memory configured")
        phase_stats = {}
        if self.phase_memory is not None:
            if not torch.all(c == 1):
                raise ValueError("The initial phase adapter expects one occurrence per input event")
            phase_scores, _ = self.phase_memory(b, ids, size)
            phase_scores = phase_scores.to(core_logits.dtype)
            margins = phase_scores.sort(-1, descending=True).values
            gap = margins[:, 0]-margins[:, 1]
            bound = self.phase_correction_bound
            bounds = gap.new_full((size,), bound)
            if self.phase_margin_guard:
                # Reserve half the phase lead, even under opposing corrections.
                # This mode can calibrate confidence but cannot change the class.
                bounds = torch.minimum(bounds, gap/4)
            correction = bounds[:, None]*torch.tanh(core_logits/bounds[:, None].clamp_min(1e-12))
            logits = phase_scores+correction
            if expert_mode == "core":
                logits = core_logits
            elif expert_mode == "memory":
                logits = phase_scores
            phase_stats = {"phase_symbols":len(b),"phase_clock_candidates":size*self.classes,
                           "phase_winners":phase_scores.argmax(-1).tolist(),
                           "phase_margin":gap.detach().tolist(),
                           "phase_effective_bound":bounds.detach().tolist(),
                           "phase_certified":int((gap>2*bounds).sum())}
        return logits, summary, {"layers": stats, "packets": len(t),
                                 "max_payload": float(x.detach().abs().max()),
                                 "mean_added_delay_ms": float((t-original_t).detach().mean()*1000),
                                 **phase_stats}, traces
