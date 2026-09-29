"""Shared event backbone: addressed temporal state, hard races and local loser credit.

A query consumes only an explicitly supplied observed prefix. This module has
no knowledge of speech, text, markets or synthetic labels. Readout semantics
and event units are declared by configuration; objective functions live in
objectives.py. E118/E119 are the zero-expert, pooled-readout special case.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .event_memory import segmented_memory

class RaceLayer(nn.Module):
    def __init__(self, dim, depth, beta, cf_credit, options=3, memory_backend="doubling",
                 global_context=False, value_backward="full"):
        super().__init__()
        self.alpha, self.cf_credit = beta/depth, cf_credit
        self.options, self.dim = options, dim
        if value_backward not in ("full", "winner"):
            raise ValueError("value_backward must be full or winner")
        self.value_backward = value_backward
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
        # Zero columns preserve the complete legacy computation and RNG stream.
        # Separate normalization keeps this new map differentiable at zero.
        self.register_parameter("bridge_value", nn.Parameter(torch.zeros(options, dim, dim+1))
                                if global_context else None)
        self.register_parameter("bridge_route", nn.Parameter(torch.zeros(options, dim+1))
                                if global_context else None)

    def forward(self, x, t, count, keys, sequential=False, override=None, trace=False,
                global_keys=None, order_override=None, schedule=None, emit_schedule=False):
        # Learned delays can reorder carriers. Memory sees actual arrival
        # order within each receiver, with the original index breaking ties.
        if order_override is not None and not trace:
            raise ValueError("Frozen arrival order is a trace-only diagnostic")
        time_order = torch.argsort(t, stable=True) if order_override is None else order_override
        if order_override is not None and (time_order.shape != t.shape or not torch.equal(
                time_order.sort().values, torch.arange(len(t),device=t.device))):
            raise ValueError("Frozen arrival order must be an event permutation")
        inverse = torch.argsort(time_order)
        mem, mass, work = self.memory(
            x[time_order], t[time_order], count[time_order], keys[time_order],
            self.log_tau.clamp(math.log(.002), math.log(4.)).exp(), sequential)
        mem, mass = mem[inverse], mass[inverse]
        features = torch.cat((x[:, None, :].expand(-1, self.options, -1), mem,
                              (mass/(1+mass))[:, :, None]), -1)
        scores = (features*self.route[None, :, :]).sum(-1) + self.route_bias
        bridge_features, bridge_w, global_work = None, None, 0
        if self.bridge_value is not None:
            if global_keys is None:
                raise ValueError("Global context requires explicit per-query receiver keys")
            global_mem, global_mass, global_work = self.memory(
                x[time_order], t[time_order], count[time_order], global_keys[time_order],
                self.log_tau.clamp(math.log(.002), math.log(4.)).exp(), sequential)
            bridge_features = torch.cat((global_mem[inverse],
                (global_mass[inverse]/(1+global_mass[inverse]))[:, :, None]), -1)
            scores = scores + (bridge_features*self.bridge_route[None, :, :]).sum(-1)
            bridge_w = self.bridge_value / self.bridge_value.abs().sum(-1, keepdim=True).clamp_min(1)
        elif global_keys is not None:
            raise ValueError("This layer has no global context channel")
        delay = .001 + .010*torch.sigmoid(-scores) if schedule is None else schedule["delays"]
        winner = delay.argmin(-1) if schedule is None else schedule["winner"]
        if override is not None:
            event, choice = override
            winner = winner.clone()
            winner[event] = choice
        w = self.value / self.value.abs().sum(-1, keepdim=True).clamp_min(1)
        rows = torch.arange(len(x))
        def selected_values():
            selected_correction = torch.zeros_like(x)
            for k in range(self.options):
                selected = torch.nonzero(winner == k, as_tuple=True)[0]
                value = F.linear(features[selected, k], w[k], self.bias[k])
                if bridge_features is not None:
                    value = value + F.linear(bridge_features[selected, k], bridge_w[k])
                selected_correction = selected_correction.index_copy(0, selected, torch.tanh(value))
            return selected_correction

        credit_active = self.training and self.cf_credit and schedule is None
        sparse_backward = credit_active and self.value_backward == "winner" and not trace
        if credit_active or trace:
            # All alternatives still teach routing. Only the realized value
            # needs ordinary differentiation; tracing keeps the full graph.
            with torch.set_grad_enabled(torch.is_grad_enabled() and not sparse_backward):
                preactivation = torch.einsum("ekf,kdf->ekd", features, w)
                if bridge_features is not None:
                    preactivation = preactivation + torch.einsum("ekf,kdf->ekd", bridge_features, bridge_w)
                alternatives = torch.tanh(preactivation + self.bias)
                correction = alternatives[rows, winner]
            value_evaluations = len(x)*self.options
            if sparse_backward:
                selected = selected_values()
                # Keep the exact original forward value. The zero-valued term
                # supplies its winner-only derivative (up to contraction rounding).
                correction = correction + (selected-selected.detach())
                value_evaluations += len(x)  # Charge the selected forward recomputation.
        else:
            # Inference and pathwise control evaluate only selected values.
            correction = selected_values()
            alternatives = None
            value_evaluations = len(x)
        winning_delay = delay[rows, winner]
        out = x + self.alpha*correction
        tout = t + winning_delay
        if credit_active:
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
        if self.value_backward == "winner":
            stats["differentiable_value_evaluations"] = len(x) if sparse_backward or alternatives is None else len(x)*self.options
        if bridge_features is not None:
            stats.update(global_scan_compositions=global_work,
                         global_context_packets=len(x),
                         global_state_receivers=int(torch.unique(global_keys).numel()))
        details = None
        if trace:
            details = {"winner": winner, "alternatives": alternatives,
                       "delays": delay, "out": out, "times": tout, "time_order": time_order}
            if bridge_features is not None:
                details["global_features"] = bridge_features
        elif emit_schedule:
            details = {"winner":winner,"delays":delay,"time_order":time_order}
        return out, tout, stats, details


class SharedEventModel(nn.Module):
    def __init__(self, bands=40, dim=32, depth=8, groups=5, beta=1., cf_credit=True,
                 memory_backend="linear", classes=20, readout="mean", continuous_dim=0,
                 evidence_count=0, phase_period=None, phase_seed=6, phase_correction_bound=.25,
                 phase_margin_guard=False, phase_only=False, global_context_layers=(), value_backward="full",
                 separate_keys=False):
        super().__init__()
        if bands < 1 or groups < 1 or classes < 1 or dim < 4 or evidence_count < 0 or continuous_dim < 0:
            raise ValueError("Invalid model dimensions")
        if readout not in ("mean", "last", "weighted"):
            raise ValueError("readout must be mean, last or weighted")
        self.readout = readout
        self.classes, self.evidence_count = classes, evidence_count
        self.bands, self.dim, self.groups = bands, dim, groups
        self.phase_only = bool(phase_only)
        global_context_layers = tuple(global_context_layers)
        if len(set(global_context_layers)) != len(global_context_layers) or any(
                not isinstance(j, int) or j < 0 or j >= depth for j in global_context_layers):
            raise ValueError("Invalid global context layers")
        self.global_context_layers = global_context_layers
        self.key_embedding, self.key_layers, self.key_continuous = None, None, None
        if self.phase_only:
            if phase_period is None or evidence_count or continuous_dim or depth != 0 or global_context_layers or separate_keys:
                raise ValueError("A phase-only model has depth zero, periodic state and no other readout")
            from .phase_memory import PhaseMemory
            self.phase_memory = PhaseMemory(bands, classes, phase_period, seed=phase_seed)
            self.layers = nn.ModuleList([])
            self.phase_margin_guard = True
            self.phase_correction_bound = 0.
            return
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
                                             memory_backend=memory_backend,
                                             global_context=j in global_context_layers,
                                             value_backward=value_backward) for j in range(depth)])
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
        self.readout_gain = None
        if readout == "weighted":
            self.readout_gain = nn.Linear(dim, 1, bias=False)
            nn.init.zeros_(self.readout_gain.weight)
        if separate_keys:self.freeze_keys_from_values()

    def freeze_keys_from_values(self):
        """Initialize an immutable local key stream for a value-learning phase.

        Keys make actual hard choices from each observed query. Values share
        those choices and clocks, but may learn without changing the keys.
        No per-example cached winner or externally frozen schedule is used.
        """
        if self.phase_only:
            raise ValueError("The phase-only primitive has no key/value carrier")
        self.key_embedding = copy.deepcopy(self.embedding).requires_grad_(False)
        self.key_layers = copy.deepcopy(self.layers).requires_grad_(False)
        self.key_continuous = copy.deepcopy(self.continuous)
        if self.key_continuous is not None:self.key_continuous.requires_grad_(False)
        for layer in self.key_layers:
            # The key stream owns the original local computation; global
            # columns teach value interactions separately in this phase.
            layer.register_parameter("bridge_value", None)
            layer.register_parameter("bridge_route", None)
            layer.cf_credit = False

    def forward(self, b, t, c, ids, size, sequential=False, overrides=None, trace=False,
                continuous=None, evidence=None, expert_mode="combined", replay_orders=None):
        if expert_mode not in ("combined", "core", "memory"):
            raise ValueError(expert_mode)
        if replay_orders is not None and not trace:
            raise ValueError("Frozen arrival schedules are trace-only diagnostics")
        if self.phase_only:
            if expert_mode == "core" or continuous is not None or evidence is not None or overrides:
                raise ValueError("The phase-only path has no neural core, continuous marks or overrides")
            if not torch.all(c == 1):
                raise ValueError("Phase state requires unit-count occurrence events")
            scores, phase = self.phase_memory(b, ids, size)
            gap = scores.sort(-1, descending=True).values
            gap = gap[:, 0]-gap[:, 1]
            angle = 2*math.pi*phase/self.phase_memory.period
            payload = torch.stack((angle.cos(), angle.sin()), -1)
            return scores, payload, {"layers":[],"packets":len(b),"max_payload":1.,
                "mean_added_delay_ms":0.,"phase_symbols":len(b),
                "phase_clock_candidates":size*self.classes,"phase_winners":scores.argmax(-1).tolist(),
                "phase_margin":gap.tolist(),"phase_effective_bound":[0.]*size,
                "phase_certified":int((gap>0).sum())}, []
        phi = torch.cat((t.new_ones((len(t), 1)), torch.exp(-t[:, None]/self.time_constants)), -1)
        x = (self.embedding(b)[:, :, None]*phi[:, None, :]).flatten(1)
        if self.continuous is not None:
            if continuous is None or len(continuous) != len(x):
                raise ValueError("Continuous event marks are required")
            x = x + self.continuous(continuous)
        elif continuous is not None:
            raise ValueError("Model has no continuous input adapter")
        original_t = t
        key_x, key_t = None, None
        if self.key_layers is not None:
            with torch.no_grad():
                key_x = (self.key_embedding(b)[:, :, None]*phi[:, None, :]).flatten(1)
                if self.key_continuous is not None:key_x = key_x+self.key_continuous(continuous)
                key_t = t
        width = self.bands//self.groups
        stats, traces = [], []
        for j, layer in enumerate(self.layers):
            keys = ids*(self.groups+1) + (b + (width//2 if j%2 and self.groups > 1 else 0))//width
            schedule, key_stats = None, None
            if self.key_layers is not None:
                key_layer = self.key_layers[j]
                key_layer.eval()
                with torch.no_grad():
                    key_x,key_t,key_stats,schedule = key_layer(key_x,key_t,c,keys,sequential,
                        (overrides or {}).get(j),False,emit_schedule=True)
            x, t, st, tr = layer(x, t, c, keys, sequential,
                                 (overrides or {}).get(j), trace,
                                 global_keys=ids if j in self.global_context_layers else None,
                                 order_override=(replay_orders or {}).get(j),schedule=schedule)
            if key_stats is not None:
                st["value_stream_value_evaluations"] = st["value_evaluations"]
                st["value_stream_scan_compositions"] = st["scan_compositions"]
                st["key_value_evaluations"] = key_stats["value_evaluations"]
                st["key_scan_compositions"] = key_stats["scan_compositions"]
                st["value_evaluations"] += key_stats["value_evaluations"]
                st["scan_compositions"] += key_stats["scan_compositions"]
                st["separate_key_stream"] = True
            stats.append(st)
            if trace:
                traces.append(tr)
        mass = x.new_zeros(size).index_add(0, ids, c)
        if self.readout_gain is not None:
            # A causal per-event gain and an associative numerator/mass state.
            # Zero initialization exactly recovers the existing count mean.
            gain = torch.exp(2*torch.tanh(self.readout_gain(x).squeeze(-1)/2))
            weights = c*gain
            weighted_mass = x.new_zeros(size).index_add(0, ids, weights)
            mean = x.new_zeros((size,self.dim)).index_add(0,ids,x*weights[:,None])/weighted_mass[:,None]
        else:
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
