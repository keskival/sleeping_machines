"""Warm common hard-race model with fine sources and trainable temporal phase."""
import torch
from torch import nn
from e139_fine_packet_model import FinePacketModel
from sleeping_machines.rotating_memory import rotating_memory


class LayerPhaseMemory:
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, x, times, counts, keys, taus, sequential=False):
        return rotating_memory(x, times, counts, keys, taus,
            self.layer.memory_phase, sequential)


class PhasePacketModel(FinePacketModel):
    def __init__(self, parent):
        super().__init__(parent)
        for layer in self.core.layers:
            layer.memory_phase = nn.Parameter(torch.zeros(layer.options, layer.dim//2))
            layer.memory = LayerPhaseMemory(layer)

    def forward(self, packed, trace=False):
        logits, payload, stats, traces = super().forward(packed, trace)
        # Two pair rotations (input/output in each layer), each with two
        # multiplications and one addition per scalar, plus angle products.
        evaluations = len(stats["layers"])*stats["packets"]*3*32
        stats.update(temporal_rotation_multiplies=evaluations*4,
            temporal_rotation_additions=evaluations*2,
            temporal_angle_multiplies=evaluations//2,
            temporal_sines=evaluations, temporal_cosines=evaluations,
            temporal_frequency_divisions=len(stats["layers"])*3*16,
            temporal_phase_parameters=sum(l.memory_phase.numel() for l in self.core.layers))
        return logits, payload, stats, traces


def phase_optimizer(model, parent, core_lr, fine_lr, phase_lr):
    # Recreate the old named groups explicitly: phase parameters are new.
    named = dict(model.core.named_parameters())
    groups = parent.get("optimizer_parameter_groups")
    if groups is None:
        groups = [[n for n in named if not n.endswith("memory_phase")]]
    expected = {n for n in named if not n.endswith("memory_phase")}
    if set(sum(groups, [])) != expected:
        raise ValueError("Parent core grouping differs")
    opt = torch.optim.Adam([named[n] for n in groups[0]], lr=core_lr)
    for group in groups[1:]:
        opt.add_param_group({"params": [named[n] for n in group], "lr": core_lr})
    opt.load_state_dict(parent["optimizer"])
    for group in opt.param_groups:
        group["lr"] = core_lr
    opt.add_param_group({"params": model.fine.parameters(), "lr": fine_lr})
    opt.add_param_group({"params": [l.memory_phase for l in model.core.layers], "lr": phase_lr})
    return opt
