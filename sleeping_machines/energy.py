"""Energy estimates from measured operation counts.

The simulators count what happened (synaptic events, spikes, plasticity updates,
multiply-accumulates). This module prices those counts on hardware profiles.
Every number is an estimate with a stated source, and results are reported for
several profiles plus a break-even cost, never as a single figure.

Per-operation energies for digital logic and SRAM are 45 nm figures from
M. Horowitz, "Computing's energy problem (and what we can do about it)",
ISSCC 2014: 8-bit add 0.03 pJ, 8-bit multiply 0.2 pJ, 32-bit add 0.1 pJ,
16-bit float add 0.4 pJ / multiply 1.1 pJ, 64-bit read from an 8 KB SRAM 10 pJ
(1.25 pJ per byte), from a 1 MB SRAM ~100 pJ (12.5 pJ per byte).
Loihi figures are measured values reported by M. Davies et al., "Loihi: a
neuromorphic manycore processor with on-chip learning", IEEE Micro 2018
(14 nm): about 23.6 pJ per synaptic operation, 120 pJ per synaptic update,
and a few pJ to route a spike. Treat these as order-of-magnitude anchors.

Profiles price the same abstract operations:

    mac          one multiply-accumulate with its weight fetch (dense)
    synop        one synaptic event: weight fetch + add into a membrane potential
    spike        emitting and routing one spike
    plasticity   one synaptic weight update (read-modify-write)
    neuron_step  one clocked neuron update (read state, integrate, compare, write)
"""
from dataclasses import dataclass, asdict

PJ = 1e-12

ADD8, MUL8, ADD16, ADD32 = 0.03, 0.2, 0.05, 0.1
FADD16, FMUL16 = 0.4, 1.1
SRAM_SMALL_BYTE, SRAM_LARGE_BYTE = 1.25, 12.5


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    mac: float = 0.0
    synop: float = 0.0
    spike: float = 0.0
    plasticity: float = 0.0
    neuron_step: float = 0.0


PROFILES = {
    # Dense, synchronous, int8 inference with weights in small on-chip buffers
    # (weight-stationary): a generous setting for the dense baseline.
    "dense_int8": Profile(
        "dense_int8", "Dense int8 accelerator, weights in local SRAM, batch 1",
        mac=MUL8 + ADD32 + SRAM_SMALL_BYTE),
    # Same, but each weight fetch is shared by a batch of 256 inputs.
    "dense_int8_batched": Profile(
        "dense_int8_batched", "Dense int8 accelerator, weight fetch shared by a batch of 256",
        mac=MUL8 + ADD32 + SRAM_SMALL_BYTE / 256),
    # Dense fp16 training arithmetic (backprop); weights read per use, batch 1 equivalent.
    "dense_fp16_train": Profile(
        "dense_fp16_train", "Dense fp16 training, weights in local SRAM",
        mac=FMUL16 + FADD16 + 2 * SRAM_SMALL_BYTE,
        plasticity=2 * 2 * SRAM_SMALL_BYTE + FADD16),
    "dense_fp16_train_batched": Profile(
        "dense_fp16_train_batched", "Dense fp16 training, weight traffic shared by a batch of 256",
        mac=FMUL16 + FADD16 + 2 * SRAM_SMALL_BYTE / 256,
        plasticity=(2 * 2 * SRAM_SMALL_BYTE + FADD16) / 256),
    # The "suitable architecture": event-driven, near-memory. A binary spike makes
    # the synaptic operation an addition: 8-bit weight fetch from a local SRAM plus a
    # 16-bit add into a register-resident potential. No clock-driven work.
    "event_ideal": Profile(
        "event_ideal", "Event-driven near-memory digital (45 nm logic, local SRAM)",
        synop=SRAM_SMALL_BYTE + ADD16, spike=2.0,
        plasticity=2 * SRAM_SMALL_BYTE + ADD16 + ADD8),
    # Event-driven, but with weights in large shared SRAM (no near-memory benefit).
    "event_shared_sram": Profile(
        "event_shared_sram", "Event-driven with weights in a large shared SRAM",
        synop=SRAM_LARGE_BYTE + ADD16, spike=5.0,
        plasticity=2 * SRAM_LARGE_BYTE + ADD16 + ADD8),
    # Measured silicon today.
    "loihi": Profile(
        "loihi", "Loihi, measured per-operation energies (14 nm)",
        synop=23.6, spike=3.5, plasticity=120.0),
    # A clocked simulation of the same spiking network: every neuron is updated at
    # every timestep (state read + add + compare + write in local SRAM).
    "clocked_snn": Profile(
        "clocked_snn", "Synchronous spiking simulation, every neuron updated every step",
        synop=SRAM_SMALL_BYTE + ADD16, spike=2.0,
        plasticity=2 * SRAM_SMALL_BYTE + ADD16 + ADD8,
        neuron_step=4 * SRAM_SMALL_BYTE + ADD16 + ADD16),
}


def energy_joules(counts, profile):
    """Price a dict of operation counts on a profile."""
    p = PROFILES[profile] if isinstance(profile, str) else profile
    return PJ * (counts.get("macs", 0) * p.mac
                 + counts.get("synops", 0) * p.synop
                 + counts.get("spikes", 0) * p.spike
                 + counts.get("plasticity", 0) * p.plasticity
                 + counts.get("neuron_steps", 0) * p.neuron_step)


def break_even_synop_pj(dense_joules, event_counts, spike_pj=2.0, plasticity_ratio=2.0):
    """The per-synop energy (pJ) at which the event network costs as much as the
    dense one, with spike and plasticity costs scaled alongside."""
    per_unit = (event_counts.get("synops", 0)
                + event_counts.get("plasticity", 0) * plasticity_ratio)
    fixed = event_counts.get("spikes", 0) * spike_pj
    return (dense_joules / PJ - fixed) / per_unit if per_unit else float("inf")


def profiles_table():
    return [asdict(p) for p in PROFILES.values()]
