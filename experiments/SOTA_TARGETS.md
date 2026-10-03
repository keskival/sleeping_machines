# Public benchmarks where the native architecture can reach the state of the art

User request (3 October 2026): find benchmarks where we can beat the published state of the art, and work toward them
systematically. Credibility rules: the official data, splits and metric code, multiple seeds or the official repeat
protocol, every published number cited to its source, and our numbers only from completed result files. A "state of the
art" claim means better than every entry on the cited leaderboard under its own metric; an efficiency claim adds the
leaderboard's own resource columns (footprint, effective operations) computed the same way.

## Selected targets (ranked by feasibility × credibility)

### 1. NeuroBench chaotic function prediction (Mackey-Glass, τ = 17)

Leaderboard ([NeuroBench leaderboard.rst](https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst)):
LSTM sMAPE **13.37** (footprint 4.90e5 bytes, dense 6.03e4, effective MACs 6.03e4), ESN 14.79 (2.81e5; effective MACs
4.37e3). Protocol (official `examples/mackey_glass/lstm_benchmark.py`, neurobench 2.3.0): the official downloaded series
`mg_17.npy` (do not regenerate), 30 repeats with start offsets 0, 0.5, …, 14.5 Lyapunov times (75 points each),
10 Lyapunov times (750 points) of teacher-forced training, then 750 points of autonomous prediction scored by the
official SMAPE metric, averaged over the repeats. Fit: tiny data, CPU minutes per repeat. Why it suits us: rotating
memories are damped complex oscillators with learned frequencies and timescales; the races give input-dependent routing
between regimes of the attractor. Risk: the leaderboard is sparse (two baselines), so beating it is necessary but the
claim must also show the comparison with the strongest published methods for this protocol if any appear.

### 2. NeuroBench non-human primate motor prediction (primate reaching)

Leaderboard (same source): **AEGRU R² 0.71** (footprint 45,500 B), GRU-t1 0.707, bigSNN 0.698, tinyRSNN 0.66 (27,144 B,
304 effective ACs), baselines 0.59. Task: decode fingertip velocity from multi-unit spike trains of macaque motor cortex
(O'Doherty et al. 2017), six sessions (three per monkey), R² averaged, 4 ms bins at 250 Hz. Why it suits us: the input
is spike events; timing and sparse addressed state are the native representation; efficiency columns reward sparse
activity. Source code for the top entries: [fmi-basel/neural-decoding-RSNN](https://github.com/fmi-basel/neural-decoding-RSNN),
[arXiv 2410.22283](https://arxiv.org/abs/2410.22283) (AEGRU, BioCAS 2024 grand challenge winner).

### 3. Spiking Heidelberg Digits, official test (stretch)

Best published accuracies in SHD_FRONTIER_PROTOCOL.md: 96.44% (Zhang et al. 2026), S7 96.3%, delay-attention 96.26%,
EventSSM 95.9%. Our best earlier private speaker-held-out result was 79.7% with a pre-credit, pre-compiled core.
Revisit only after targets 1–2, with route credit, compiled training and width.

### Not selected now

Keyword FSCIL and event-camera detection (GPU-scale vision/audio), DVS128 Gesture official (best published about 98–99%;
our coarse-packet protocol is at 69% and its strong control at 77.6%), Long Range Arena (16K-length tasks; CPU cost),
PhysioNet 2012 (AUROC about 0.87–0.90; tabular-irregular, credentialing and many imputation baselines).

## Work plan

1. Mackey-Glass: data from the official URL; our driver trains the integrated native core (route credit, compiled)
   teacher-forced on each repeat and predicts autonomously; the official SMAPE implementation computes the score.
   Development: tune on repeats outside the scored protocol only where the protocol allows (the official scripts tune
   on τ = 17 itself; we fix all hyperparameters before the 30-repeat run and report every variant tried).
2. Primate reaching: official loader and splits; spike events as native events; R² per session; footprint and traced
   effective operations, plus the NeuroBench hook-based counts where they apply.
3. For each completed target: FINDINGS, report appendix, and a reproducible one-command script.
