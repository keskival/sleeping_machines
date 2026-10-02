# AWS: timestamp-aware dense controls for the joint text + event task (Theory §394)

Purpose: the calibration bar for milestone 3. The native pilot runs on curie
(queue curie_joint_event_language_20261002T133000Z). The task, its shortcut-removal contracts and the table bar
are in experiments/joint_event_language_tasks.py and tests/test_joint_event_language_tasks.py.

Task version 2 (phase-of-elapsed-time rule; v1 leaked and is retired, Theory §394 revision).

Controls (experiments/joint_event_dense_controls.py, contracts in tests/test_joint_event_dense_controls.py):
continuous-time GRU (learned per-unit decay, log(1+dt) input) and a causal time-encoded Transformer (absolute
time + gap encodings). Same fit/dev/confirmation seeds (1301/2301/3301), 512 fit / 256 dev episodes, 8 passes,
lr .003, 16-episode Adam windows, clipping 1, selection by lowest dev NLL, all fitting work traced.

One-job queues through run_safe, seeds 6/7/8, width 64, both models (six fits). The parameter counts differ from the
native core; report both whole-fit and per-query work in the same units as the native result. Calibration passes if
a control beats the table bar by >= 20 points. Only then do native-versus-control comparisons count as evidence
about learning (§394). Dense controls are labelled controls, never promoted into the research architecture.

    python experiments/joint_event_dense_controls.py --tag aws_joint_event_gru_w64_s6_<ts> --model gru --seed 6
    python experiments/joint_event_dense_controls.py --tag aws_joint_event_transformer_w64_s6_<ts> --model transformer --seed 6

## History-length ladder (Theory §395), after the base calibration

`--background 30 40` and `--background 120 140`: same seeds and models, `--fit 256 --epochs 4` to bound cost.
Report accuracy, per-query inference work and whole-fit work per rung. The Transformer's per-query work should
grow ∝ history; the GRU's should stay flat. These fits begin only after the base controls pass calibration
(≥ 20 points over the table bar).
