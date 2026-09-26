# E18: does sparse fan-in make race networks cheaper than an equally accurate dense model?

Written 2026-09-26, before any E18 run. The pilot choice below is made on the validation split
only and recorded here, dated, before any test-set run.

## Why this experiment

The energy verdict so far: only a single racing layer beats an equally accurate dense model at
inference (about 2.4× at batch 1). Hidden-layer race networks use about 108k synaptic events per
image, while a 32-unit MLP reaches the same accuracy (0.966) with 25.4k multiply-accumulates. In
race networks, work is paid for **inputs that arrive before a node decides**, so fan-in is the
lever. A short pilot (E9) gave fan-in 32 14× fewer events *and* higher accuracy. This experiment
settles, at full length and several seeds, whether sparse race networks beat dense models at
matched accuracy. Either answer matters: the hardware argument for the proposal rests on it.

## Networks

- **Race (E14 code):** one hidden layer (the most accurate depth), crl_fa with credit conservation
  (`--zero-sum 1`), k = 3 winners per group of 10, 3 epochs. Each hidden node reads a random subset
  of `--fanin-in` F input pixels among the pixels that ever spike.
- **Pilot grid (validation split, seed 0):** F ∈ {16, 32, 64, 128, dense} × width ∈ {400, 1000}.
- **Pilot choice (fixed rule):** among pilot configurations with validation accuracy ≥ 0.955, take
  the one with the lowest idealized-event inference energy; also carry forward the lowest-energy
  configuration with accuracy ≥ 0.945.
- **Dense baselines:** the existing MLP frontier (hidden 16 / 32 / 64 / 128 / 256; linear models with
  pooling), trained on the same latency-coded input, plus MLP hidden 16, 24, 32 re-run at the
  confirmatory seeds.

## Measures (test set, 3 seeds per chosen configuration)

- Test accuracy (mean, 95% interval over seeds).
- Synaptic events and spikes per image (inference, counted by the simulator).
- Inference energy under `sleeping_machines/energy.py` profiles: idealized event-driven near-memory
  (1.3 pJ per synaptic event, 2 pJ per spike) and measured Loihi (23.6 pJ per synaptic event),
  against dense int8 (1.55 pJ per multiply-accumulate at batch 1; 0.31 pJ at batch 256).
- Matched accuracy: each race configuration is compared with the **cheapest dense model whose
  accuracy is at least the race's mean accuracy**. Dense energies are interpolated along the MLP
  frontier only for reporting, never for the verdict.

## Decision rules (fixed now)

- **Event inference wins (idealized profile, batch 1)** if the race's inference energy is below the
  matched dense model's at batch 1 for every seed.
- **Wins against batched hardware** if it is also below the matched dense model at batch 256.
- **Robust to hardware** if it also wins under the Loihi profile at batch 1.
- Otherwise the verdict is that sparse race networks do not beat an equally accurate dense model, and
  the report says so.

## Known limits

- Energies are operation counts priced with published per-operation figures, not chip measurements.
- Fan-in is random, not learned or local (patches); learned sparsity could do better.
- Latency-coded MNIST is one task; the dense model sees the same intensities.
