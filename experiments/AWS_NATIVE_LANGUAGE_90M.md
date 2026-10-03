# AWS: native language at 90M characters (segment-batched, compiled)

User request (3 October 2026): run the 90M native language variants on the AWS host, through its queue. Use `main`.
Read `AGENTS.md`, `experiments/HANDOFF.md` and THEORY §§409–412 (experiments/theory/59_statistic_valued_race_memory.md)
first.

## Revision 2 (10:30 UTC): run these arms, with route credit

THEORY §413 found that the fast language path trained the race address by timing alone. Adding the linearized
local-expectation route credit (`--route-credit linear`; forward values unchanged, same work) improved p32/d4 pool 2
at 10M from 2.507 to **2.370** test bpc. That beats the no-selection pool-1 control (2.439) and the one-pass
Transformer-256×2 (2.427). Without credit, pool 4 was worse than pool 2. **Revision 1 (below) is superseded before it
ran. Do not admit its queues** (`aws_language_90M_*_20261003T063000Z.txt`; files kept for the record). Admit instead:

| Queue (one job each) | Arm | Parameters |
|---|---|---:|
| `queue/aws_language_90M_contracts_20261003T103000Z.txt` | contracts, then 60-window pilots of the three arms | — |
| `queue/aws_language_90M_r2_p32d4_linear_20261003T103000Z.txt` | payload 32, depth 4, pool 2, route credit | 108,875 |
| `queue/aws_language_90M_r2_p32d8_skip2_linear_20261003T103000Z.txt` | payload 32, depth 8, pool 2, near-identity init from layer 2, route credit | 210,043 |
| `queue/aws_language_90M_r2_p64d4_linear_20261003T103000Z.txt` | payload 64, depth 4, pool 2, route credit | about 422K |

All other settings, the evaluation, the references and the admission steps below are unchanged. curie throughput at
10M: p32/d4 about 7,760 characters/s with credit (90M about 3.3 h), p32/d8 about 4,400 (about 5.7 h), p64/d4 not yet
measured (curie v5 arm). The curie v5 queue also tests the credit at pool 4, at depth 8 and in its write-address variant.
A later revision may add a pool-4 or write-credit arm, always under new names.

## Revision 1 (06:30 UTC; superseded, not run)

## What runs

The integrated native core (AddressedEventHeads: factorized temporal races, sparse addressed writes into persistent
rotating memories, transport between layers; no dense carrier, no statistical experts) trained on text8[0:90M] by
experiments/language_batched_benchmark.py. That is one pass of 64 lanes × 128-character segments (state reset per
segment; exact BPTT within it) = 8,192 characters per Adam update, about 10,990 updates. lr .004 with cosine annealing,
clip 1, near-identity init from layer 2 (unit gates −4, intermediate transport closed), seed 6. Compiled layer steps
(§412; contract-tested against the batched path). Exact-resume checkpoints every 250 windows.

| Queue (one job each) | Arm | Parameters | Selected writes / char |
|---|---|---:|---:|
| `queue/aws_language_90M_p16d8_skip2_20261003T063000Z.txt` | payload 16, depth 8, pool 2 | 54,907 | 16 |
| `queue/aws_language_90M_p32d8_skip2_20261003T063000Z.txt` | payload 32, depth 8, pool 2 | 210,043 | 16 |
| `queue/aws_language_90M_p32d8_pool4_skip2_20261003T063000Z.txt` | payload 32, depth 8, pool 4 | 346,331 | 16 |

The pool-4 arm holds more state and parameters at the same selected activity: capacity beyond activity.

**Evaluation.** E64 windows on the 1M DEV text8[90M:91M] and the 1M test text8[95M:96M], at T = 128 (the training
length) and at T = 256 (`--eval-segment 256`, the saved controls' window), with the same final weights. Development
progress on the first 50K DEV characters every 1,000 windows. Configuration selection among the 90M arms is by DEV
bpc (the §409 correction). Test is reporting-only.

**References** (same test targets): LSTM-512 90M / 6 passes 1.661 bpc (3.89 PFLOPs); Transformer 256×4 90M / 4 passes
1.604 (8.00 PFLOPs). These are multi-pass, larger-model references; the native arms make one pass. At 10M the matched
one-pass E64 controls are LSTM-256 2.171 and Transformer-256×2 2.427. The completed 10M native p16/d8 arm scored
2.719 (T = 128) / 2.719 (T = 256) test bpc. Table every row with its passes, parameters and whole-fit and per-character
work in the same units. Do not fill a cell with a pilot or curve value.

## Admission on the AWS host

1. Pull `main`, inspect the active slots/coordinator, `MemAvailable`, and running jobs (AGENTS.md). Admit these only
   into free guarded one-thread slots (or the normal host lock), never beside the AGENTS limit, and never bypassing a
   lock.
2. Run `queue/aws_language_90M_contracts_20261003T063000Z.txt` first. It runs the compiled/driver contracts, then one
   60-window pilot per arm to measure characters/s and RSS on this CPU. torch.compile's inductor needs a C++ compiler
   and the Python development headers (`Python.h`). If the compiled contract fails for that reason, stop and report; do
   not silently drop `--compiled` (the eager path is numerically equivalent but about 4× slower).
3. Set each arm's `JOB_TIMEOUT_S` from its pilot: 90,000,000 / (pilot characters/s) + final evaluation (about 4M
   forward characters) with a 1.5× margin. Set RSS caps from the pilot RSS with margin. Keep the RSS watchdog and the
   8 GiB `MemAvailable` floor. On curie (one thread), p16/d8 trained at 5,700–5,900 characters/s (90M ≈ 4.4 h) with RSS
   1.28 GB. The curie 10M p32 arms are measuring now; their throughput will be added here when they complete.
4. After an interruption, resume with a new one-job queue whose job name is new and whose arguments are the same plus
   `--resume`. The checkpoint (`results/language_batched/checkpoints/<tag>.pt`) refuses changed arguments. Preserve the
   runner log, the job log, the result JSON and the checkpoint.

Results go to `experiments/results/language_batched/<tag>.json` (status, curve, DEV/test at both window lengths,
throughput, parameters, traced work per character, source hashes, hardware). Record completed numbers in FINDINGS and
the report only from completed result files.
