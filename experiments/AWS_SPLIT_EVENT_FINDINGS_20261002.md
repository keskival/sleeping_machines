# Completed split-event evidence — 2 October 2026

All 11 pilots and 22 prerequisite stages completed and were published. Seed6 only; 128 fitting queries ×4 passes, 256 development queries, H2/d8/depth8. Development selects epochs; intervals do not correct selection or measure seed uncertainty. No supremacy claim.

| Model | Dev accuracy | Cleared state | Fit MFLOPs | Fit MFLOPs/query | Inference MFLOPs/query |
|---|---:|---:|---:|---:|---:|
| order_S16_private_P0 | 0.4414 | 0.2070 | 274.359 | 0.536 | 0.107 |
| order_S16_private_P2 | 0.4492 | 0.2617 | 266.103 | 0.520 | 0.104 |
| order_S16_shared_P0 | 0.7539 | 0.2031 | 256.914 | 0.502 | 0.107 |
| order_S16_shared_P2 | 0.5625 | 0.3047 | 248.943 | 0.486 | 0.104 |
| order_S4_private_P0 | 0.5430 | 0.2539 | 260.112 | 0.508 | 0.107 |
| order_S4_private_P2 | 0.4727 | 0.2539 | 252.024 | 0.492 | 0.104 |
| order_S4_shared_P0 | 0.7070 | 0.2617 | 256.541 | 0.501 | 0.107 |
| order_S4_shared_P2 | 0.5664 | 0.2344 | 248.574 | 0.485 | 0.104 |
| paired_timing_S4_private_P0_observed | 0.9531 | 0.5000 | 259.847 | 0.508 | 0.106 |
| paired_timing_S4_private_P0_rank | 0.5000 | 0.5000 | 259.850 | 0.508 | 0.106 |
| paired_timing_S4_private_P2_observed | 0.8242 | 0.5000 | 251.760 | 0.492 | 0.104 |

Arithmetic uses 2 FLOPs/MAC; special functions are separately saved in raw JSON. Fitting includes losing proposals, backward, clipping and Adam. All rows use the same denominator and units.

Paired timing: observed P0 reaches 244/256 (95.31%) versus rank-only 128/256 (50%). Identical marks/address/order with opposite labels make this an information-matched temporal signal. P2 reaches 211/256 (82.42%); protection is not a free accuracy improvement. This establishes useful physical-time information in this variant, not the unique value of race computation versus a timestamp-aware dense control.

Order: S16 shared P0 reaches 75.39% versus private P0 44.14%, but sharing also changes source embeddings. P2 shared improves long-gap64 accuracy to47.27% from20.31%, while ordinary dev accuracy drops from75.39% to56.25%. Replicate and isolate embedding/map sharing before attributing the gain.

The prepared factorial route/write audit completed in7.107s at399708KiB RSS. Mean absolute persistent-commit effect .012694 substantially exceeds value-linearization residual .000288 across the twelve fixed probes. These are separate absolute summaries, not an additive percentage attribution. Weight/RNG contracts passed;48 forwards/12 backwards/12288 races were charged. Fixed time/noise, one population/address; no expected-gradient claim. This prioritizes addressed-write credit diagnosis over a payload-only correction.

Next: the prespecified banknote confirmation battery uses unchanged integrated architecture and three stronger controls; no new architectural scale-up is inferred from these small pilots. Event replication and an integrated state-aware credit contract remain subsequent research priorities.
