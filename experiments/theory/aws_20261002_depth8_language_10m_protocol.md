# Depth8 causal text8 comparison at ten million characters

Explicit user direction: at least10M characters; small fits are correctness
and resource admission only, never advantage evidence. Two integrated native
models: original private maps versus maps shared acrossdepth/pool perhead.
Retain native clocks, causal time evolution, independent keys/values, sparse
receiver writes, private deep persistent state, small content-memory messages
and original counterfactual teacher. This driver does NOT install corrected
full replay; a language replay port still requires all-target causal return,
first-time-preserving shadow state and independent gradient/recovery contracts.

Frozen protocol: first10,000,000 text8 chars (9,999,999 next-character targets),
one pass, disjoint1M DEV chars at90,000,000; no officialtest. Depth8,H2,p16,pool2,
credit16, target-weighted Adam updates256, lr.002, warmup4096,clip1,seed7.
Both models share tokenizer/data/passes/update settings and original
initialization draws. Persistent state crosses credit and optimizer windows;
only credit is truncated. Report causal next-character BPC, measuredwall/RSS,
whole-fit and per-character full work and inference work together. Dense/count
references require identical train/DEV intervals and must retain their own
resource boundary. Single seed/reused DEV does not establish supremacy.

Each family must pass causal token/teacher parity, actual native live-state
Adam/RNG recovery and summed-gradient/partial-window recovery contracts, then
1025-char learning smoke under RSS1GB before ANY10M training admission. These
small runs are diagnostic only. At most2 guarded CPU slots, unique one-job
queues, globalreservation/slot locks,2GB RSS/6GB virtual,8GiB floor. Seven-day
watchdog budget accommodates sequential CPU emulation; actual smoke throughput
must be reported. Do not bypass locks; language matrix waits for the gesture
matrix to drain before acquiring host reservation. Each completed result and
checkpoint is committed/pushed. Long runs retain running JSON and restartable
checkpoint every16 optimizerwindows; source-exact recovery only.
