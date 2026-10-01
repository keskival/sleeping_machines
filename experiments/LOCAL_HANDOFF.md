# Local host: current research continuation

Use this file for future local progress updates. HANDOFF.md preserves the shared
history and AWS-host notes; append here to avoid competing end-of-file commits.
Do not modify running model/driver/helper sources or reuse changed run tags.

## Current priority, 1 October 2026

Native order S4 pilot: 100% selected development accuracy. Native language2K:
3.764712 bpc / 3.778244 whole CPU fit GFLOPs / 54,907 parameters. Against the
saved KV2K construction: 6.02x less fit work at .032126 bpc worse, with differing
width/capacity/history construction. No frontier/energy claim.

Full-depth R0/R2/R4/uniform/late/waiting contracts passed. R2 and R4 accounting
smokes completed; R4-late follows. Guarded tabular classification/regression and
both tree-control smokes passed; full pilot results remain AWS work.

The first recovery waited safely through a second rebase, published R4 smoke,
then stopped on changed run_safe.sh fingerprint. This was the explicitly
approved AWS slot extension, not a model/source/optimizer change. Reviewed the
diff: local default still uses the ordinary host lock and identical caps;
AWS slot environment is absent locally. Preserve prior needs_review status/logs.

New manifest:
`experiments/queue/local_delay_feature_guard_recovery_20261001T215000Z.json`.
Run `scripts/run_delay_feature_recovery.py --manifest-stem local_delay_feature_guard_recovery_20261001T215000Z`
once in same-stem tmux. It revalidates/skips unchanged successful queues, finishes
R4-late smoke, fits matched R2/R4/R4-late 2K, refits the winner's waiting control,
uses declared gates for delay8K and the separate near-quality/work native8K test,
then hands to `local_native_research_guard_recovery_20261001T215000Z`.

The local native source64/payload16 extension is deferred rather than launched
under an unmeasured unsafe envelope: AWS's smaller payload8/source64 contract
already exceeded the local RSS cap (measured 2,585,864 KiB). Its original queue
and manifest remain unchanged; prioritize the existing AWS source64 pilot and
measure/provision before the wider extension. Source16, controls/replications and
all original quality gates remain in the bounded local continuation.

Local caps: VMS4,000,000 KiB, groupRSS2,500,000 KiB, minavailable8192MiB;
CPU-only ~31GiB host, idleavailable ~12GiB. One trainer. Read status/processes
before assuming later stages completed. Model/driver/helper/queue fingerprints
are unchanged; new manifests explicitly accept only the reviewed guard revision.

At 21:57 UTC the new continuation is healthy: R4-late accounting smoke was
published in 4663d51, and the full 2K/R2 reception pilot has been training since
21:51 UTC. Group RSS is about 447MiB, MemAvailable about 11.4GiB. No completed
quality score from that pilot yet. AWS recovery pilot results are now included
by the report loader across both original/recovery plans, deduplicated by tag;
contracts and smokes remain excluded from benchmark plots.

The design update in `theory/RESULTS_DESIGN_UPDATE_20261001.md` records useful
depth versus weak tested residual paths, the native 6.02x work/.032126bpc
tradeoff, proved but not yet fitted temporal functions, and training-memory
exposure. These are the current empirical constraints, not assumed supremacy.

REPORT/PDF now contains new AWS language and complete residual4 hierarchy results;
63-page PDF bounds checked. Plain depth3 85.06% remains best saved64K RHM point,
residual4 72.56%/ten passes still improving. Early AWS matrix pilots are to be
reported separately from 8,191-target language protocols. Numerical primitives
for trainable windows/trains are proved/tested, not a fitted full-model claim.
Robotics adapters remain pending; the task is supervised asynchronous streams,
not RL. Core gaps: observed fixed addresses, forced activity, local surrogate
loser credit, bounded ordinary credit and no validated physical clockless ASIC.
