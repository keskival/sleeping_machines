# AWS joint event calibration, 2 October

Assigned comparison: AWS_JOINT_EVENT_CONTROLS.md, Theory §394/395.
Read current handoffs/theory before admission. CPU host has 30 GiB available,
no CUDA device and no trainer. Three independent guarded slots allowed here;
2 GiB RSS / 6 GiB virtual per job, 8 GiB host floor, one CPU thread.

Causality, target independence, time sensitivity and gradient contracts:
10 tests passed via the uniquely named guarded contracts queue. Before fits,
recompute the stateful table under CURRENT v2 source/data hashes: the saved
historical table is not assumed comparable. Two width64 dense controls receive
48/32 fitting/development episodes for one accounting smoke pass. No pilot
until both audits have full operator coverage and resource margin.

AWS driver copies the assigned dense driver without architectural changes;
adds complete inference ledger over first16 development queries, confirmation
data hash and selected-model checkpoint for subsequent frozen analyses.
Checkpoint is not an optimizer-resume checkpoint. Training arithmetic remains
a representative-window estimate, including Adam and clipping, scaled by
observed fitting events. Not measured energy or exact full-job arithmetic.
All passes/research jobs remain visible. Controls do not replace native races.

Planned base fits after prerequisites: GRU/Transformer width64, seeds6/7/8,
512 fit,256 dev,8 passes,16-episode Adam windows,lr.003,clip1; seeds1301/2301/3301.
Selection by lowest development NLL; fixed1024 confirmation queries. Calibration
requires >=20 percentage points over matching table accuracy. No history ladder
until completed calibration passes. A failed calibration is preserved and
blocks automatic scaling; no native advantage claim from failed controls.

Initial smokes stopped before fitting: operation audit lacked unsafe_split,
linspace and triu formulas. Failed records preserved. AWS-local audit adds
split view zero arithmetic, triangular mask comparisons and conservative
2N+2 grid-generation arithmetic; shared audit/model files stay unchanged.
New immutable recovery plan/tags required. Current v2 table confirmation is
58.3984% (NLL .741589), so base calibration requires >=78.3984% accuracy.
