# Three distinct replay sites: independent FIT confirmation

Known failures: trained ordinal and learned k2-with-replacement priority gates
fail. Followup exploration of old heldout32..63 suggests a distinct hypothesis:
three weighted sites WITHOUT replacement may reduce conditional variance while
using six rather than eight shadow lanes. These old observations are selection
evidence only. Fix fresh critic-heldout FIT indices64..95, seeds7/8. Producers
previously saw all FIT labels; no generalization or quality claim from this.

Reuse frozen64-tree magnitude predictor trained ONLY on FIT0..31. Do not refit.
Same90%predictedmagnitudes/10%uniformproposal. Sequential weighted selection
without replacement (Plackett-Luce ordered draw law), k3. Enumerate all legal
orders for R<=32/k<=3 to obtain marginal pi_i and pair pi_ij. The proper
Horvitz-Thompson correction is sum_selected g_i/pi_i, not g_i/(k*p_i).
Positive proposal guarantees support. Production admission cost includes tree
evaluation and inclusion enumeration; record timing. Exact enumeration limits
are explicit, not scalable fine-84-site inference claims.

Check exhaustive three-site mean/shared-vector variance and uniform inclusion
nesting before new replays. Conditional score variance sum_i(1/pi_i-1)||g_i||²;
shared-parameter variance sum_ij(pi_ij/(pi_i*pi_j)-1)<g_i,g_j>. Evaluate all
producer parameter coordinates on fresh indices64/65, both seeds. Combined
nomination requires scoreaggregate<=uniformk4 AND every parametercase<=uniformk4
in both seeds. Uniformk3 control mandatory; no retuning k or features after
confirmation. Baseline uniformk4 WITHOUTreplacement consistent with current
teacher. No model/critic optimizer or DEV/test quality. Full diagnostic actually
enumerates40replay lanes/prefix; six-lane proposed cost is NOT its measured cost.

All timing, hard routes, legal first-time alternative writes, key/value paths
unchanged. Discovery/proposal/inclusion work and cached producer/tree work paid.
If nomination passes, integrated update/recovery/work contracts and matched
quality pilots still required, and main phase0 gate remains independently owned.
Unique guardedCPUqueue,2GiB RSS/6GiBvirtual,8GiBfloor,600s,one thread.
