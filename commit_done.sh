#!/usr/bin/env bash
# Stage the completed Sleeping Machines architecture, theory, and report work,
# then commit it from the host checkout (outside the Codex/Docker sandbox).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "Refusing to commit: this checkout has no current branch." >&2
  exit 2
fi
if [[ "$BRANCH" != "main" && "${ALLOW_NON_MAIN_BRANCH:-0}" != "1" ]]; then
  echo "Refusing to commit on '$BRANCH'; set ALLOW_NON_MAIN_BRANCH=1 to opt in." >&2
  exit 2
fi

# Deliberate allowlist: exclude logs, checkpoints, and caches. Include the
# E83/E84 and E114 result JSON summaries so completed pilots are ready for a
# host-side commit.
FILES=(
  AGENTS.md
  REPORT.md
  AWS_EXPERIMENT_INSTANCE.md
  AWS_EXPERIMENT_RUNBOOK.md
  Dockerfile
  codex.sh
  dev.sh
  scripts/resolve-claude-version.sh
  scripts/bootstrap_aws_experiments.sh
  experiments/FINDINGS.md
  experiments/MATHEMATICAL_PROGRAM.md
  experiments/ROADMAP.md
  experiments/THEORY.md
  experiments/theory
  experiments/e64_lm_baselines.py
  experiments/e68_race_transformer.py
  experiments/e74_time_vector_net.py
  experiments/e75_equivariant_tvn.py
  experiments/e77_tv_lm.py
  experiments/e83_deep_shd.py
  experiments/e83_countmark_summary.py
  experiments/queue/e83_shd_countmark_control_20260929.txt
  experiments/queue/e83_shd_countmark_additive_20260929.txt
  experiments/queue/e83_shd_countmark_address_neutral_20260929.txt
  experiments/queue/e83_countmark_summary_20260929.txt
  experiments/queue/report_shd_countmark_20260929.txt
  report/figures/e83_countmark_coupling.png
  experiments/e83_route_dynamics_audit.py
  experiments/e83_route_option_value_audit.py
  experiments/e83_route_cost_audit.py
  experiments/e83_route_pair_occupancy_audit.py
  experiments/e83_spike_boundary_audit.py
  experiments/e83_emission_contract_audit.py
  experiments/queue/e83_emission_contract_audit.txt
  experiments/queue/e83_emission_grid.txt
  experiments/e116_optionality_contract.py
  experiments/queue/e116_optionality_contract.txt
  experiments/queue/e116_optionality_contract_v2.txt
  experiments/results/e116/optionality_contract.json
  experiments/results/e116/optionality_contract_v2.json
  experiments/queue/e83_spike_boundary_l4_causal_1024.txt
  experiments/e83_spike_pair_audit.py
  experiments/e83_conditioned_margin_audit.py
  experiments/e83_counterfactual_audit.py
  experiments/plot_e83_firing.py
  experiments/sparse_anytime_readout.py
  experiments/e84_deep_market.py
  experiments/e114_attention_bound.py
  experiments/queue/e83_e84_depth.txt
  experiments/queue/e83_objective_controls_2ep.txt
  experiments/queue/e83_race_stable_2ep.txt
  experiments/queue/e83_counterfactual_readout_audit.txt
  experiments/queue/e83_event_prefix_cf_smoke.txt
  experiments/queue/e83_event_prefix_d4_pilot.txt
  experiments/queue/e83_event_prefix_cf_local_smoke.txt
  experiments/queue/e83_event_prefix_d4_cf_local.txt
  experiments/queue/e83_deep_readout_fusion.txt
  experiments/queue/e83_count_mark_ablation.txt
  experiments/queue/e83_d4_readout_validation.txt
  experiments/queue/e83_d4_early_skip.txt
  experiments/queue/e83_d4_all_depths_replication.txt
  experiments/queue/e83_data_budget_equal_updates.txt
  experiments/queue/e83_spike_boundary_audit.txt
  experiments/queue/e83_spike_boundary_late_audit.txt
  experiments/queue/e83_spike_pair_audit.txt
  experiments/queue/e83_spike_pair_depth_audit.txt
  experiments/queue/e83_spike_pair_l23_1024.txt
  experiments/queue/e83_spike_pair_shared_receiver_1024.txt
  experiments/queue/e83_spike_boundary_l3_1024.txt
  experiments/queue/e83_spike_pair_l23_1024.txt
  experiments/queue/e83_route_bundle_pair.txt
  experiments/queue/e83_route_bundle_pair_layer_balanced.txt
  experiments/queue/e83_route_bundle_pair_late_balanced.txt
  experiments/queue/e83_route_cost_audit.txt
  experiments/queue/e83_route_pair_occupancy_audit.txt
  experiments/queue/e83_moe_top2_route_swap.txt
  experiments/queue/e83_route_dynamics_audit.txt
  experiments/queue/e83_route_option_value_audit.txt
  experiments/queue/e83_spike_option_training.txt
  experiments/queue/e83_optionality_matched_6ep.txt
  experiments/queue/e83_conditioned_margin_audit.txt
  experiments/queue/e77_route_cf_smoke.txt
  experiments/queue/e77_route_cf_bootstrap.txt
  experiments/queue/e77_depth4_bootstrap_compare.txt
  experiments/queue/e77_default_width_bootstrap_smoke.txt
  experiments/queue/e77_deep_lm_benchmark.txt
  experiments/queue/e77_depth8_lm_100k.txt
  experiments/queue/e71a.txt
  experiments/queue/chain_0928.sh
  experiments/queue/e64b.txt
  experiments/queue/e71b.txt
  experiments/queue/run_safe.sh
  experiments/results/e64/lstm_D1000000_s256_p20_dr0.2_v.json
  experiments/results/e64/lstm_D10000000_s512_p6_dr0.1_v.json
  experiments/results/e64/tf_D10000000_s256_L4_p4_dr0.1_v_checkpoint.json
  experiments/results/e64/tf_D1000000_s256_p20_dr0.2_v.json
  experiments/results/e83/*.json
  experiments/results/e83/e83_final_layer_activity.png
  experiments/results/e77/tvlm_D4096_p0.25_r1_M8-8-8_depth4_c0_eh1_ek0_s77.json
  experiments/results/e77/tvlm_D4096_p0.25_r1_M8-8-8_depth4_c0_eh1_ek0_s77_cf4_b0.5_sg0.25_w1_dl5_lr0_gc1_boot0.1_cb4.json
  experiments/results/e77/tvlm_D4096_p0.25_r1_M8-8-8_depth4_c0_eh1_ek0_s77_cf4_b0.5_sg0.25_w1_dl5_lr0_gc1_boot0.1_cb4_cs6.json
  experiments/results/e77/tvlm_D4096_p0.25_r1_M8-8-8_depth8_c0_eh1_ek0_s77_boot0.1_cb4_cs6.json
  experiments/results/e77/tvlm_D10000_p0.5_r1_M128-128-64_depth4_c0_eh1_ek0_s77_boot0.1_cb4.json
  experiments/results/e77/tvlm_D10000_p0.5_r1_M128-128-64_depth4_c0_eh1_ek0_s77_boot0.1_cb4_cs6.json
  experiments/results/e64/tf_D1000000_s112_L8_p5_b4_dr0_v.json
  experiments/results/e64/tf_D100000_s112_L8_p1_b2_dr0_v.json
  experiments/results/e77/tvlm_D100000_p1_r1_M128-128-64_depth8_c0_eh1_ek0_s0_boot0.1_cb4_cs6.json
  experiments/results/e84
  experiments/results/e114
  experiments/results/e79/race_mixer_D1000000_K5_e77none.json
  experiments/results/e79/race_mixer_D10000000_K6_e77none.json
  report/make_pdf.py
  report/frontier_potential.py
  report/figures/e83_emission_contract.png
  report/figures/optionality_contract.png
  report/figures/potential_evidence.png
  report/figures/e83_route_gradient_diagnostics.png
  report/figures/e83_d4_readout_support.png
  report/figures/e83_equal_update_data_budget.png
  report/figures/e83_route_reachability.png
  report/figures/e83_route_bundle_pair.png
  report/figures/e83_route_cost_audit.png
  report/figures/e83_route_option_value.png
  report/figures/e83_spike_option_training.png
  report/figures/e83_route_pair_occupancy.png
  report/figures/e83_spike_boundary_late.png
  report/figures/e83_spike_pair_audit.png
  report/figures/e83_optionality_state_value.png
  report/figures/e83_conditioned_margin_support.png
  report/figures/e77_depth_trainability_bootstrap.png
  report/sleeping_machines_status.pdf
  commit_done.sh
)

# Expand allowlisted globs to their matching files, and retain tracked
# deletions. Skip absent literal paths and unmatched globs so git add never
# fails with a missing-pathspec error.
STAGE_FILES=()
SKIPPED_FILES=()
for file in "${FILES[@]}"; do
  if [[ "$file" == *'*'* || "$file" == *'?'* || "$file" == *'['* ]]; then
    found=0
    while IFS= read -r match; do
      [[ -e "$match" || -L "$match" ]] || continue
      STAGE_FILES+=("$match")
      found=1
    done < <(compgen -G "$file" || true)
    while IFS= read -r -d '' match; do
      STAGE_FILES+=("$match")
      found=1
    done < <(git ls-files -z -- "$file")
    if (( ! found )); then
      SKIPPED_FILES+=("$file")
    fi
  elif [[ -e "$file" || -L "$file" ]] || git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
    STAGE_FILES+=("$file")
  else
    SKIPPED_FILES+=("$file")
  fi
done
if ((${#SKIPPED_FILES[@]})); then
  echo "Skipping allowlisted paths absent from this checkout:" >&2
  printf '  %s\n' "${SKIPPED_FILES[@]}" >&2
fi
if ((${#STAGE_FILES[@]} == 0)); then
  echo "No allowlisted paths exist in this checkout."
  exit 0
fi

# A previous run may have staged this allowlist and stopped at its whitespace
# check. Permit that exact partial state, but protect any unrelated staged work.
while IFS= read -r -d '' staged; do
  allowed=0
  for file in "${FILES[@]}"; do
    if [[ "$staged" == "$file" || "$staged" == $file || ( "$file" == "experiments/theory" && "$staged" == "$file/"* ) ]]; then
      allowed=1
      break
    fi
  done
  if (( ! allowed )); then
    echo "Refusing to commit: unrelated staged path: $staged" >&2
    echo "Commit or unstage it first, then run this script again." >&2
    exit 2
  fi
done < <(git diff --cached --name-only -z)

git add -A -- "${STAGE_FILES[@]}"

if git diff --cached --quiet; then
  echo "No allowlisted changes to commit."
  exit 0
fi

# PDF structure commonly uses trailing spaces; check source and prose only.
CHECK_FILES=()
for file in "${STAGE_FILES[@]}"; do
  [[ "$file" == "report/sleeping_machines_status.pdf" ]] || CHECK_FILES+=("$file")
done
git diff --cached --check -- "${CHECK_FILES[@]}"
git diff --cached --stat

MESSAGE="${1:-Advance event semantics and optionality theory; reorganize frontier report}"
git commit -m "$MESSAGE"
