#!/usr/bin/env bash
# Stage the completed Sleeping Machines architecture, theory, and report work,
# then commit it from the host checkout (outside the Codex/Docker sandbox).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Refusing to commit: this checkout is not on main." >&2
  exit 2
fi

# Deliberate allowlist: exclude logs, checkpoints, and caches. Include the small
# E83/E84 and E114 JSON summaries so completed pilots are ready for host-side commit.
FILES=(
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
  experiments/THEORY.md
  experiments/theory
  experiments/e64_lm_baselines.py
  experiments/e68_race_transformer.py
  experiments/e74_time_vector_net.py
  experiments/e75_equivariant_tvn.py
  experiments/e77_tv_lm.py
  experiments/e83_deep_shd.py
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
  experiments/queue/e77_route_cf_smoke.txt
  experiments/queue/e77_route_cf_bootstrap.txt
  experiments/queue/e77_depth4_bootstrap_compare.txt
  experiments/queue/e77_default_width_bootstrap_smoke.txt
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
  experiments/results/e84
  experiments/results/e114
  experiments/results/e79/race_mixer_D1000000_K5_e77none.json
  experiments/results/e79/race_mixer_D10000000_K6_e77none.json
  report/make_pdf.py
  report/figures/potential_evidence.png
  report/figures/e83_route_gradient_diagnostics.png
  report/figures/e77_depth_trainability_bootstrap.png
  report/sleeping_machines_status.pdf
  commit_done.sh
)

# The allowlist can span experiment artifacts that do not exist in every
# checkout. Stage existing paths and tracked deletions; skip absent paths that
# have never been tracked so git add cannot fail with a missing-pathspec error.
STAGE_FILES=()
SKIPPED_FILES=()
for file in "${FILES[@]}"; do
  if [[ -e "$file" || -L "$file" ]] || git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
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

MESSAGE="${1:-Update frontier evidence report and AWS experiment setup}"
git commit -m "$MESSAGE"
