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

# Deliberate allowlist: exclude experiment logs, checkpoints, caches, and result
# files that have not been reviewed for the report.
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
  experiments/e64_lm_baselines.py
  experiments/e68_race_transformer.py
  experiments/e74_time_vector_net.py
  experiments/e75_equivariant_tvn.py
  experiments/e77_tv_lm.py
  experiments/queue/chain_0928.sh
  experiments/queue/e64b.txt
  experiments/queue/e71b.txt
  experiments/queue/run_safe.sh
  experiments/results/e64/lstm_D1000000_s256_p20_dr0.2_v.json
  experiments/results/e64/lstm_D10000000_s512_p6_dr0.1_v.json
  experiments/results/e64/tf_D1000000_s256_p20_dr0.2_v.json
  experiments/results/e79/race_mixer_D1000000_K5_e77none.json
  experiments/results/e79/race_mixer_D10000000_K6_e77none.json
  report/make_pdf.py
  report/figures/potential_evidence.png
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
    if [[ "$staged" == "$file" ]]; then
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
