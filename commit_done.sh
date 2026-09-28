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
  Dockerfile
  codex.sh
  dev.sh
  scripts/resolve-claude-version.sh
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
  experiments/results/e64/tf_D1000000_s256_p20_dr0.2_v.json
  report/make_pdf.py
  report/sleeping_machines_status.pdf
  commit_done.sh
)

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

git add -- "${FILES[@]}"

if git diff --cached --quiet; then
  echo "No allowlisted changes to commit."
  exit 0
fi

# PDF structure commonly uses trailing spaces; check source and prose only.
CHECK_FILES=()
for file in "${FILES[@]}"; do
  [[ "$file" == "report/sleeping_machines_status.pdf" ]] || CHECK_FILES+=("$file")
done
git diff --cached --check -- "${CHECK_FILES[@]}"
git diff --cached --stat

MESSAGE="${1:-Advance E77 theory, report, and safe experiment setup}"
git commit -m "$MESSAGE"
