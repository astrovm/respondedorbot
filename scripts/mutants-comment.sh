#!/usr/bin/env bash
# Keep one comment on the pull request listing the mutants no test caught
# (what scripts/mutants.sh found), and delete it once every one is caught.
# Usage: scripts/mutants-comment.sh RESULTS
# Needs GH_TOKEN, GITHUB_REPOSITORY, PR and RUN; FORK=true only summarizes.
set -euo pipefail
results=${1:?Usage: scripts/mutants-comment.sh RESULTS}
marker='<!-- mutants -->'
lived=$(cat "$results/missed.txt" "$results/timeout.txt" 2>/dev/null | sort -u || true)
# No results (nothing to mutate) counts as every mutant caught.
caught=$(grep -c . "$results/caught.txt" 2>/dev/null || true)
caught=${caught:-0}
total=$((caught + $(grep -c . <<<"$lived" || true)))
body="$marker
**Mutation testing:** $caught of $total changes caught by a test.

\`\`\`
$lived
\`\`\`

Each one is a change to the code no test noticed. Not a blocker: real gaps get their own PR. [Full report]($RUN)"
[[ -z $lived ]] || printf '%s\n' "$body" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
[[ ${FORK:-false} != true ]] || exit 0
old=$(gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR/comments" \
    --jq ".[] | select(.user.login == \"github-actions[bot]\" and (.body | startswith(\"$marker\"))) | .id" | head -n 1)
if [[ -z $lived ]]; then
    if [[ -n $old ]]; then gh api -X DELETE "repos/$GITHUB_REPOSITORY/issues/comments/$old"; fi
elif [[ -n $old ]]; then
    gh api -X PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$old" -f body="$body" >/dev/null
else
    gh api "repos/$GITHUB_REPOSITORY/issues/$PR/comments" -f body="$body" >/dev/null
fi
