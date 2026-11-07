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
# A comment holds 65,536 characters and a step summary 1 MiB. A longer list is cut between
# lines inside the code fence, and says how many rows are missing.
make_body() {
    local limit=$1 shown=$lived note=
    if (($(printf '%s' "$lived" | LC_ALL=C wc -c) > limit)); then
        shown=$(printf '%s\n' "$lived" | LC_ALL=C awk -v max="$limit" '{ n += length($0) + 1; if (n > max) exit; print }')
        note="
$(($(grep -c . <<<"$lived") - $(grep -c . <<<"$shown"))) more not shown here, see the full report."
    fi
    # shellcheck disable=SC2016 # the backticks are Markdown, not command substitution
    printf '%s\n**Mutation testing:** %s of %s changes caught by a test.\n\n```\n%s\n```%s\n\nEach one is a change to the code no test noticed. Not a blocker: real gaps get their own PR. [Full report](%s)' \
        "$marker" "$caught" "$total" "$shown" "$note" "$RUN"
}
body=$(make_body "${MUTANTS_COMMENT_LIMIT:-58000}")
[[ -z $lived ]] || make_body 900000 >>"${GITHUB_STEP_SUMMARY:-/dev/null}"
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
