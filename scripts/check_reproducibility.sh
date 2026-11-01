#!/usr/bin/env bash
# Reproducibility check (docs/PLAN.md, M3 and Verification 5-6).
#
# Rebuilds the submission from a clean copy of the repository and checks that:
#   1. the copy holds no gated data, caches, private ground truth or submissions;
#   2. `predict` gives byte-identical files run twice in the copy (cold cache, then warm);
#   3. the file passes the organisers' validator;
#   4. it is byte-identical to the file this checkout produces;
#   5. retraining the control model from the public Metrica data gives the shipped
#      models/control_v1.json byte for byte (the raw files come from data/other_matches/, linked
#      like the rest of data/, so they are downloaded at most once and never deleted);
#   6. the test suite passes in the copy, including the tests that need the Metrica data.
#
#   scripts/check_reproducibility.sh           # copy of what git would commit (works before a commit)
#   scripts/check_reproducibility.sh --head    # a real `git clone` of HEAD (after committing)
#
# The data bundle is linked from this checkout's data/ (it is never in git). This checkout's files
# are not modified apart from its own .cache/; the copy lives in a temporary folder that is
# removed at the end (KEEP=1 keeps it). EXPECTED_SHA256=<hash> also checks the file against a
# recorded hash.
set -euo pipefail
unset VIRTUAL_ENV            # the copy uses its own .venv

repo=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
mode=${1:-worktree}
work=$(mktemp -d "${TMPDIR:-/tmp}/mtah-repro.XXXXXX")
trap '[[ "${KEEP:-0}" == 1 ]] && echo "kept $work" || rm -rf "$work"' EXIT
clone="$work/repo"
step() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

step "clean copy ($mode) in $clone"
if [[ "$mode" == "--head" ]]; then
    git clone --quiet --no-local "$repo" "$clone"
    files=$(git -C "$clone" ls-files)
else
    mkdir -p "$clone"
    files=$(git -C "$repo" ls-files --cached --others --exclude-standard)
    (cd "$repo" && git ls-files -z --cached --others --exclude-standard \
        | tar --null --files-from=- -cf -) | tar -xf - -C "$clone"
fi
echo "$(wc -l <<<"$files") files"

step "leak check"
leaks=$(grep -E '^data/|^\.cache/|^evaluation/results/.*\.csv$|_possession_gt(_meta|_reference)?\.(csv|json)$|\.(npz|mp4|pkl|pdf)$' <<<"$files" \
    | grep -vE '^data/README\.md$|^evaluation/results/baseline_[a-z0-9]+\.csv$' || true)
[[ -z "$leaks" ]] || fail "files that must never be published:"$'\n'"$leaks"
big=$(cd "$clone" && find . -type f -size +5M -not -path './.git/*' || true)
[[ -z "$big" ]] || fail "unexpectedly large files:"$'\n'"$big"
echo "ok: no data, caches, private ground truth, submissions or files over 5 MB"

step "link the data bundle"
for item in "$repo"/data/*; do
    name=$(basename "$item")
    [[ -e "$clone/data/$name" ]] || ln -s "$item" "$clone/data/$name"
done
ls "$clone/data"

cd "$clone"
step "uv sync (from uv.lock)"
uv sync --frozen --quiet

step "predict twice: cold cache, then warm"
uv run --frozen more-than-a-hack-2027 predict --out "$work/cold.csv"
uv run --frozen more-than-a-hack-2027 predict --out "$work/warm.csv"
cold=$(sha256sum "$work/cold.csv" | cut -d' ' -f1)
warm=$(sha256sum "$work/warm.csv" | cut -d' ' -f1)
echo "cold $cold"
echo "warm $warm"
[[ "$cold" == "$warm" ]] || fail "cold and warm runs differ"

step "validate"
uv run --frozen python evaluation/validate_submission.py "$work/cold.csv"

step "compare with this checkout"
(cd "$repo" && uv run --frozen more-than-a-hack-2027 predict --out "$work/here.csv" >/dev/null)
here=$(sha256sum "$work/here.csv" | cut -d' ' -f1)
echo "here $here"
[[ "$here" == "$cold" ]] || fail "the clean copy and this checkout give different files"
if [[ -n "${EXPECTED_SHA256:-}" ]]; then
    [[ "$cold" == "$EXPECTED_SHA256" ]] || fail "expected $EXPECTED_SHA256"
    echo "matches EXPECTED_SHA256"
fi

step "retrain the control model from the public Metrica data"
uv run --frozen more-than-a-hack-2027 train-control --out "$work/control.json" | tail -1
cmp "$work/control.json" src/more_than_a_hack_2027/models/control_v1.json \
    || fail "retrained control model differs from the shipped one"
echo "ok: identical to models/control_v1.json"

step "tests in the clean copy"
uv run --frozen pytest -q

step "REPRODUCIBLE: sha256 $cold"
