#!/usr/bin/env bash
# Compare with the last verified release tag so cancelled builds cannot hide pending changes.
# Forks may not have a GHCR CUDA image or upstream's private build-cache credentials.
# Manual dispatch forces all scopes. Automatic runs classify paths without starting a compiler.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
force() {
  printf '%s\n' "$1" >&2
  python3.11 "$script_dir/change-scopes.py" --force </dev/null
  exit 0
}
[[ "${GITHUB_EVENT_NAME:-}" != workflow_dispatch ]] || force 'Manual dispatch: all checks and builds requested.'
base=''
case "${GITHUB_EVENT_NAME:-}:${GITHUB_REF:-}" in
  pull_request:*)
    base="$(git rev-parse --verify --quiet 'HEAD^1')" || base=''
    ;;
  push:refs/heads/master)
    # The latest published *tag* is an immutable source identity. release.target_commitish
    # commonly contains "master", not a commit, and private GHCR images are unavailable
    # in forks. Annotated tags need the dereferenced commit.
    tag="$(gh api "repos/$GITHUB_REPOSITORY/releases/latest" --jq .tag_name 2>/dev/null)" || tag=''
    if [[ "$tag" =~ ^[a-zA-Z0-9._-]+$ ]]; then
      base="$(git ls-remote origin "refs/tags/$tag^{}" | cut -f1)" || base=''
      if [[ ! "$base" =~ ^[0-9a-f]{40}$ ]]; then
        base="$(git ls-remote origin "refs/tags/$tag" | cut -f1)" || base=''
      fi
    fi
    ;;
  push:*)
    base="$(git rev-parse --verify --quiet 'HEAD^1')" || base=''
    ;;
esac
if [[ ! "$base" =~ ^[0-9a-f]{40}$ ]]; then
  force 'No verified comparison tag available: bootstrap all build/check scopes.'
fi
if ! git cat-file -e "$base^{commit}" 2>/dev/null; then
  if ! git fetch --quiet --no-tags --depth=1 origin "$base"; then
    force 'Published baseline cannot be fetched: run all scopes rather than silently skipping.'
  fi
fi
printf 'Comparing %s with HEAD\n' "$base" >&2
git diff --name-only --no-renames -z "$base" HEAD | python3.11 "$script_dir/change-scopes.py"
