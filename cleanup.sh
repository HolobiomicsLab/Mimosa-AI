#!/bin/bash
# Reset Mimosa runtime state without touching named campaign folders.
#
#   ./cleanup.sh               Delete top-level generation folders in
#                              sources/workflows/ and empty tmp/.
#   ./cleanup.sh <subfolder>   Empty sources/workflows/<subfolder>/ only
#                              (for example a finished campaign, run_1).
#   -n, --dry-run              Print what would be deleted; delete nothing.
#   -y, --yes                  Do not ask before emptying a subfolder.
#
# The default never deletes named subfolders (run_1, asb_iterative_1, ...),
# their .policy_bt/ tournaments, or top-level logs such as qd_archive.jsonl.

set -euo pipefail
cd "$(dirname "$0")"

WORKFLOWS_DIR="sources/workflows"
SCRATCH_DIR="tmp"
# Generation folders: YYYYMMDD_HHMMSS_<id>, single-agent runs, failed slots.
GENERATION_GLOBS=(
    "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_*"
    "single_agent_*"
    "generation_failed*"
)
DRY_RUN=0
ASSUME_YES=0
SUBFOLDER=""

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

remove() {
    # Delete one path, or only print it on a dry run.
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would delete: $1"
    else
        rm -rf -- "$1"
        echo "deleted: $1"
    fi
}

clean_generation_folders() {
    # Delete top-level generation folders of one workflow directory.
    local dir="$1" pattern path
    [ -d "$dir" ] || { echo "Skipping missing directory: $dir"; return; }
    shopt -s nullglob
    for pattern in "${GENERATION_GLOBS[@]}"; do
        for path in "$dir"/$pattern; do
            if [ -d "$path" ]; then remove "$path"; fi
        done
    done
    shopt -u nullglob
}

empty_directory() {
    # Delete everything inside a directory, keeping the directory.
    local dir="$1" path
    [ -d "$dir" ] || { echo "Skipping missing directory: $dir"; return; }
    shopt -s nullglob dotglob
    for path in "$dir"/*; do
        remove "$path"
    done
    shopt -u nullglob dotglob
}

confirm_subfolder() {
    # Ask once before emptying a whole subfolder.
    local target="$1" answer
    if [ "$ASSUME_YES" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    if [ ! -t 0 ]; then
        echo "Refusing to empty $target without a terminal; pass --yes." >&2
        exit 1
    fi
    read -r -p "Empty $target ($(ls -A "$target" | wc -l | tr -d ' ') entries)? [y/N] " answer
    [ "$answer" = "y" ] || [ "$answer" = "Y" ] || { echo "Cancelled."; exit 0; }
}

clean_subfolder() {
    # Empty one named subfolder of the workflow directory.
    local name="$1" target
    case "$name" in
        ""|.|..|*/*) echo "Give a direct subfolder name of $WORKFLOWS_DIR, e.g. run_1." >&2; exit 1 ;;
    esac
    target="$WORKFLOWS_DIR/$name"
    [ -d "$target" ] || { echo "Not a directory: $target" >&2; exit 1; }
    confirm_subfolder "$target"
    empty_directory "$target"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) [ -z "$SUBFOLDER" ] || { echo "Give one subfolder only." >&2; exit 1; }
           SUBFOLDER="$1" ;;
    esac
    shift
done

if [ -n "$SUBFOLDER" ]; then
    clean_subfolder "$SUBFOLDER"
else
    clean_generation_folders "$WORKFLOWS_DIR"
    empty_directory "$SCRATCH_DIR"
fi
echo "Cleanup completed."
