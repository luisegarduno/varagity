#!/bin/bash

# create_worktree.sh - Create a new worktree for development work
# If no name provided, generates a unique human-readable one
# If no base branch provided, uses current branch
# Use --existing-branch to specify an existing branch (local or remote like origin/branch-name) with a different directory name
#
# Usage:
#   ./scripts/create_worktree.sh [--existing-branch <branch>] [worktree-name] [base-branch]

set -e  # Exit on any error

# Always operate from the repository root, regardless of where the script is invoked from
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

REMOTE=origin

# Function to generate a unique worktree name
generate_unique_name() {
    local adjectives=("swift" "bright" "clever" "smooth" "quick" "clean" "sharp" "neat" "cool" "fast")
    local nouns=("fix" "task" "work" "dev" "patch" "branch" "code" "build" "test" "run")

    local adj=${adjectives[$RANDOM % ${#adjectives[@]}]}
    local noun=${nouns[$RANDOM % ${#nouns[@]}]}
    local timestamp=$(date +%H%M)

    echo "${adj}_${noun}_${timestamp}"
}

# Parse flags
EXISTING_BRANCH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --existing-branch)
            if [ -z "$2" ]; then
                echo "❌ Error: --existing-branch requires a branch name"
                exit 1
            fi
            EXISTING_BRANCH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Get worktree name from parameter, the existing branch, or generate one
if [ -n "$1" ]; then
    WORKTREE_NAME="$1"
elif [ -n "$EXISTING_BRANCH" ]; then
    WORKTREE_NAME="${EXISTING_BRANCH#${REMOTE}/}"
else
    WORKTREE_NAME="$(generate_unique_name)"
fi

# Strip any prefix before slash for directory name (e.g., "luis/branch-name" -> "branch-name")
# Keep the full name for the git branch, but use stripped version for directory
WORKTREE_DIR_NAME="${WORKTREE_NAME##*/}"

# Get base branch from second parameter or use current branch (fall back to HEAD when detached)
CURRENT_BRANCH="$(git branch --show-current)"
BASE_BRANCH=${2:-${CURRENT_BRANCH:-HEAD}}

REPO_BASE_NAME=varagity

if [ -n "$VARAGITY_WORKTREE_OVERRIDE_BASE" ]; then
    WORKTREES_BASE="${VARAGITY_WORKTREE_OVERRIDE_BASE}/${REPO_BASE_NAME}"
else
    WORKTREES_BASE="$HOME/wt/${REPO_BASE_NAME}"
fi

WORKTREE_PATH="${WORKTREES_BASE}/${WORKTREE_DIR_NAME}"

echo "🌳 Creating worktree: ${WORKTREE_NAME}"
echo "📁 Location: ${WORKTREE_PATH}"

# Create worktrees base directory if needed
mkdir -p "$WORKTREES_BASE"

# Check if worktree already exists
if [ -d "$WORKTREE_PATH" ]; then
    echo "❌ Error: Worktree directory already exists: $WORKTREE_PATH"
    exit 1
fi

echo "🔄 Fetching latest from ${REMOTE}..."
git fetch "$REMOTE" 2>/dev/null || echo "⚠️  Could not fetch from ${REMOTE}"

# Tracks whether this script created the branch, so cleanup never deletes a pre-existing branch
CREATED_BRANCH=0

# Create worktree
if [ -n "$EXISTING_BRANCH" ]; then
    # Check if it's a remote branch (starts with origin/)
    if [[ "$EXISTING_BRANCH" == ${REMOTE}/* ]]; then
        LOCAL_BRANCH_NAME="${EXISTING_BRANCH#${REMOTE}/}"

        if git show-ref --verify --quiet "refs/heads/${LOCAL_BRANCH_NAME}"; then
            echo "📋 Using existing local branch: ${LOCAL_BRANCH_NAME}"
            git worktree add "$WORKTREE_PATH" "$LOCAL_BRANCH_NAME"
        else
            echo "🆕 Creating local tracking branch: ${LOCAL_BRANCH_NAME}"
            git worktree add --track -b "$LOCAL_BRANCH_NAME" "$WORKTREE_PATH" "$EXISTING_BRANCH"
            CREATED_BRANCH=1
        fi
        ACTUAL_BRANCH="$LOCAL_BRANCH_NAME"
    else
        echo "📋 Using specified branch: ${EXISTING_BRANCH}"
        git worktree add "$WORKTREE_PATH" "$EXISTING_BRANCH"
        ACTUAL_BRANCH="$EXISTING_BRANCH"
    fi
elif git show-ref --verify --quiet "refs/heads/${WORKTREE_NAME}"; then
    echo "📋 Using existing local branch: ${WORKTREE_NAME}"
    git worktree add "$WORKTREE_PATH" "$WORKTREE_NAME"
    ACTUAL_BRANCH="$WORKTREE_NAME"
elif git show-ref --verify --quiet "refs/remotes/${REMOTE}/${WORKTREE_NAME}"; then
    echo "🔄 Found branch on ${REMOTE}, creating local tracking branch: ${WORKTREE_NAME}"
    git worktree add --track -b "$WORKTREE_NAME" "$WORKTREE_PATH" "${REMOTE}/${WORKTREE_NAME}"
    ACTUAL_BRANCH="$WORKTREE_NAME"
    CREATED_BRANCH=1
else
    # Prefer the freshly fetched remote copy of the base branch when it exists
    if git show-ref --verify --quiet "refs/remotes/${REMOTE}/${BASE_BRANCH}"; then
        START_POINT="${REMOTE}/${BASE_BRANCH}"
    else
        START_POINT="$BASE_BRANCH"
    fi
    echo "🔀 Creating from: ${START_POINT}"
    echo "🆕 Creating new branch: ${WORKTREE_NAME}"
    git worktree add --no-track -b "$WORKTREE_NAME" "$WORKTREE_PATH" "$START_POINT"
    ACTUAL_BRANCH="$WORKTREE_NAME"
    CREATED_BRANCH=1
fi

cleanup_worktree() {
    cd "$REPO_ROOT"
    git worktree remove --force "$WORKTREE_PATH"
    if [ "$CREATED_BRANCH" -eq 1 ]; then
        git branch -D "$ACTUAL_BRANCH" 2>/dev/null || true
    fi
}

# Copy the whole .claude directory (skills, agents, commands, settings.local.json, ...).
# .claude/ is gitignored, so `git worktree add` checks out none of it. worktrees/ is skipped:
# `claude --worktree` keeps its own checkouts there, and those must not be duplicated.
if [ -d ".claude" ]; then
    echo "📋 Copying .claude/ (skills, agents, commands, settings) to worktree..."
    mkdir -p "$WORKTREE_PATH/.claude"
    find .claude -mindepth 1 -maxdepth 1 ! -name worktrees -exec cp -a {} "$WORKTREE_PATH/.claude/" \;
fi

# Copy all gitignored .env* files (root and subdirectories), preserving their relative paths.
ENV_FILES_COPIED=0
while IFS= read -r -d '' env_file; do
    env_file="${env_file#./}"
    # The tracked .env.example files are the branch's own: never overwrite them.
    git check-ignore -q -- "$env_file" || continue
    echo "📋 Copying $env_file to worktree..."
    mkdir -p "$WORKTREE_PATH/$(dirname "$env_file")"
    cp "$env_file" "$WORKTREE_PATH/$env_file"
    ENV_FILES_COPIED=$((ENV_FILES_COPIED + 1))
done < <(find . -path ./.git -prune -o -type f -name '.env*' -print0)

if [ $ENV_FILES_COPIED -eq 0 ]; then
    echo "⚠️  No .env* files found to copy"
elif [ $ENV_FILES_COPIED -eq 1 ]; then
    echo "✅ Copied 1 env file"
else
    echo "✅ Copied $ENV_FILES_COPIED env files"
fi

# Link thoughts/tasks (gitignored) to the main checkout's, so plans edited in the worktree
# are the same files main sees. thoughts/ stays a real directory: the ignore rule `thoughts/`
# matches directories only, so a symlinked thoughts would show up as untracked.
MAIN_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
if [ -d "$MAIN_ROOT/thoughts/tasks" ]; then
    echo "🔗 Linking thoughts/tasks to ${MAIN_ROOT}/thoughts/tasks..."
    mkdir -p "$WORKTREE_PATH/thoughts"
    ln -s "$MAIN_ROOT/thoughts/tasks" "$WORKTREE_PATH/thoughts/tasks"
else
    echo "⚠️  No thoughts/tasks in ${MAIN_ROOT} to link"
fi

echo "✅ Worktree created successfully!"
echo "📁 Path: ${WORKTREE_PATH}"
echo "🔀 Branch: ${ACTUAL_BRANCH}"
echo ""
echo "To work in this worktree:"
echo "  cd ${WORKTREE_PATH}"
echo ""
echo "To remove this worktree later:"
echo "  git worktree remove ${WORKTREE_PATH}"
