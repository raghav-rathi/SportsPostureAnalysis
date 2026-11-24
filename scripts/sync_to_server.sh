#!/usr/bin/env bash
set -euo pipefail

# Sync local source code to a remote CUDA GPU server.
# Configure REMOTE_HOST and REMOTE_DIR via environment variables or a local gitignored .env file.
# Example: REMOTE_HOST='user@gpu-host' REMOTE_DIR='/path/to/workspace' ./scripts/sync_to_server.sh
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$LOCAL_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$LOCAL_DIR/.env"
fi

: "${REMOTE_HOST:?Set REMOTE_HOST (e.g., user@gpu-server) in environment or .env}"
: "${REMOTE_DIR:?Set REMOTE_DIR (e.g., /data/SportsPostureAnalysis) in environment or .env}"

RSYNC_ARGS=(
  -avz
  --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache'
  --exclude '.venv' --exclude 'cache' --exclude 'models' --exclude 'data' --exclude 'outputs' --exclude '.env'
  --exclude 'web/node_modules' --exclude 'web/dist' --exclude '*.tsbuildinfo' --exclude '*.db*' --exclude '*.sqlite*'
)

if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  sshpass -e rsync "${RSYNC_ARGS[@]}" "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
else
  rsync "${RSYNC_ARGS[@]}" "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
fi
