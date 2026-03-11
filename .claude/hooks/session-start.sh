#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies via Poetry
poetry install --no-interaction

# Install frontend dependencies
cd "$CLAUDE_PROJECT_DIR/app/frontend"
npm install
