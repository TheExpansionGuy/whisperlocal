#!/bin/bash
# Pulls latest from GitHub and updates the running app if anything changed.
set -e

cd "$(dirname "$0")"

git fetch origin main --quiet

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" != "$REMOTE" ]; then
  echo "$(date): New update found, applying..."
  git pull origin main --quiet
  ./update.sh
else
  echo "$(date): Already up to date."
fi
