#!/bin/bash
# Simple wrapper script for automated news collection

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run collection with uv
uv run python -m ai_news.cli collect

echo "News collection completed at $(date)"