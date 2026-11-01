#!/usr/bin/env bash
# GeoPredict pipeline execution script
# Loads environment, activates virtualenv, runs the pipeline, and logs output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Ensure logs directory exists
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Activate virtual environment (adjust if needed)
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "Virtualenv not found at $REPO_ROOT/.venv, proceeding without activation"
fi

# Timestamp for log file
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_${TS}.log"

# Run the pipeline (add any desired CLI flags here)
python "$REPO_ROOT/main.py" "$@" 2>&1 | tee "$LOG_FILE"
