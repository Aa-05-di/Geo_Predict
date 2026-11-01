# Scheduling GeoPredict Pipeline

## Overview
This document explains how to run the **GeoPredict** scraping, sentiment analysis, and alert pipeline at regular intervals (e.g., hourly or daily) without manual intervention.

## 1. Provided helper script
A Bash wrapper has been added to the repository:

- **`run_pipeline.sh`** – Activates the virtual environment, ensures a `logs/` folder exists, and runs `main.py` while streaming stdout/stderr to a timestamped log file.

```bash
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
```

Make sure the script is executable:
```bash
chmod +x run_pipeline.sh
```

## 2. Adding a cron job
The simplest way to schedule regular execution on a Linux machine is via **cron**.

### Example: Hourly execution
```cron
0 * * * * /home/blackeye/Projects/Geo_Predict/run_pipeline.sh >> /home/blackeye/Projects/Geo_Predict/logs/cron.log 2>&1
```
- `0 * * * *` – Fires at minute 0 of every hour.
- The command runs the wrapper script and appends its output to `logs/cron.log` for a quick overview. Detailed per‑run logs are stored under `logs/pipeline_<timestamp>.log` by the script itself.

### Example: Daily execution at 03:30 AM
```cron
30 3 * * * /home/blackeye/Projects/Geo_Predict/run_pipeline.sh >> /home/blackeye/Projects/Geo_Predict/logs/cron.log 2>&1
```

### Installing the cron entry
1. Open your user’s crontab editor:
   ```bash
   crontab -e
   ```
2. Append one of the lines above (or customize the schedule).
3. Save and exit. Cron will automatically pick up the new schedule.

## 3. Verifying the schedule
- **Immediate test**: Run the script manually to ensure it works:
  ```bash
  ./run_pipeline.sh --dry-run
  ```
- **Cron log**: After the first scheduled run, check:
  ```bash
  tail -n 20 /home/blackeye/Projects/Geo_Predict/logs/cron.log
  ```
- **Per‑run logs**: Individual logs are saved as `logs/pipeline_YYYYMMDD_HHMMSS.log`. They contain the full console output of each pipeline execution.

## 4. Adjusting the interval
Edit the cron expression in your crontab:
- `*/15 * * * *` – every 15 minutes
- `0 9 * * 1-5` – weekdays at 09:00 AM
- See `man 5 crontab` for the full syntax.

## 5. Disabling the schedule
To temporarily stop the pipeline, comment out the line in `crontab -e` by prefixing it with `#`.

---
### TL;DR
```bash
chmod +x run_pipeline.sh   # once
crontab -e                 # add e.g. "0 * * * * /home/blackeye/Projects/Geo_Predict/run_pipeline.sh >> /home/blackeye/Projects/Geo_Predict/logs/cron.log 2>&1"
```
Your pipeline will now run automatically at the chosen interval.
