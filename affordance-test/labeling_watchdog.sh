#!/bin/bash
# Watchdog for batch_final_labeling.py: restarts it if killed or crashed (e.g. the agent session itself
# ending mid-run, which is what killed a Qwen3-VL test run once already), using --resume so a restart
# picks up only the objects that don't have an output yet instead of redoing completed ones.
#
# Launch this DETACHED from the controlling shell/session, not just as an ordinary background job --
# a plain "&" is still a child of that shell's process group and dies with it. Use:
#   cd affordance-test && setsid nohup bash labeling_watchdog.sh <model_key> [object ...] \
#       </dev/null >/dev/null 2>&1 & disown
# setsid detaches it into its own session (immune to the launching shell's signals), nohup ignores
# SIGHUP, and disown removes it from the launching shell's job table -- all three together are what
# let it survive the agent session ending, not any one alone.
#
# Usage: labeling_watchdog.sh <model_key> [object ...]
# Logs to logs/watchdog_<model_key>.log. Stops on the first exit-0 (batch_final_labeling.py returns 0
# both for "finished a full pass" and for "--resume found nothing left to do" -- either way, done).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KEY="${1:?usage: labeling_watchdog.sh <model_key> [object ...]}"
shift
MAX_RETRIES=20
RETRY_DELAY_S=15
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/watchdog_${MODEL_KEY}.log"

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== watchdog start (model=$MODEL_KEY, objects=${*:-<all>}, pid=$$, sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')) ==="
for attempt in $(seq 1 "$MAX_RETRIES"); do
    log "attempt $attempt/$MAX_RETRIES: launching batch_final_labeling.py --model $MODEL_KEY --resume $*"
    "$HERE/venv_vlm/bin/python3" "$HERE/batch_final_labeling.py" --model "$MODEL_KEY" --resume "$@" >> "$LOG" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        log "attempt $attempt: exited 0 (done, or --resume found nothing left) -- stopping"
        exit 0
    fi
    log "attempt $attempt: exited $code (killed or crashed) -- retrying in ${RETRY_DELAY_S}s"
    sleep "$RETRY_DELAY_S"
done
log "giving up after $MAX_RETRIES attempts, still failing"
exit 1
