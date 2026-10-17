#!/bin/bash
# Orchestrates re-labeling the 10 novel shape_* objects across the 3 VLM models (qwen2-vl-7b
# dropped -- outdated, no longer used), sequentially -- each model shards across all 4 GPUs via
# device_map="auto", so they can't run in parallel on this machine. Runs each model's watchdog
# in-process (blocking, one after another) so this single script's lifetime covers the whole chain.
#
# Launch DETACHED (same reasoning as labeling_watchdog.sh itself):
#   cd affordance-test && setsid nohup bash relabel_novel_all_models.sh </dev/null >/dev/null 2>&1 & disown
#
# Progress: logs/relabel_novel_chain.log (this script) + logs/watchdog_<model>.log (per model, via
# labeling_watchdog.sh, which already handles crash-restart with --resume).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/relabel_novel_chain.log"
OBJECTS="shape_16 shape_27 shape_1 shape_2 shape_3 shape_10 shape_11 shape_12 shape_13 shape_14"

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== chain start (pid=$$) ==="

for model in qwen3-vl-8b internvl3-8b molmo2-8b; do
    log "starting $model"
    bash "$HERE/labeling_watchdog.sh" "$model" $OBJECTS
    log "$model finished (watchdog exited)"
done

log "=== chain complete: all 3 models done for all 10 novel objects ==="
