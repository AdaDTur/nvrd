#!/bin/bash
# Same pattern as relabel_novel_all_models.sh, for the 20 newly-fetched novel objects (scaling the
# study from 10 to all 30 novel objects in the dataset) across the 3 current VLM models.
#
# Launch DETACHED:
#   cd affordance-test && setsid nohup bash relabel_new20_all_models.sh </dev/null >/dev/null 2>&1 & disown
#
# Progress: logs/relabel_new20_chain.log (this script) + logs/watchdog_<model>.log (per model,
# shared log file with the earlier 10-object run -- labeling_watchdog.sh appends, doesn't overwrite).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/relabel_new20_chain.log"
OBJECTS="shape_15 shape_17 shape_18 shape_19 shape_20 shape_21 shape_22 shape_23 shape_24 shape_25 shape_26 shape_28 shape_29 shape_30 shape_31 shape_32 shape_33 shape_38 shape_39 shape_40"

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== chain start (pid=$$) ==="

for model in qwen3-vl-8b internvl3-8b molmo2-8b; do
    log "starting $model"
    bash "$HERE/labeling_watchdog.sh" "$model" $OBJECTS
    log "$model finished (watchdog exited)"
done

log "=== chain complete: all 3 models done for all 20 new novel objects ==="
