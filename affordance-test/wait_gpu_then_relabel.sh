#!/bin/bash
# Waits for at least MIN_FREE_MIB free on every GPU (this shared machine has other users' jobs
# fluctuating in and out), then launches relabel_new20_all_models.sh. Must itself be launched via
# setsid+nohup+disown (same reasoning as labeling_watchdog.sh) -- a harness-tracked background Bash
# wait dies when the Claude Code session ends/reconnects, since that's session-level tracking, not
# OS-level detachment; this script is, so it survives.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/wait_gpu_then_relabel.log"
MIN_FREE_MIB=5000

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== wait start (pid=$$), waiting for >${MIN_FREE_MIB}MiB free on every GPU ==="
while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt "$MIN_FREE_MIB" ]; then
        log "GPU memory available (min free across GPUs: ${min_free}MiB) -- launching relabeling"
        break
    fi
    sleep 30
done

bash "$HERE/relabel_new20_all_models.sh"
log "=== relabel_new20_all_models.sh exited ==="
