#!/bin/bash
# Emit SLURM job state changes and key log lines for one job, exit when it leaves the queue.
JOB="${1:?usage: watch_job.sh <jobid>}"
LOG="outputs/slurm_logs/whiten_${JOB}.log"
FILTER='Loading cached|Cache miss|Evaluating|[0-9]+/[0-9]+ nodes|Skipped|Wrote .* rows|Traceback|Error|Exception|FAILED|Killed|[Oo]ut of memory|slurmstepd'
prev="init"
emitted=0
while true; do
  st=$(squeue -h -j "$JOB" -o "%t %R" 2>/dev/null)
  if [ "$st" != "$prev" ]; then
    echo "[state] ${st:-<not in queue>}"
    prev="$st"
  fi
  if [ -f "$LOG" ]; then
    total=$(grep -acE "$FILTER" "$LOG" 2>/dev/null || echo 0)
    if [ "$total" -gt "$emitted" ]; then
      grep -aE "$FILTER" "$LOG" | tail -n +$((emitted + 1))
      emitted=$total
    fi
  fi
  if [ -z "$(squeue -h -j "$JOB" -o '%t' 2>/dev/null)" ]; then
    echo "[done] job $JOB left queue: $(sacct -n -j "$JOB" --format=State,ExitCode,Elapsed 2>/dev/null | head -1 | tr -s ' ')"
    [ -f "$LOG" ] && grep -aE "$FILTER" "$LOG" | tail -n +$((emitted + 1))
    break
  fi
  sleep 20
done
