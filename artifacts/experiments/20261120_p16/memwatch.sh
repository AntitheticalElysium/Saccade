#!/bin/sh
# Host-memory log for the resumed P1.6 evaluation (11-21: the first evaluation froze the host in S1, no swap, the
# unit's 24G cap above what the host had left). Every 20 s: MemAvailable, Shmem, the unit's current and peak memory,
# the top shmem and RSS holders. Stops the evaluation unit if MemAvailable < 2 GB twice in a row (a stop, not a freeze).
UNIT=saccade-p16-eval
LOG=artifacts/experiments/20261120_p16/checks/memory_resume.log
low=0
while :; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  shm=$(awk '/^Shmem:/ {print int($2/1024)}' /proc/meminfo)
  cur=$(systemctl --user show $UNIT -p MemoryCurrent --value 2>/dev/null)
  peak=$(systemctl --user show $UNIT -p MemoryPeak --value 2>/dev/null)
  top=$(for s in /proc/[0-9]*/status; do awk -v p="${s%/status}" '/^Name:/ {n=$2} /^RssShmem:/ {sh=$2} /^VmRSS:/ {r=$2} END {if (r) printf "%d %d %s %s\n", sh/1024, r/1024, n, p}' "$s" 2>/dev/null; done)
  tsh=$(echo "$top" | sort -rn | head -3 | awk '{printf "%s:%sM ", $3, $1}')
  trs=$(echo "$top" | sort -k2 -rn | head -4 | awk '{printf "%s:%sM ", $3, $2}')
  echo "$(date +%T) avail ${avail}M shmem ${shm}M unit ${cur} peak ${peak} | shmem: $tsh| rss: $trs" >> $LOG
  if [ "$avail" -lt 2048 ]; then low=$((low + 1)); else low=0; fi
  if [ "$low" -ge 2 ]; then
    echo "$(date +%T) MemAvailable < 2 GB twice: stopping $UNIT" >> $LOG
    systemctl --user stop $UNIT; low=0
  fi
  systemctl --user is-active -q $UNIT || [ -n "$started" ] || { sleep 20; continue; }
  started=1
  systemctl --user is-active -q $UNIT || { echo "$(date +%T) $UNIT inactive, logger exits" >> $LOG; exit 0; }
  sleep 20
done
