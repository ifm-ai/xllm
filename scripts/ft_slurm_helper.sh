#!/usr/bin/env bash
# Host-side SLURM helper for xLLM in-band fault tolerance (ft).
#
# The ft agents run inside the container, where the SLURM commands may not
# work, so they ask this helper through a mailbox directory instead:
#   req.<id>  "request_node <exclude-hosts or ->"  -> sbatch a 1-node refill-spare job
#   req.<id>  "release_node <jobid>"               -> scancel that refill job
# The helper answers with rep.<id> ("ok <jobid>" or "error <message>"). When
# the main job ends (this script is signalled or the stop file appears), it
# cancels every refill job it submitted that is still pending or running.
#
# Usage (started in the background by the sbatch script, outside the container):
#   scripts/ft_slurm_helper.sh <mailbox dir> <refill-spare sbatch script>
#
# Invariant: FT_SBATCH_TIMEOUT (default 90s) < SlurmMailbox.timeout (120s)
set -u
MBOX=$1
SPARE_SCRIPT=$2
POLL=${FT_HELPER_POLL:-1}
FT_SBATCH_TIMEOUT=${FT_SBATCH_TIMEOUT:-90}
FT_SCANCEL_TIMEOUT=${FT_SCANCEL_TIMEOUT:-30}
mkdir -p "$MBOX"
SUBMITTED=()

# Re-adopt orphaned requests previously claimed by a helper on this host that died
START_TIME=$(date +%s)
HOST=$(hostname)
for taken in "$MBOX"/req.*.taken."$HOST".*; do
  [ -e "$taken" ] || continue
  pid=${taken##*.}
  if ! kill -0 "$pid" 2>/dev/null; then
    mtime=$(stat -c %Y "$taken" 2>/dev/null || echo "$START_TIME")
    if [ "$mtime" -le "$START_TIME" ]; then
      orig=${taken%%.taken.*}
      mv -n "$taken" "$orig" 2>/dev/null || rm -f "$taken"
    fi
  fi
done

cleanup() {
  for j in "${SUBMITTED[@]}"; do scancel "$j" 2>/dev/null; done
  exit 0
}
trap cleanup TERM INT EXIT

RUN=${FT_MAIN_RESTART:-}
ticker() {  # its own loop: a slow sbatch must not look like a dead helper to the watchdogs
  local n=0
  while kill -0 $$ 2>/dev/null; do
    n=$((n + 1))
    echo "$(hostname) $$ $n" > "$MBOX/.helper.tick.$$" && mv -f "$MBOX/.helper.tick.$$" "$MBOX/helper.tick"
    sleep 1
  done
}
ticker &
# until the job ends, or a later run of the job (a requeue) owns the mailbox; a watchdog judges this helper alive by
# helper.tick (scripts/ft_watchdog.sh), so exactly one helper answers requests
while [ ! -e "$MBOX/stop" ] && { [ -z "$RUN" ] || [ "$(cat "$MBOX/run" 2>/dev/null || echo "$RUN")" = "$RUN" ]; }; do
  for req in "$MBOX"/req.*; do
    case $req in *.taken.*) continue ;; esac               # another helper's (during a takeover)
    id=${req##*/req.}
    mine=$req.taken.$(hostname).$$
    mv "$req" "$mine" 2>/dev/null || continue              # gone: withdrawn, or another helper took it
    req=$mine
    read -r action arg < "$req" || { rm -f "$req"; continue; }
    case $action in
      request_node)
        excl=()
        [ "$arg" != "-" ] && excl=(--exclude="$arg")
        if out=$(timeout "$FT_SBATCH_TIMEOUT" sbatch --parsable "${excl[@]}" "$SPARE_SCRIPT" 2>&1); then
          jobid=${out%%;*}
          SUBMITTED+=("$jobid")
          : > "$MBOX/submitted.$jobid"  # also for the watchdog, should this helper die with its node
          reply="ok $jobid"
        else
          reply="error ${out:-timeout}"
        fi ;;
      release_node)
        if timeout "$FT_SCANCEL_TIMEOUT" scancel "$arg" 2>&1; then reply="ok $arg"; rm -f "$MBOX/submitted.$arg"; else reply="error scancel $arg failed"; fi ;;
      *)
        reply="error unknown request '$action'" ;;
    esac
    echo "$reply" > "$MBOX/rep.$id.tmp" && mv "$MBOX/rep.$id.tmp" "$MBOX/rep.$id"
    rm -f "$req"
  done
  sleep "$POLL"
done
