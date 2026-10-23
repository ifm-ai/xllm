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
set -u
MBOX=$1
SPARE_SCRIPT=$2
POLL=${FT_HELPER_POLL:-1}
mkdir -p "$MBOX"
SUBMITTED=()

cleanup() {
  for j in "${SUBMITTED[@]}"; do scancel "$j" 2>/dev/null; done
  exit 0
}
trap cleanup TERM INT EXIT

while [ ! -e "$MBOX/stop" ]; do
  for req in "$MBOX"/req.*; do
    [ -e "$req" ] || continue
    id=${req##*/req.}
    read -r action arg < "$req" || continue                # withdrawn by the client after its timeout
    case $action in
      request_node)
        excl=()
        [ "$arg" != "-" ] && excl=(--exclude="$arg")
        if out=$(sbatch --parsable "${excl[@]}" "$SPARE_SCRIPT" 2>&1); then
          jobid=${out%%;*}
          SUBMITTED+=("$jobid")
          reply="ok $jobid"
        else
          reply="error $out"
        fi ;;
      release_node)
        if scancel "$arg" 2>&1; then reply="ok $arg"; else reply="error scancel $arg failed"; fi ;;
      *)
        reply="error unknown request '$action'" ;;
    esac
    echo "$reply" > "$MBOX/rep.$id.tmp" && mv "$MBOX/rep.$id.tmp" "$MBOX/rep.$id"
    rm -f "$req"
  done
  sleep "$POLL"
done
