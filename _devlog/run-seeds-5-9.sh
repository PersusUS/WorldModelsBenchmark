#!/usr/bin/env bash
# Seeds 5-9 on the six cells that discriminate (D20, step 2).
#
# The three control cells are left at five seeds on purpose: they discriminate
# between methods on nothing (D16), so seeds spent there buy no resolution.
#
# Same protocol as results/, so this writes into it and the runner skips the
# seeds already there. It is resumable -- rerun this exact script if it is
# interrupted and it picks up where it stopped.
#
#   bash _devlog/run-seeds-5-9.sh
#
# 150 cells + 30 reference pairs = 360 trainings, roughly a day and a half.
set -u
cd "$(dirname "$0")/../cf_worldmodels" || exit 1

SEEDS="5 6 7 8 9"

run () {
  echo ""
  echo "############ $* ############"
  python experiments/run_full_benchmark.py "$@" --seeds $SEEDS
  echo "############ exit $? ############"
}

run --families minigrid  --distances distance_min distance_med distance_max
run --families gymnasium --distances distance_max
run --families dmcontrol --distances distance_med distance_max

echo ""
echo "############ all three families done ############"
