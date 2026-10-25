#!/usr/bin/env bash
# Full run of experiment 07. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/07__synchronization_primitives run
# Order: clock check → K1 (intra-block) → K3 (mbarrier) → K4 (flag ping-pong, CS 2/4) → K4 CS-16 pairs (TPC indices
# 0-0, 0-1, 0-3, 0-5, 0-7 of one GPC) → K5 (grid sync) → K2 (cluster barrier, the largest output) → every binary's ncu
# mode under ncu (stall sampling, opcode counts) → clock check.
# Usage: src/run.sh [--quick] [step ...], steps: k1 k3 k4 k4cs16 k5 k2 ncu (default: all, in this order); e.g.
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/07__synchronization_primitives run STEPS="k4cs16"
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-k1 k3 k4 k4cs16 k5 k2 ncu}"
want() { [[ " $STEPS " == *" $1 "* ]]; }

bin/sync_cluster clockcheck --label start $Q
if want k1; then echo "== K1 intra-block primitives";       bin/sync_block k1 $Q; fi
if want k3; then echo "== K3 mbarrier operations";          bin/sync_mbarrier k3 $Q; fi
if want k4; then echo "== K4 flag ping-pong (CS 2, CS 4)";  bin/sync_flags k4 $Q; fi
if want k4cs16; then echo "== K4 flag ping-pong, CS-16 pairs"; bin/sync_flags k4cs16 $Q; fi
if want k5; then echo "== K5 grid sync";                    bin/sync_grid k5 $Q; fi
if want k2; then echo "== K2 cluster barrier components";   bin/sync_cluster k2 $Q; fi

if want ncu; then
# Kernel replay, caches untouched (--cache-control none), our clock lock kept (--clock-control none). Warp sampling every
# 2^5 cycles. Every launch of the ncu mode is profiled and joined to the manifest by order (src/post_ncu.py).
# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
for v in block:k1 cluster:k2 mbarrier:k3 flags:k4 grid:k5; do
    IFS=: read -r b t <<< "$v"
    echo "== ncu $b ($t)"
    rm -f "bin/ncu_manifest_$t.csv"  # the manifest (bin/, scratch) must list exactly the launches of this profile
    ncu --clock-control none --cache-control none --section WarpStateStats --section SourceCounters \
        --section InstructionStats --warp-sampling-interval 0 \
        --metrics sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,smsp__cycles_active.avg \
        -f -o "bin/ncu_$t" "bin/sync_$b" ncu
    ncu -i "bin/ncu_$t.ncu-rep" --page raw --csv --print-units base > "bin/ncu_${t}_raw.csv"
    ncu -i "bin/ncu_$t.ncu-rep" --page raw --csv --print-units base --print-metric-instances details \
        --metrics sass__inst_executed_per_opcode > "bin/ncu_${t}_opc.csv"
    # The source page of a multi-launch report repeats one launch per kernel function: export launch by launch, each
    # export behind a "Launch Index" marker (an export prints its block twice; post_ncu.py keeps the first).
    n=$(( $(wc -l < "bin/ncu_manifest_$t.csv") - 1 ))
    : > "bin/ncu_${t}_source.csv"
    for ((j = 0; j < n; ++j)); do
        echo "\"Launch Index\",\"$j\"" >> "bin/ncu_${t}_source.csv"
        ncu -i "bin/ncu_$t.ncu-rep" --page source --csv --print-source sass -s "$j" -c 1 >> "bin/ncu_${t}_source.csv"
    done
    python3 src/post_ncu.py --tag "$t"
done
fi

bin/sync_cluster clockcheck --label end $Q
