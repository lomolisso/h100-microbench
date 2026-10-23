#!/usr/bin/env bash
# Full run of experiment 01. Called by `make run` inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/01__timers_and_harness run
# Order: clock check → harness self-test → K1, K2, K3, K5 (no profiler) → K4 under ncu with
# --clock-control none (our lock) → K4 under ncu with --clock-control base (last, because ncu changes the
# clocks itself) → final clock check. Every step appends to results/ with its own run_id.
set -euo pipefail
cd "$(dirname "$0")/.."

T=bin/timers
# .avg.per_second: clock of each domain; gpc/sm__cycles_elapsed.sum exported per instance (per GPC / per SM).
METRICS=sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,gpc__cycles_elapsed.avg.per_second,gpu__time_duration.sum,gpc__cycles_elapsed.sum,sm__cycles_elapsed.sum

$T clockcheck --label start
echo "== harness self-test";  bin/harness_selftest
echo "== K1 timer cost";      $T k1
echo "== K2 globaltimer";     $T k2
echo "== K2 all SMs";         $T k2all
echo "== K3 loop overhead";   $T k3
echo "== K5 event floor";     $T k5

echo "== K4 metric availability on this device"
ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
for m in sm__cycles_elapsed lts__cycles_elapsed dram__cycles_elapsed gpc__cycles_elapsed gpu__time_duration; do
    if grep -q "^$m " bin/ncu_query_metrics.txt; then echo "  $m: available"; else echo "  $m: NOT LISTED"; fi
done

for mode in none base; do
    echo "== K4 ncu --clock-control $mode"
    ncu --clock-control "$mode" --cache-control none --launch-skip 2 --launch-count 10 \
        --metrics "$METRICS" -f -o "bin/k4_$mode" $T k4 --label "k4_$mode"
    # --print-metric-name is rejected with --page raw by this ncu (2025.2.1); raw headers are metric names already.
    ncu -i "bin/k4_$mode.ncu-rep" --csv --page raw --print-units base --print-metric-instances values > "bin/k4_$mode.csv"
    python3 src/ncu_clock_domains.py --mode "$mode" --ncu-csv "bin/k4_$mode.csv"
    $T clockcheck --label "after_k4_$mode"
done
