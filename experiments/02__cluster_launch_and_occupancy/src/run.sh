#!/usr/bin/env bash
# Full run of experiment 02. Called by `make run` inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/02__cluster_launch_and_occupancy run
# Order: clock check → K1 (event floor + overhead sweep) → K2 (occupancy probe) → launch-metric name check →
# ncu (launch statistics, --clock-control none keeps our lock) → nsys (back-to-back launch gaps) → clock check.
# Every step appends to results/ with its own run_id; profiler reports stay in bin/ (not results).
set -euo pipefail
cd "$(dirname "$0")/.."

B=bin/cluster_launch
METRICS=sm__cycles_active.avg,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,launch__grid_size,launch__waves_per_multiprocessor,launch__cluster_dim_x,launch__cluster_max_active,launch__cluster_max_potential_size,launch__cluster_scheduling_policy,launch__occupancy_cluster_pct,gpc__cgas_launched.sum,gpc__cgas_active.avg

$B clockcheck --label start
echo "== K1 event floor + overhead sweep"; $B overhead
echo "== K2 occupancy probe";               $B occupancy

echo "== launch metric names on this device"
ncu --query-metrics --query-metrics-collection launch > bin/ncu_query_launch.txt 2>&1 || true
ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
for m in ${METRICS//,/ }; do
    base=${m%.*}; base=${base%.per_second}; base=${base%.avg}; base=${base%.sum}
    if grep -q "^${base} " bin/ncu_query_launch.txt bin/ncu_query_metrics.txt; then echo "  $m: listed"; else echo "  $m: NOT LISTED"; fi
done

echo "== ncu launch statistics"
ncu --clock-control none --cache-control none --kernel-name sleep_kernel --metrics "$METRICS" -f -o bin/ncu_launch $B ncu --label ncu
# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
ncu -i bin/ncu_launch.ncu-rep --csv --page raw --print-units base --print-metric-instances values > bin/ncu_launch.csv
python3 src/post_profile.py --ncu-csv bin/ncu_launch.csv

echo "== nsys launch gaps"
nsys profile --trace cuda --force-overwrite true -o bin/nsys $B nsys --label nsys
nsys stats --help-reports > bin/nsys_reports.txt 2>&1 || true
nsys stats --force-export true --report cuda_gpu_trace,cuda_api_trace,cuda_gpu_kern_sum,cuda_api_sum --format csv --output bin/nsys bin/nsys.nsys-rep
python3 src/post_profile.py --nsys-prefix bin/nsys

$B clockcheck --label end
