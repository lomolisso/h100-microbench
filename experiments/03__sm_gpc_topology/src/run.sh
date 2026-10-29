#!/usr/bin/env bash
# Full run of experiment 03. Called by `make run` inside
#   h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/03__sm_gpc_topology run
# Order: clock check → card identity (nvidia-smi -q) and MIG profile query → cosched (cluster co-scheduling probe)
# → clockdom (per-SM clock rate) → ncu (per-GPC cluster counts, --clock-control none keeps our lock) → map → clock check.
# Every step appends to results/ with its own run_id; profiler reports stay in bin/ (not results).
set -euo pipefail
cd "$(dirname "$0")/.."

B=bin/gpc_probe
R=results/03__sm_gpc_topology
METRICS=gpc__cgas_launched.sum,gpc__cgas_launched.min,gpc__cgas_launched.max,gpc__cgas_launched.avg,launch__cluster_max_active,launch__cluster_dim_x,launch__grid_size,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum

$B clockcheck --label start

echo "== card identity and MIG profiles (appended, never overwritten)"
{ echo "=== $(date -Iseconds) nvidia-smi -q"; nvidia-smi -q 2>&1; } >> ${R}__card.txt
grep -E "Serial Number|GPU UUID|VBIOS Version|Board Part Number|Product Name" ${R}__card.txt | tail -5
{ echo "=== $(date -Iseconds) nvidia-smi mig -lgip"; set +e; nvidia-smi mig -lgip 2>&1; echo "exit code $?"; set -e; } >> ${R}__mig.txt
tail -3 ${R}__mig.txt

echo "== cosched: cluster co-scheduling probe"; $B cosched
echo "== clockdom: per-SM clock rate";          $B clockdom

echo "== ncu per-GPC cluster counts"
ncu --clock-control none --cache-control none --kernel-name sleep_kernel --metrics "$METRICS" -f -o bin/ncu_gpc $B ncu --label ncu
# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
ncu -i bin/ncu_gpc.ncu-rep --csv --page raw --print-units base --print-metric-instances values > bin/ncu_gpc.csv
python3 src/post_profile.py --ncu-csv bin/ncu_gpc.csv

echo "== smid -> GPC map"; python3 src/build_map.py

$B clockcheck --label end
