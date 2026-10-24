#!/usr/bin/env bash
# Full run of experiment 04. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/04__memory_hierarchy_latency run
# Order: clock check → K1 (smem) → K3 calib → K3 matrix → pick B touchers and K2 SMs → K3 raw → K2 footprint sweep
# → K2 under ncu (--clock-control none keeps our lock, --cache-control none keeps the warm chase) → clock check.
# The summary (src/summarize.py) needs no GPU and runs afterwards: tools/run --no-gpu -- python3 src/summarize.py
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
[[ "${1:-}" == "--quick" ]] && Q="--quick"
S=bin/lat_smem; F=bin/lat_footprint; X=bin/lat_fine

$S clockcheck --label start $Q
echo "== K1 shared-memory latency";           $S k1 $Q
echo "== K3 timer calibration";               $X calib $Q
echo "== K3 toucher x probe matrix";          $X matrix $Q
PAIRS=$(python3 src/pick_toucher.py $Q)
echo "== K3 raw samples, B touchers: $PAIRS"; $X raw $Q --pairs "$PAIRS"
K2SMS=$(python3 src/pick_toucher.py --k2-sms $Q)
echo "== K2 footprint sweep, SMs $K2SMS";       $F k2 $Q --sms "$K2SMS"

echo "== K2 under ncu: metric availability"
WANT="l1tex__t_sector_hit_rate.pct l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct lts__t_sector_hit_rate.pct
lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum dram__bytes_read.sum
lts__t_sectors_srcunit_ltcfabric.sum lts__d_sectors_fill_device.sum l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
sm__cycles_elapsed.avg.per_second lts__cycles_elapsed.avg.per_second gpu__time_duration.sum"
ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
METRICS="launch__shared_mem_config_size,launch__grid_size,profiler__replayer_passes"
for m in $WANT; do
    base=${m%%.*}
    if grep -q "^$base " bin/ncu_query_metrics.txt; then METRICS="$METRICS,$m"; else echo "  $m: NOT LISTED (dropped)"; fi
done
echo "  metrics: $METRICS"
# Application replay: each pass re-runs the binary, so the profiled launch always follows the warm launch with the
# L2 state it left (kernel replay saves/restores device memory between passes, which wipes the warm L2).
ncu --replay-mode application --clock-control none --cache-control none --kernel-name regex:k2_ncu --metrics "$METRICS" -f -o bin/ncu_k2 $F ncu $Q
# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
ncu -i bin/ncu_k2.ncu-rep --csv --page raw --print-units base > bin/ncu_k2.csv
python3 src/post_ncu.py --ncu-csv bin/ncu_k2.csv --metrics "$METRICS" $Q

$S clockcheck --label end $Q
