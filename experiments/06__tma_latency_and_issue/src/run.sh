#!/usr/bin/env bash
# Full run of experiment 06. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/06__tma_latency_and_issue run
# Order: clock check → K2 (completion, predicate costs) → K3 (issue / admission) → K4 (boxes) → K1/K5 (chase, the long
# step) → the three ncu-mode invocations under ncu → clock check. The summary (src/summarize.py) needs no GPU:
#   tools/run --no-gpu -- python3 src/summarize.py
# Usage: src/run.sh [--quick] [step ...], steps: k2 k3 k4 chase ncu (default: all, in this order).
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-k2 k3 k4 chase ncu}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
C=bin/tma_chase; P=bin/tma_completion; I=bin/tma_issue; B=bin/tma_boxes

$C clockcheck --label start $Q
if want k2;    then echo "== K2 completion vs payload, predicate and expect_tx costs"; $P k2 $Q; fi
if want k3;    then echo "== K3 issue gap and admission limit";                         $I k3 $Q; fi
if want k4;    then echo "== K4 tensor-map boxes and descriptor state";                 $B k4 $Q; fi
if want chase; then echo "== K1/K5 TMA and generic pointer chase vs footprint";         $C chase $Q; fi

if want ncu; then
echo "== ncu: metric availability"
ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
WANT="sm__sass_inst_executed_op_tma_ld.sum l1tex__tmain_requests.sum
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.per_cycle_active
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.max.per_cycle_active l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.avg.peak_sustained
sm__pipe_tma_cycles_active.avg sm__pipe_tma_cycles_active.max sm__cycles_active.max
lts__t_sectors_srcunit_tex_op_read.sum lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum dram__bytes_read.sum
sm__cycles_elapsed.avg.per_second lts__cycles_elapsed.avg.per_second gpu__time_duration.sum"
METRICS="launch__shared_mem_config_size,launch__grid_size,launch__block_size,profiler__replayer_passes"
for m in $WANT; do
    if grep -q "^${m%%.*} " bin/ncu_query_metrics.txt; then METRICS="$METRICS,$m"; else echo "  $m: NOT LISTED (dropped)"; fi
done
echo "  metrics: $METRICS"
# Application replay: each pass re-runs the binary, so the profiled launch always follows its own warm launch with the
# L2 state it left (kernel replay's save/restore of device memory wipes a warm L2). --clock-control none keeps our lock.
# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
for v in chase:k1_ncu:$C completion:k2_ncu:$P issue:k3_ncu:$I; do
    IFS=: read -r name kern bin <<< "$v"
    echo "== ncu $name"
    ncu --replay-mode application --clock-control none --cache-control none --kernel-name "regex:$kern" --metrics "$METRICS" \
        -f -o "bin/ncu_$name" $bin ncu $Q
    ncu -i "bin/ncu_$name.ncu-rep" --csv --page raw --print-units base > "bin/ncu_$name.csv"
    python3 src/post_ncu.py --ncu-csv "bin/ncu_$name.csv" --manifest "bin/ncu_manifest_$name.csv" --metrics "$METRICS" \
        --out "results/06__tma__ncu_$name.csv"
done
fi

$C clockcheck --label end $Q
