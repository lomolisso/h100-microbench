#!/usr/bin/env bash
# Full run of experiment 10. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/10__dsmem_latency_heatmap run
# Order: clock check → steering pilot → schedules E, A, C, B, D, D8 (src/heatmap.cu) → K1 under ncu → clock check.
# Usage: src/run.sh [--quick] [step ...], steps: steer E E2 A C B D D8 K ncu (default: all, in this order; K = clock check).
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."
# the binary also sets it itself: lazy module loading deadlocks against the resident placeholder kernel
export CUDA_MODULE_LOADING=EAGER

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-steer E E2 A C B D D8 K ncu}"
H=bin/heatmap

$H clockcheck --label start $Q
for s in $STEPS; do
    case $s in
    steer) echo "== steering pilot"; $H steer $Q ;;
    E|E2|A|C|B|D|D8|K) echo "== schedule $s"; $H sched --schedule $s $Q ;;
    ncu)
        echo "== K1 under ncu (lone CS-16 and CS-8 cluster, chase + all-exit baseline)"
        WANT="sm__sass_inst_executed_op_dshared_ld.sum sm__sass_inst_executed_op_dshared.sum l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld.sum
l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared_loopback.sum l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared.sum
sm__cycles_elapsed.avg.per_second gpu__time_duration.sum"
        ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
        METRICS="launch__grid_size,launch__cluster_dim_x,profiler__replayer_passes"
        for m in $WANT; do
            base=${m%%.*}
            if grep -q "^$base " bin/ncu_query_metrics.txt; then METRICS="$METRICS,$m"; else echo "  $m: NOT LISTED (dropped)"; fi
        done
        rm -f bin/ncu_manifest_k1.csv
        # kernel replay, caches untouched, our clock lock kept; joined to the manifest by launch order (src/post_ncu.py)
        ncu --clock-control none --cache-control none --kernel-name regex:k1_chase --metrics "$METRICS" -f -o bin/ncu_k1 $H ncu
        ncu -i bin/ncu_k1.ncu-rep --csv --page raw --print-units base > bin/ncu_k1_raw.csv
        python3 src/post_ncu.py
        ;;
    *) echo "unknown step $s"; exit 1 ;;
    esac
done
$H clockcheck --label end $Q
