#!/usr/bin/env bash
# Full run of experiment 11. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/11__dsmem_bandwidth_vs_cluster_size run
# Order: clock check → steps A S B C D E F P1 K4 of bin/dsmem_bw → K5 (bin/dsmem_l2mix) → GMEM reference lines (05's
# L2 and HBM bandwidth steps, appended to 05's results with their own run_ids) → profiled launches under ncu → clock check.
# Usage: src/run.sh [--quick] [step ...], steps: A S B C D E F P1 K4 K5 ref ncu (default: all, in this order).
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-A S B C D E F P1 K4 K5 ref ncu}"
B=bin/dsmem_bw

$B clockcheck --label start $Q
for s in $STEPS; do
    case $s in
    A|S|B|C|D|E|F|P1|K4) echo "== step $s"; $B $s $Q ;;
    K5) echo "== step K5 (DSMEM vs L2 interference)"; bin/dsmem_l2mix k5 $Q ;;
    ref)
        echo "== GMEM reference lines: 05's L2 (k3) and HBM (k4) bandwidth on this card"
        make -s -C ../05__memory_hierarchy_bandwidth bin/bw_l2 bin/bw_dram
        (cd ../05__memory_hierarchy_bandwidth && bin/bw_l2 k3 $Q && bin/bw_dram k4 $Q)
        ;;
    ncu)
        echo "== k_stream under ncu (CS 8, 96 KiB, 1 sample per launch)"
        WANT="l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum.per_second l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum.per_second
l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed l1tex__m_xbar2l1tex_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed
l1tex__m_l1tex2xbar_write_bytes_mem_dshared.avg.peak_sustained l1tex__m_xbar2l1tex_read_bytes_mem_dshared.avg.peak_sustained
sm__sass_inst_executed_op_dshared_tma_st.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st.sum l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared.max l1tex__m_l1tex2xbar_write_bytes_mem_dshared.avg l1tex__m_l1tex2xbar_write_bytes_mem_dshared.min
l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum
l1tex__m_xbar2l1tex_read_bytes_mem_dshared.max l1tex__m_xbar2l1tex_read_bytes_mem_dshared.avg l1tex__m_xbar2l1tex_read_bytes_mem_dshared.min
l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum
sm__sass_inst_executed_op_dshared_ld.sum sm__sass_inst_executed_op_dshared_st.sum sm__sass_inst_executed_op_dshared_stas.sum
sm__sass_inst_executed_op_dshared_tma_red.sum
sm__cycles_elapsed.avg.per_second gpu__time_duration.sum"
        ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
        METRICS="launch__grid_size,launch__cluster_dim_x,launch__block_size,profiler__replayer_passes"
        for m in $WANT; do
            base=${m%%.*}
            if grep -q "^$base " bin/ncu_query_metrics.txt; then METRICS="$METRICS,$m"; else echo "  $m: NOT LISTED (dropped)"; fi
        done
        rm -f bin/ncu_manifest.csv
        # kernel replay, caches untouched, our clock lock kept; joined to the manifest by launch order (src/post_ncu.py)
        ncu --clock-control none --cache-control none --kernel-name regex:k_stream --metrics "$METRICS" -f -o bin/ncu_k $B ncu
        ncu -i bin/ncu_k.ncu-rep --csv --page raw --print-units base > bin/ncu_raw.csv
        python3 src/post_ncu.py
        ;;
    *) echo "unknown step $s"; exit 1 ;;
    esac
done
$B clockcheck --label end $Q
