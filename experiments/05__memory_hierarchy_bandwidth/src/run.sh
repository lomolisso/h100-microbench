#!/usr/bin/env bash
# Full run of experiment 05. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/05__memory_hierarchy_bandwidth run
# Order: clock check → K1/K2 (SMEM, L1) → K3/K5 (L2, incl. home labelling) → L2 partition link → first-pass test with
# residency check → store copies → K4/K5 (HBM) → the binaries under ncu → clock check. The summary (src/summarize.py) needs no GPU:
#   tools/run --no-gpu -- python3 src/summarize.py
# Usage: src/run.sh [--quick] [step ...], steps: k12 k3 link firstpass storecopy k4 ncu12 ncul2 ncufp ncudram
# (default: all, in this order).
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-k12 k3 link firstpass storecopy k4 ncu12 ncul2 ncufp ncudram}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
S=bin/bw_smem_l1; L=bin/bw_l2; D=bin/bw_dram

$S clockcheck --label start $Q
if want k12;  then echo "== K1/K2 shared memory and L1";  $S k12 $Q; fi
if want k3;   then echo "== K3/K5 L2";                    $L k3 $Q; fi
if want link; then echo "== L2 partition link";           $L link $Q; fi
if want firstpass; then echo "== L2 first pass (one direction, residency check)"; $L firstpass $Q; fi
if want storecopy; then echo "== L2 store copies";       $L storecopy $Q; fi
if want k4;   then echo "== K4/K5 HBM";                   $D k4 $Q; fi

echo "== ncu: metric availability"
ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
available() {  # keep the metrics whose base name the device lists
    local out="launch__shared_mem_config_size,launch__grid_size,launch__block_size,profiler__replayer_passes"
    for m in $1; do
        if grep -q "^${m%%.*} " bin/ncu_query_metrics.txt; then out="$out,$m"; else echo "  $m: NOT LISTED (dropped)" >&2; fi
    done
    echo "$out"
}
COMMON="sm__cycles_elapsed.avg.per_second sm__cycles_elapsed.max gpu__time_duration.sum lts__cycles_elapsed.avg.per_second
l1tex__throughput.avg.pct_of_peak_sustained_elapsed lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed lts__t_bytes.sum.per_second dram__bytes.sum.per_second
lts__t_sectors_op_read.sum lts__t_sectors_op_write.sum dram__bytes_read.sum dram__bytes_write.sum"
M12=$(available "$COMMON l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
smsp__inst_executed_op_shared_ld.sum smsp__inst_executed_op_shared_st.sum smsp__inst_executed_op_global_ld.sum
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
ML2=$(available "$COMMON lts__t_sectors_srcunit_ltcfabric.sum lts__t_sector_hit_rate.pct lts__d_sectors_fill_device.sum
lts__t_sectors_srcunit_tex_op_read.sum lts__t_sectors_srcunit_tex_op_write.sum lts__t_bytes.sum")
MD=$(available "$COMMON lts__t_sectors_srcunit_ltcfabric.sum lts__t_sector_hit_rate.pct lts__d_sectors_fill_device.sum lts__t_bytes.sum")
echo "  smem/l1: $M12"; echo "  l2: $ML2"; echo "  dram: $MD"

# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
if want ncu12; then
echo "== ncu K1/K2 (kernel replay; each kernel warms its own L1)"
ncu --clock-control none --cache-control all --kernel-name regex:k12 --metrics "$M12" -f -o bin/ncu_smem_l1 $S ncu $Q
ncu -i bin/ncu_smem_l1.ncu-rep --csv --page raw --print-units base > bin/ncu_smem_l1.csv
python3 src/post_ncu.py --ncu-csv bin/ncu_smem_l1.csv --manifest bin/ncu_manifest_smem_l1.csv --metrics "$M12" --out results/05__bandwidth__ncu_smem_l1.csv
fi

if want ncul2; then
# Application replay: each pass re-runs the binary, so every profiled launch follows its own warm-up and keeps the L2
# state it would have (kernel replay's save/restore of device memory wipes a warm L2).
echo "== ncu K3/K5 L2 (application replay, warm L2)"
ncu --replay-mode application --clock-control none --cache-control none --kernel-name regex:k3_ncu --metrics "$ML2" -f -o bin/ncu_l2 $L ncu $Q
ncu -i bin/ncu_l2.ncu-rep --csv --page raw --print-units base > bin/ncu_l2.csv
python3 src/post_ncu.py --ncu-csv bin/ncu_l2.csv --manifest bin/ncu_manifest_l2.csv --metrics "$ML2" --out results/05__bandwidth__ncu_l2.csv
fi

if want ncufp; then
# This ncu collects one lts__ counter per replay pass (any two lts__ metrics need 4 passes; dram__ and gpu__time fit in
# the same pass). So one application run per L2 counter, each a single pass with its own same-grid all-exit baseline
# profiled in the same session (the profiler's own traffic depends on the grid).
echo "== ncu first-pass residency (one pass per L2 counter)"
for m in lookup_miss:lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum lookup_hit:lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum \
         fabric:lts__t_sectors_srcunit_ltcfabric.sum write:lts__t_sectors_op_write.sum; do
    tag=${m%%:*}
    MFP=$(available "${m#*:} dram__bytes_read.sum gpu__time_duration.sum")
    ncu --replay-mode application --clock-control none --cache-control none --kernel-name regex:k3_ncu --metrics "$MFP" -f -o bin/ncu_fp_$tag $L ncufp $Q
    ncu -i bin/ncu_fp_$tag.ncu-rep --csv --page raw --print-units base > bin/ncu_fp_$tag.csv
    python3 src/post_ncu.py --ncu-csv bin/ncu_fp_$tag.csv --manifest bin/ncu_manifest_fp.csv --metrics "$MFP" --out results/05__bandwidth__ncu_fp_$tag.csv
done
fi

if want ncudram; then
echo "== ncu K4/K5 HBM (kernel replay, L2 flushed before every pass)"
ncu --clock-control none --cache-control all --kernel-name regex:k4 --metrics "$MD" -f -o bin/ncu_dram $D ncu $Q
ncu -i bin/ncu_dram.ncu-rep --csv --page raw --print-units base > bin/ncu_dram.csv
python3 src/post_ncu.py --ncu-csv bin/ncu_dram.csv --manifest bin/ncu_manifest_dram.csv --metrics "$MD" --out results/05__bandwidth__ncu_dram.csv
fi

$S clockcheck --label end $Q
