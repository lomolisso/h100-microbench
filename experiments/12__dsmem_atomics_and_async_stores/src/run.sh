#!/usr/bin/env bash
# Full run of experiment 12. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   h100 run --profile --lock-sm-mhz 1635 -- make -C experiments/12__dsmem_atomics_and_async_stores run
# Order: clock check → K0a (per-SM L2 partition, writes results/12__atomics__sm_partition.csv, read by every later step)
# → K0b labels → K1 (atom_lat) → K6 global flags → K2/K3 (atom_tput) → K4–K7 (dsmem_async) → profiled launches under ncu
# → clock check.
# Usage: src/run.sh [--quick] [step ...], steps: k0a k0b calib smem dsmem pairs l2 l2fine hbm sys flags k2 k2all k3a k3b k3c
#   k3d k4 k4ring k5 k6 k7 ncu (default: all, in this order). Every binary invocation appends to results/ with its own run_id;
# profiler reports and manifests stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-k0a k0b calib smem dsmem pairs l2 l2fine hbm sys flags k2 k2all k2x k3a k3b k3c k3d k4 k4ring k5 k6 k6poll1 k7 ncu}"
LAT=bin/atom_lat TPUT=bin/atom_tput DS=bin/dsmem_async

# metric list → only the names this ncu lists (dropped names are printed)
filter_metrics() {
    local out="launch__grid_size,launch__block_size,launch__cluster_dim_x,profiler__replayer_passes" m base
    for m in $1; do
        base=${m%%.*}
        if grep -q "^$base " bin/ncu_query_metrics.txt; then out="$out,$m"; else echo "  $m: NOT LISTED (dropped)" >&2; fi
    done
    echo "$out"
}
GLOBAL_M="lts__t_sectors_op_atom.sum lts__t_sectors_op_red.sum lts__t_requests_srcunit_tex_op_atom.sum lts__t_requests_srcunit_tex_op_red.sum
lts__t_sectors_srcunit_ltcfabric.sum lts__t_sectors_op_atom.max lts__t_sectors_op_red.max lts__t_sectors_op_atom.avg lts__t_sectors_op_red.avg
lts__t_sectors_srcunit_tex_lookup_hit.sum lts__t_sectors_srcunit_tex_lookup_miss.sum lts__t_sectors_op_read.sum lts__t_sectors_op_write.sum
dram__bytes_read.sum dram__bytes_write.sum smsp__inst_executed_op_global_atom.sum smsp__inst_executed_op_global_red.sum
l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum l1tex__t_requests_pipe_lsu_mem_global_op_red.sum sm__cycles_elapsed.avg.per_second
gpu__time_duration.sum"
SMEM_M="sm__sass_inst_executed_op_dshared.sum sm__sass_inst_executed_op_dshared_atom.sum sm__sass_inst_executed_op_dshared_redas.sum
sm__sass_inst_executed_op_dshared_stas.sum l1tex__t_requests_pipe_lsu_mem_dshared_op_atom.sum l1tex__t_requests_pipe_lsu_mem_dshared_op_redas.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_atom.sum l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_redas.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared.max
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum l1tex__t_set_conflicts_pipe_lsu_mem_dshared_op_atom.sum
smsp__inst_executed_op_shared_atom.sum l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
sm__cycles_elapsed.avg.per_second gpu__time_duration.sum"
NCU="ncu --clock-control none --csv --page raw --print-units base"

# one profiled invocation: $1 = tag, $2 = replay mode, $3 = kernel regex, $4 = launch-skip, $5 = launch-count, $6 = metrics, rest = command
ncu_one() {
    local tag=$1 replay=$2 kre=$3 skip=$4 count=$5 met=$6; shift 6
    rm -f bin/ncu_manifest.csv
    $NCU --replay-mode "$replay" --cache-control none --kernel-name "regex:$kre" --launch-skip "$skip" --launch-count "$count" \
        --metrics "$met" -f -o "bin/ncu_$tag" "$@" > "bin/ncu_$tag.log" 2>&1 || { echo "ncu $tag failed"; tail -5 "bin/ncu_$tag.log"; return 1; }
    ncu -i "bin/ncu_$tag.ncu-rep" --csv --page raw --print-units base > "bin/ncu_raw_$tag.csv"
    python3 src/post_ncu.py "$tag" "bin/ncu_raw_$tag.csv" bin/ncu_manifest.csv
}

$LAT clockcheck --label start $Q
for s in $STEPS; do
    echo "== step $s"
    case $s in
    k0a|k0b|calib|smem|dsmem|pairs|l2|l2fine|hbm|sys|flags) $LAT $s $Q ;;
    k2|k2all|k2x|k3a|k3b|k3c|k3d) $TPUT $s $Q ;;
    k4|k4ring|k5|k6|k6poll1|k7) $DS $s $Q ;;
    ncu)
        ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
        GM=$(filter_metrics "$GLOBAL_M"); SM=$(filter_metrics "$SMEM_M")
        # K1 L2 protocols (application replay: the L2 state of the protocol is part of the measurement; every k1_glob launch)
        ncu_one l2 application '^k1_glob' 0 25 "$GM" $LAT ncu_l2
        # K2/K3: per configuration; the k_tput launches are baseline (warm-ups 0, 1, recorded 2) then profiled (3, 4, 5):
        # profile 2 … 5 (post_ncu keeps the baseline and the profiled recorded launch)
        $TPUT ncu1 --list | grep '^[0-9]' > bin/ncu_tput_list.txt
        n=$(wc -l < bin/ncu_tput_list.txt)
        for i in $(seq 0 $((n - 1))); do
            lab=$(awk -v i=$i '$1 == i {print $2}' bin/ncu_tput_list.txt)
            if [[ $lab == */global/* ]]; then ncu_one "tput$i" application '^k_tput' 2 4 "$GM" $TPUT ncu1 --cfg $i
            else ncu_one "tput$i" kernel '^k_tput' 2 4 "$SM" $TPUT ncu1 --cfg $i; fi
        done
        # K4/K5 DSMEM (kernel replay): baseline launch then the profiled one
        n=$($DS ncu1 | grep -c '^[0-9]' || true)
        for i in $(seq 0 $((n - 1))); do ncu_one "ds$i" kernel '^k[45]_' 0 2 "$SM" $DS ncu1 --cfg $i; done
        ;;
    *) echo "unknown step $s"; exit 1 ;;
    esac
done
$LAT clockcheck --label end $Q
