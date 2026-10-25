#!/usr/bin/env bash
# Full run of experiment 08. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/08__l2_partitions_and_first_touch run
# Order: clock check → calib → K1 → K2 → K3 → K5 → K4 (with the calibrated timer pair) → ncu (per-line slice
# instances + L1 bypass, partition-link sectors of first/second reads, K4 hit rates) → clock check.
# Usage: src/run.sh [--quick] [step ...], steps: calib k1 k2 k2vmm k3 k5 k4 ncu ncufab1 (default: all, in this order;
# ncu also runs the superseded ncu_fabric, ncufab1 its replacement).
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
# The summary (src/summarize.py) needs no GPU: tools/run --no-gpu -- python3 src/summarize.py
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-calib k1 k2 k2vmm k3 k5 k4 ncu ncufab1}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
P=bin/l2_probe; C=bin/l2_capacity

$P clockcheck --label start $Q
if want calib; then echo "== timer calibration";             $P calib $Q; fi
if want k1;    then echo "== K1 single-line probe";          $P k1 $Q; fi
if want k2;    then echo "== K2 partition hash bits";        $P k2 $Q; fi
if want k2vmm; then echo "== K2 on cuMemCreate allocations"; $P k2vmm $Q; fi
if want k3;    then echo "== K3 replication";                $P k3 $Q; fi
if want k5;    then echo "== K5 per-SM signature";           $P k5 $Q; fi
if want k4;    then
    TP=$(cd src && python3 -c "import common08; print(common08.timer_pair('../results', quick=$([[ -n $Q ]] && echo True || echo False)))")
    echo "== K4 effective capacity (timer pair $TP cycles)";  $C k4 $Q --timer-pair "$TP"
fi

if want ncu; then
    WANT="lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum
lts__t_sectors_srcunit_tex_op_read_lookup_hit.max lts__t_sectors_srcunit_tex_op_read_lookup_miss.max lts__t_sectors_srcunit_ltcfabric.max
lts__t_sectors_srcunit_ltcfabric.sum lts__t_sector_hit_rate.pct l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum lts__t_sectors_op_read.sum lts__t_sectors_op_write.sum dram__bytes_read.sum
device__attribute_l2s_count sm__cycles_elapsed.avg.per_second gpu__time_duration.sum"
    ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
    METRICS="profiler__replayer_passes"
    for m in $WANT; do
        base=${m%%.*}
        if grep -q "^$base " bin/ncu_query_metrics.txt || [[ $base == device__attribute_* ]]; then METRICS="$METRICS,$m"
        else echo "  $m: NOT LISTED (dropped)"; fi
    done
    echo "  metrics: $METRICS"
    # Application replay: every pass re-runs the binary with the same cache-state protocol, so each profiled launch sees
    # the L2 state its setup left (kernel replay's save/restore would wipe it). Our clock lock stays (--clock-control none).
    # --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
    for v in lines:$P:ncu_lines fabric:$P:ncu_fabric capacity:$C:ncu; do
        IFS=: read -r t b m <<< "$v"
        echo "== ncu $t"
        SINCE=$(date -u +%Y-%m-%dT%H:%M:%S)
        ncu --replay-mode application --clock-control none --cache-control none --kernel-name regex:k_ncu --metrics "$METRICS" \
            -f -o "bin/ncu_$t" "$b" "$m" $Q
        ncu -i "bin/ncu_$t.ncu-rep" --csv --page raw --print-units base --print-metric-instances values > "bin/ncu_$t.csv"
        python3 src/post_ncu.py --tag "$t" --mode "$m" --since "$SINCE" --ncu-csv "bin/ncu_$t.csv" --metrics "$METRICS"
    done
fi

if want ncufab1; then
    # One profiled kernel per binary run (--launch-skip k --launch-count 1: the unprofiled launches before it leave the
    # cache state undisturbed by the profiler) and one lts__ counter per run (this ncu needs 4 replay passes for any
    # two lts__ metrics; dram__ and gpu__time fit in the same pass). Launch 0 = same-grid all-exit baseline.
    for m in lookup_miss:lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum lookup_hit:lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum \
             fabric:lts__t_sectors_srcunit_ltcfabric.sum; do
        tag=${m%%:*}; MET="profiler__replayer_passes,${m#*:},dram__bytes_read.sum,gpu__time_duration.sum"
        for k in 0 1 2 3; do
            echo "== ncu fabric1 $tag launch $k"
            SINCE=$(date -u +%Y-%m-%dT%H:%M:%S)
            ncu --replay-mode application --clock-control none --cache-control none --kernel-name regex:k_ncu_fab --launch-skip $k \
                --launch-count 1 --metrics "$MET" -f -o "bin/ncu_fabric1_${tag}_$k" $P ncu_fabric1 $Q
            ncu -i "bin/ncu_fabric1_${tag}_$k.ncu-rep" --csv --page raw --print-units base > "bin/ncu_fabric1_${tag}_$k.csv"
            python3 src/post_ncu.py --tag "fabric1_$tag" --mode ncu_fabric1 --since "$SINCE" --ncu-csv "bin/ncu_fabric1_${tag}_$k.csv" \
                --metrics "$MET" --order $k
        done
    done
fi

$P clockcheck --label end $Q
