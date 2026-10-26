#!/usr/bin/env bash
# Full run of experiment 09. Called by `make run` (or `make quick`: src/run.sh --quick) inside
#   tools/run --profile --lock-sm-mhz 1635 -- make -C experiments/09__dsmem_latency_vs_cluster_size run
# Order: clock check → K3 (local / L2 / HBM references) → K1 (+ K1-fine) all-to-all DSMEM chase → K1f loaded → K2 store
# visibility → K1 under ncu (DSMEM counters) → clock check.
# Usage: src/run.sh [--quick] [step ...], steps: k3 k1 loaded k2 ncu (default: all, in this order).
# Every binary invocation appends to results/ with its own run_id; profiler reports stay in bin/.
set -euo pipefail
cd "$(dirname "$0")/.."

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-k3 k1 loaded k2 ncu}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
L=bin/dsmem_lat; V=bin/dsmem_vis

$L clockcheck --label start $Q
if want k3;     then echo "== K3 local / L2 / HBM reference chases"; $L k3 $Q; fi
if want k1;     then echo "== K1 all-to-all DSMEM chase";            $L k1 $Q; fi
if want loaded; then echo "== K1f loaded DSMEM latency";             $L loaded $Q; fi
if want k2;     then echo "== K2 remote store visibility";           $V k2 $Q; fi

if want ncu; then
echo "== K1 under ncu: metric availability"
WANT="sm__sass_inst_executed_op_dshared_ld.sum sm__sass_inst_executed_op_dshared.sum l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld.sum
l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld.sum l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum
l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared_loopback.sum
l1tex__m_l1tex2xbar_req_cycles_active_mem_dshared.sum lts__t_sectors_srcunit_tex.sum
sm__cycles_elapsed.avg.per_second gpu__time_duration.sum"
ncu --query-metrics > bin/ncu_query_metrics.txt 2>&1 || true
METRICS="launch__grid_size,launch__cluster_dim_x,profiler__replayer_passes"
for m in $WANT; do
    base=${m%%.*}
    if grep -q "^$base " bin/ncu_query_metrics.txt; then METRICS="$METRICS,$m"; else echo "  $m: NOT LISTED (dropped)"; fi
done
echo "  metrics: $METRICS"
rm -f bin/ncu_manifest_k1.csv  # the manifest must list exactly the launches of this profile
# Kernel replay, caches untouched, our clock lock kept. Every k1_chase launch of the ncu mode is profiled (chase + all-exit
# baseline for CS 2 and CS 8) and joined to the manifest by order (src/post_ncu.py).
# --print-metric-name is rejected together with --page raw by this ncu (2025.2.1); raw headers are metric names already.
ncu --clock-control none --cache-control none --kernel-name regex:k1_chase --metrics "$METRICS" -f -o bin/ncu_k1 $L ncu
ncu -i bin/ncu_k1.ncu-rep --csv --page raw --print-units base > bin/ncu_k1_raw.csv
python3 src/post_ncu.py
fi

$L clockcheck --label end $Q
