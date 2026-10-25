#!/usr/bin/env bash
# Experiment 13, Step 0: instance-export re-check. Run inside
#   tools/run --profile --lock-sm-mhz 1635 -- bash experiments/13__l2_slice_mapping/src/step0.sh
# Profiles bin/step0_instances (one line, 4096 dependent reads) four ways and checks whether any form yields per-unit
# instance values: (i) kernel replay, --print-metric-instances details; (ii) the same under application replay;
# (iii) --set full report read back with --print-metric-instances details and with the ncu_report Python module
# (num_instances of lts__/fbpa__/l1tex__/sm__ metrics); (iv) the unsuffixed base metric name; (v) beyond the spec:
# --print-metric-instances values, and range / app-range replay of the cudaProfilerStart/Stop range. Also lists whether
# fbpa__ metrics exist. Everything goes to results/13__step0_*.txt (local raw output); ncu --version first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin results
H=../harness
nvcc -arch=sm_90a -O3 -lineinfo -std=c++17 -I$H -o bin/step0_instances src/step0_instances.cu -L${CUDA_HOME:-/usr/local/cuda}/lib64/stubs -lcuda -lnvidia-ml
cuobjdump -sass bin/step0_instances > results/sass/step0_instances.sass
B=bin/step0_instances
M=lts__t_sectors_srcunit_tex_op_read.sum
O=results/13__step0
ncu --version > ${O}_ncu_version.txt
ncu --query-metrics > ${O}_query_metrics.txt 2>&1 || true
{ echo "fbpa__ base metrics listed: $(grep -c '^fbpa__' ${O}_query_metrics.txt)"; grep '^fbpa__dram' ${O}_query_metrics.txt | head -20; } > ${O}_fbpa.txt
for mode in kernel application; do
    for page in raw details; do
        ncu --replay-mode $mode --clock-control none --metrics "$M,fbpa__dram_read_sectors.sum,fbpa__dram_read_sectors.max,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__cycles_active.sum" \
            --print-metric-instances details --csv --page $page $B > ${O}_${mode}_${page}.txt 2>&1 || true
    done
done
for mode in "kernel values" "range details" "app-range details" "range values"; do
    set -- $mode
    ncu --replay-mode $1 --clock-control none --metrics "$M,lts__t_sectors_srcunit_tex_op_read.max,fbpa__dram_read_sectors.sum,sm__cycles_active.sum" \
        --print-metric-instances $2 --csv --page raw $B > ${O}_${1}_${2}_raw.txt 2>&1 || true
done
ncu --replay-mode kernel --clock-control none --set full -f -o bin/step0_full $B > ${O}_full_log.txt 2>&1 || true
ncu -i bin/step0_full.ncu-rep --page raw --csv --print-metric-instances details > ${O}_full_raw.txt 2>&1 || true
ncu --replay-mode kernel --clock-control none --metrics lts__t_sectors_srcunit_tex_op_read,fbpa__dram_read_sectors --csv --page raw $B > ${O}_unsuffixed.txt 2>&1 || true
ncu --replay-mode kernel --clock-control none --metrics "$M,fbpa__dram_read_sectors.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__cycles_active.sum" \
    -f -o bin/step0_sel $B > /dev/null 2>&1 || true
python3 - > ${O}_ncu_report.txt 2>&1 <<'PY' || true
import glob, sys
sys.path.insert(0, glob.glob("/opt/nvidia/nsight-compute/*/extras/python")[0])
import ncu_report
for rep in ["bin/step0_full.ncu-rep", "bin/step0_sel.ncu-rep"]:
    ctx = ncu_report.load_report(rep)
    act = ctx.range_by_idx(0).action_by_idx(0)
    names = sorted(n for n in act.metric_names() if n.split("__")[0] in ("lts", "fbpa", "l1tex", "sm"))
    inst = [(n, act.metric_by_name(n).num_instances()) for n in names]
    with_inst = [(n, k) for n, k in inst if k > 0]
    print(f"{rep}: {len(names)} lts__/fbpa__/l1tex__/sm__ metrics, {len(with_inst)} with instances")
    for n in ["lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex_op_read.max", "fbpa__dram_read_sectors.sum",
              "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sm__cycles_active.sum"]:
        m = act.metric_by_name(n)
        print(f"  {n}: " + ("absent" if m is None else f"value {m.as_double():.0f}, num_instances {m.num_instances()}"))
    for n, k in with_inst[:20]:
        m = act.metric_by_name(n)
        print(f"  WITH INSTANCES {n}: {k} -> {[m.as_double(i) for i in range(min(k, 8))]}")
PY
echo "== summary"; cat ${O}_fbpa.txt | head -3; cat ${O}_ncu_report.txt
cat ${O}_ncu_version.txt
for f in ${O}_kernel_raw.txt ${O}_application_raw.txt ${O}_unsuffixed.txt ${O}_kernel_values_raw.txt ${O}_range_details_raw.txt \
         ${O}_app-range_details_raw.txt ${O}_range_values_raw.txt; do echo "== $f"; python3 - "$f" <<'PY'
import csv, sys
rows = list(csv.reader(l for l in open(sys.argv[1]) if l.startswith('"')))
h = rows[0]; v = [r for r in rows[1:] if r and r[0].strip().isdigit()][0]
print({c: v[i][:80] for i, c in enumerate(h) if c.split(".")[0] in ("lts__t_sectors_srcunit_tex_op_read", "fbpa__dram_read_sectors")})
PY
done
