#!/usr/bin/env bash
# Pick the highest SM clock that GPU <N> sustains without throttling under the suite's heaviest load.
#   tools/clocktest/select_clock.sh [--gpu N] [--seconds S] [MHz ...]   (default candidates: 1785 1755 1725 ... 1500)
# For each candidate, top-down: tools/run --profile --lock-sm-mhz C runs clock_stress (HBM copy, then
# L2 read, S seconds each) while the host samples clock / throttle reasons / power / temperature every
# 100 ms. A candidate passes if no throttle bit (SW/HW power, SW/HW thermal, power brake) is ever set
# and the SM clock never drops below C during load. Stops at the first passing candidate.
# Safety: aborts (docker stop → clock reset) if the GPU exceeds ABORT_TEMP_C. Writes the raw samples
# and a summary to docs/environment/clock_selection/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

gpu=$DEFAULT_GPU; seconds=30; ABORT_TEMP_C=75
while [[ $# -gt 0 ]]; do
    case $1 in --gpu) gpu=$2; shift 2 ;; --seconds) seconds=$2; shift 2 ;; *) break ;; esac
done
cands=("$@"); (( ${#cands[@]} )) || cands=(1785 1755 1725 1695 1665 1635 1605 1575 1545 1515)
resolve_gpu "$gpu"

out=$H100MB_ROOT/docs/environment/clock_selection; mkdir -p "$out"
run_id=$(date -Iseconds)
raw=$out/samples.csv; summary=$out/summary.csv
[[ -s $raw ]]     || echo "run_id,candidate_mhz,timestamp,sm_mhz,mem_mhz,reasons_hex,power_w,gpu_temp_c,mem_temp_c,util_pct" > "$raw"
[[ -s $summary ]] || echo "run_id,gpu_uuid,candidate_mhz,seconds_per_phase,hbm_tbps,l2_tbps,min_sm_mhz_under_load,max_power_w,max_gpu_temp_c,max_mem_temp_c,throttle_samples,verdict" > "$summary"

THROTTLE_MASK=$((0x4 | 0x8 | 0x20 | 0x40 | 0x80))
chosen=""
for c in "${cands[@]}"; do
    say "=== candidate ${c} MHz"
    log=$(mktemp); prog=$(mktemp)
    nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,clocks_event_reasons.active,power.draw,temperature.gpu,temperature.memory,utilization.gpu \
        --format=csv,noheader,nounits -lms 100 > "$log" &
    mon=$!
    ( "$H100MB_ROOT/tools/run" --gpu "$GPU_IDX" --profile --lock-sm-mhz "$c" -- bash -c \
        "nvcc -arch=sm_90a -O3 -o /tmp/cs '$H100MB_ROOT/tools/clocktest/clock_stress.cu' && /tmp/cs --seconds $seconds" > "$prog" 2>&1 ) &
    job=$!
    # Temperature guard while the job runs.
    while kill -0 "$job" 2>/dev/null; do
        t=$(tail -n1 "$log" | awk -F', ' '{print $6+0}')
        if (( ${t:-0} > ABORT_TEMP_C )); then
            say "ABORT: GPU at ${t} C > ${ABORT_TEMP_C} C, stopping"
            docker ps -q --filter "label=h100mb.owner=$USER" --filter "label=h100mb.gpu=$GPU_IDX" | xargs -r docker stop -t 15 >/dev/null
            kill "$mon"; cat "$prog" >&2; exit 2
        fi
        sleep 0.5
    done
    rc=0; wait "$job" || rc=$?
    kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
    cat "$prog" | grep -E 'phase=|ERROR|Lock|reset' >&2 || true
    (( rc == 0 )) || { say "run failed (rc=$rc), stopping"; cat "$prog" >&2; exit 1; }

    awk -F', ' -v r="$run_id" -v c="$c" 'NF>=8 {printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", r,c,$1,$2,$3,$4,$5,$6,$7,$8}' "$log" >> "$raw"
    # Load samples = utilization > 50 %.
    read -r minclk maxp maxt maxmt thr < <(gawk -F', ' -v m=$THROTTLE_MASK '
        NF>=8 && $8+0 > 50 { n++; if (min=="" || $2+0 < min) min=$2+0; if ($5+0 > p) p=$5+0; if ($6+0 > t) t=$6+0; if ($7+0 > mt) mt=$7+0;
                             if (and(strtonum($4), m)) th++ }
        END { print (min==""?0:min), p+0, t+0, mt+0, th+0 }' "$log")
    hbm=$(awk -F'tbps=' '/phase=hbm_copy  tbps/ {print $2}' "$prog"); l2=$(awk -F'tbps=' '/phase=l2_read   tbps/ {print $2}' "$prog")
    verdict=FAIL; (( thr == 0 && minclk >= c )) && verdict=PASS
    echo "$run_id,$UUID,$c,$seconds,$hbm,$l2,$minclk,$maxp,$maxt,$maxmt,$thr,$verdict" >> "$summary"
    say "    min SM clock under load ${minclk} MHz, max power ${maxp} W, max temp ${maxt}/${maxmt} C (gpu/mem), throttle samples ${thr} → $verdict"
    rm -f "$log" "$prog"
    if [[ $verdict == PASS ]]; then chosen=$c; break; fi
    sleep 20   # let the card cool a little before the next candidate
done
[[ -n "$chosen" ]] && say "Highest sustainable SM clock: ${chosen} MHz" || say "No candidate passed"
