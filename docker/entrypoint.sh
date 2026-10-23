#!/usr/bin/env bash
# In-container entrypoint, started by tools/run. Responsibilities:
#   1. verify that exactly the expected GPU is visible (H100MB_GPU_UUID) — or none for --no-gpu;
#   2. in --profile mode (container started as root with CAP_SYS_ADMIN): optionally lock clocks,
#      then drop to the host user while keeping CAP_SYS_ADMIN as an ambient capability (ncu needs it);
#   3. run the command; on SIGTERM (docker stop) forward it and still reset the clock lock.
set -euo pipefail

die() { echo "h100mb-entrypoint: $*" >&2; exit 97; }

if [[ -n "${H100MB_GPU_UUID:-}" ]]; then
    seen=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null || true)
    [[ "$seen" == "$H100MB_GPU_UUID" ]] || die "expected only $H100MB_GPU_UUID, container sees: ${seen:-<none>}"
else
    if nvidia-smi -L >/dev/null 2>&1; then die "--no-gpu container can see a GPU: $(nvidia-smi -L)"; fi
fi

if [[ "$(id -u)" != 0 ]]; then
    exec "$@"
fi

# ---- --profile mode: running as root inside the container ----
: "${H100MB_UID:?}" "${H100MB_GID:?}"
reset_clocks() { [[ -n "${H100MB_LOCK_SM_MHZ:-}" ]] && nvidia-smi -i 0 -rgc >/dev/null && echo "h100mb: clocks reset" >&2 || true; }
if [[ -n "${H100MB_LOCK_SM_MHZ:-}" ]]; then
    nvidia-smi -i 0 -lgc "${H100MB_LOCK_SM_MHZ},${H100MB_LOCK_SM_MHZ}" >&2 || die "clock lock failed"
    trap reset_clocks EXIT
fi
capsh --keep=1 --gid="$H100MB_GID" --groups="$H100MB_GID" --uid="$H100MB_UID" \
      --inh=cap_sys_admin --addamb=cap_sys_admin -- -c 'exec "$@"' h100mb "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
rc=0; wait "$child" || rc=$?
# A trapped signal interrupts the first wait; wait again for the child to actually exit.
kill -0 "$child" 2>/dev/null && { wait "$child" || rc=$?; }
exit "$rc"
