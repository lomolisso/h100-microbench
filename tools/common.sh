# Shared configuration for tools/*. Sourced, not executed.
# shellcheck shell=bash

H100MB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

# GPUs this group is authorized to use, pinned by UUID and PCI bus id (verified 2026-10-23 with
# `nvidia-smi -q`: index == /dev/nvidiaN minor == CDI device N). Any other index is always refused.
declare -A GPU_UUID=([2]=GPU-428e544c-5649-1dde-a6bf-95de7485dd6a [3]=GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126)
declare -A GPU_PCI=([2]=00000000:C3:00.0 [3]=00000000:DA:00.0)
DEFAULT_GPU=3

# Busy thresholds for the preflight (an idle H100 NVL here shows 14 MiB: Xorg holds 4 MiB on every GPU).
BUSY_MEM_MIB=1024
BUSY_UTIL_PCT=5

# Cooperative lock directory: local to this host, shared by all users (sticky, world-writable).
LOCK_DIR=/tmp/h100-gpu-locks

# Optional personal reservation windows: "<gpu>\t<start ISO>\t<end ISO>\t<note>" per line.
RESERVATIONS=${H100MB_RESERVATIONS:-$HOME/.config/h100-microbench/reservations.tsv}

# Docker's image store and every container's writable layer live in containerd's directory,
# which is a symlink onto the shared scratch volume (/var/lib/containerd -> /srv/local/scratch/containerd).
CONTAINERD_DIR=/srv/local/scratch/containerd
MIN_BUILD_FREE_GB=40   # image build (~15 GB of layers plus headroom)
MIN_RUN_FREE_GB=2      # container start (writable layer is tiny: /tmp is a tmpfs, data goes to NFS)
containerd_free_gb() { df -P -BG "$(dirname "$CONTAINERD_DIR")" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4+0}'; }

IMAGE_REPO=h100-microbench
# Tag = content hash of everything that goes into the image, so a stale image is never used silently.
IMAGE_TAG=$(cat "$H100MB_ROOT/docker/Dockerfile" "$H100MB_ROOT/docker/entrypoint.sh" \
                "$H100MB_ROOT/requirements.lock" | sha256sum | cut -c1-12)
IMAGE="$IMAGE_REPO:cuda12.9.1-$IMAGE_TAG"

say()  { printf '%s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# resolve_gpu <index>: validates against the authorized table; sets GPU_IDX, UUID, PCI.
resolve_gpu() {
    GPU_IDX=$1
    [[ -n "${GPU_UUID[$GPU_IDX]:-}" ]] || die "GPU '$GPU_IDX' is not authorized for this group (allowed: ${!GPU_UUID[*]})"
    UUID=${GPU_UUID[$GPU_IDX]}; PCI=${GPU_PCI[$GPU_IDX]}
}

image_present() { docker image inspect "$IMAGE" >/dev/null 2>&1; }
