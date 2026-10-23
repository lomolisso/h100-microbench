# h100-microbench toolchain image. Build with `tools/env build`; never `docker run` it directly —
# use `tools/run`, which does the GPU preflight, the lock and the device restriction.
#
# CUDA 12.9.1 matches the ptxas version the docs were validated with (and the host's Lmod cuda-toolkit/12.9).
# Ships nvcc 12.9.86, Nsight Compute 2025.2.1, Nsight Systems, cuda-gdb, compute-sanitizer.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git make gdb rsync less jq bc procps \
        libcap2-bin util-linux ca-certificates \
        cuda-nsight-compute-12-9 cuda-nsight-systems-12-9 \
    && (command -v ncu || ln -s /opt/nvidia/nsight-compute/*/ncu /usr/local/bin/ncu) \
    && (command -v nsys || ln -s /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/bin/nsys) \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /opt/requirements.lock
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /opt/requirements.lock \
    && /opt/venv/bin/python -m ipykernel install --prefix /opt/venv --name python3

# Record exact versions inside the image (read by tools/env verify and the notebooks).
RUN { nvcc --version | tail -2; ncu --version | tail -1; nsys --version; python3 --version; \
      dpkg-query -W -f '${Package}=${Version}\n'; /opt/venv/bin/pip freeze; } > /opt/env-manifest.txt

COPY docker/entrypoint.sh /usr/local/bin/h100mb-entrypoint
RUN chmod 0755 /usr/local/bin/h100mb-entrypoint

# NVIDIA_VISIBLE_DEVICES=void fails closed: the host's default docker runtime is `nvidia`, which would
# otherwise expose every GPU; tools/run always overrides it with the single selected UUID.
# BLAS/OpenMP caps: on this 256-core shared host an unbounded pool spins one thread per core on import.
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=void \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/mpl

ENTRYPOINT ["/usr/local/bin/h100mb-entrypoint"]
CMD ["bash"]
