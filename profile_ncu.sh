#!/bin/bash
# ====================================================================================
# NCU MPI Rank Wrapper (STRICT Rank 0 ONLY, Kernel Replay)
# ====================================================================================

# Detect rank
if [ -n "$OMPI_COMM_WORLD_RANK" ]; then RANK=$OMPI_COMM_WORLD_RANK
elif [ -n "$PMIX_RANK" ]; then RANK=$PMIX_RANK
elif [ -n "$SLURM_PROCID" ]; then RANK=$SLURM_PROCID
else RANK=0; fi

# Force Absolute Path for Output
OUT_DIR=$(readlink -f "${NCU_OUT_DIR:-.}")
mkdir -p "$OUT_DIR"

if [ "$RANK" -eq 0 ]; then
    echo "[Rank 0] NCU Start. Target Dir: $OUT_DIR"
    
    # Absolute path to report file
    OUT_BASE="$OUT_DIR/ncu_metrics_rank0"
    
    # Essential metrics only (1-2 passes)
    METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

    # NCU Configuration:
    # 1. We remove --nvtx-include because it was skipping all kernels.
    # 2. We use --kernel-name regex to catch exactly what we need.
    # 3. --profile-from-start=no ensures we only capture between gpuProfilerStart/Stop.
    ncu --profile-from-start=no \
        --nvtx \
        --kernel-name "regex:kernel_compute|kernel_pack|kernel_unpack" \
        --kernel-name-base=function \
        --replay-mode=kernel \
        --clock-control=base \
        --metrics="$METRICS" \
        --force-overwrite \
        -o "$OUT_BASE" \
        "$@" 2>&1 | tee -a "$OUT_DIR/output_rank0.txt"

    
    NCU_STATUS=$?
    sync
    
    # Debug: Check if file exists
    if [ -f "${OUT_BASE}.ncu-rep" ]; then
        echo "[Rank 0] SUCCESS: ${OUT_BASE}.ncu-rep created."
    else
        echo "[Rank 0] ERROR: NCU finished but ${OUT_BASE}.ncu-rep is MISSING."
    fi
    exit $NCU_STATUS
else
    # Non-zero ranks: Run benchmark directly.
    exec "$@"
fi
