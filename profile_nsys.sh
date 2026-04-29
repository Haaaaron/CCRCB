#!/bin/bash
# ====================================================================================
# NSYS MPI Rank Wrapper (Rank 0 ONLY for stability)
# ====================================================================================

# Detect rank
if [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    RANK=$OMPI_COMM_WORLD_RANK
elif [ -n "$PMIX_RANK" ]; then
    RANK=$PMIX_RANK
elif [ -n "$SLURM_PROCID" ]; then
    RANK=$SLURM_PROCID
else
    RANK=0
fi

# Force Absolute Path for Output
OUT_DIR=$(readlink -f "${NSYS_OUT_DIR:-.}")
mkdir -p "$OUT_DIR"

if [ "$RANK" -eq 0 ]; then
    echo "[Rank 0] Starting NSYS Profiling..."
    OUT_NAME="nsys_rank0"
    
    # NSYS Configuration:
    # Tracing cuda, mpi, nvtx, and osrt for rank 0.
    nsys profile \
        --trace=cuda,mpi,nvtx,osrt \
        --gpu-metrics-devices=cuda-visible \
        --gpu-metrics-set=gh100 \
        --gpu-metrics-frequency=10000 \
        --nic-metrics=true \
        --force-overwrite=true \
        --wait=all \
        --output="$OUT_DIR/$OUT_NAME" \
        "$@" \
        2>&1 | tee -a "$OUT_DIR/output_rank0.txt"

    NSYS_STATUS=$?
    sync
    exit $NSYS_STATUS
else
    # Non-zero ranks: Run benchmark directly.
    # We use a separate output file for non-zero ranks to avoid mixing stdout.
    "$@" > "$OUT_DIR/output_rank${RANK}.txt" 2>&1
    exit $?
fi
