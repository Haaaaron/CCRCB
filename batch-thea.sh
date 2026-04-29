#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=gh
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH -o ./slurm_output/bench_%j.txt
#SBATCH -e ./slurm_errors/bench_%j.txt

# --- 1. SYSTEM CONFIGURATION ---
module load cuda/13.0.0 nvhpc/25.9 openmpi/5.0.8_gcc-15.1.0-cuda_13.0.0 ucx/1.19.0_gcc-13.3.0-cuda_13.0.0

export NVHPC_ROOT=/global/exafs/groups/gh/spack-1.0.2-20251008/opt/linux-neoverse_v2/nvhpc-25.9-wzo57cr7ncwj6refsynd5b4bkjtz34dt/Linux_aarch64/25.9
export NCCL_HOME=$NVHPC_ROOT/comm_libs/nccl
export NVSHMEM_HOME=$NVHPC_ROOT/comm_libs/nvshmem
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NVSHMEM_HOME/lib:$NVHPC_ROOT/cuda/13.0/lib64:$LD_LIBRARY_PATH

# --- 2. COMMUNICATION TUNING ---
export NCCL_NET_GDR_LEVEL=3
export NCCL_DMABUF_ENABLE=1
export NCCL_CROSS_NIC='1'
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_PROTO="simple"
export NVSHMEM_SYMMETRIC_SIZE=16G
export NVSHMEM_BOOTSTRAP=mpi
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export OMPI_MCA_opal_cuda_support=1
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# --- 3. EXECUTION SETUP ---
NUM_TASKS=$SLURM_NTASKS
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
mkdir -p slurm_output slurm_errors
export TMPDIR=/tmp
export OMPI_MCA_prte_silence_shared_fs=1

MODE=${1:-1}
CONFIG="${2:-${SUBMIT_DIR}/runs/runs-test-analysis.txt}"
BINARY="${SUBMIT_DIR}/build/benchmark"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Executing Benchmark on $NUM_TASKS ranks"
echo "Mode: $MODE"
echo "Config: $CONFIG"

case $MODE in
  1)
    RESULTS_DIR="${SUBMIT_DIR}/results_timing_${NUM_TASKS}_ranks_${TIMESTAMP}"
    mkdir -p "$RESULTS_DIR"
    mpirun -np $NUM_TASKS --map-by ppr:1:node:PE=36 --report-bindings \
        bash -c 'if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then "$0" "$@" 2>&1 | tee -a '"$RESULTS_DIR"'/output_rank${OMPI_COMM_WORLD_RANK:-0}.txt; else "$0" "$@" >> '"$RESULTS_DIR"'/output_rank${OMPI_COMM_WORLD_RANK:-0}.txt 2>&1; fi' $BINARY $CONFIG $RESULTS_DIR
    ;;

  2|profile_ncu)
    RESULTS_DIR="${SUBMIT_DIR}/results_ncu_${NUM_TASKS}_ranks_${TIMESTAMP}"
    NCU_DIR="$RESULTS_DIR/ncu"
    mkdir -p "$NCU_DIR"

    export CCRCB_PROFILING_ONLY=1
    cp "$CONFIG" "$RESULTS_DIR/run_config.txt"

    echo "Running Nsight Compute (Targeting Rank 0)..." | tee -a "$RESULTS_DIR/output.txt"
    chmod +x ${SUBMIT_DIR}/profile_ncu.sh
    export CCRCB_NCU_PASS=1
    export NCU_OUT_DIR="$NCU_DIR"
    
    mpirun -np $NUM_TASKS --map-by ppr:1:node:PE=36 --report-bindings \
        -x CCRCB_PROFILING_ONLY -x CCRCB_NCU_PASS -x NCU_OUT_DIR \
        "${SUBMIT_DIR}/profile_ncu.sh" "$BINARY" "$CONFIG" "$RESULTS_DIR" 2>&1 | tee -a "$RESULTS_DIR/output.txt"
    wait
    unset CCRCB_NCU_PASS

    echo "Exporting NCU reports..." | tee -a "$RESULTS_DIR/output.txt"
    if [ -f "${NCU_DIR}/ncu_metrics_rank0.ncu-rep" ]; then
        ncu --import "${NCU_DIR}/ncu_metrics_rank0.ncu-rep" --csv --page=raw > "${NCU_DIR}/ncu_metrics_rank0.csv" 2>/dev/null
        echo "  -> Rank 0 NCU metrics produced." | tee -a "$RESULTS_DIR/output.txt"
    else
        echo "  -> ERROR: ncu_metrics_rank0.ncu-rep not found in $NCU_DIR" | tee -a "$RESULTS_DIR/output.txt"
    fi
    ;;

  3|profile_nsys)
    RESULTS_DIR="${SUBMIT_DIR}/results_metrics_${NUM_TASKS}_ranks_${TIMESTAMP}"
    NSYS_DIR="$RESULTS_DIR/nsys"
    mkdir -p "$NSYS_DIR"

    export CCRCB_PROFILING_ONLY=1
    cp "$CONFIG" "$RESULTS_DIR/run_config.txt"

    echo "Running Nsight Systems (Full Timeline)..." | tee -a "$RESULTS_DIR/output.txt"
    chmod +x ${SUBMIT_DIR}/profile_nsys.sh
    export CCRCB_NSYS_PASS=1
    export NSYS_OUT_DIR="$NSYS_DIR"
    
    mpirun -np $NUM_TASKS --map-by ppr:1:node:PE=36 --report-bindings \
        -x CCRCB_PROFILING_ONLY -x CCRCB_NSYS_PASS -x NSYS_OUT_DIR \
        "${SUBMIT_DIR}/profile_nsys.sh" "$BINARY" "$CONFIG" "$RESULTS_DIR" 2>&1 | tee -a "$RESULTS_DIR/output.txt"
    wait
    unset CCRCB_NSYS_PASS

    echo "Waiting for NSYS files to flush..." | tee -a "$RESULTS_DIR/output.txt"
    sleep 5

    echo "Exporting NSYS timelines to SQLite..." | tee -a "$RESULTS_DIR/output.txt"

    STATS_DIR="${NSYS_DIR}/stats"
    mkdir -p "$STATS_DIR"
    
    if [ -f "${NSYS_DIR}/nsys_rank0.nsys-rep" ]; then
        echo "  -> Exporting nsys_rank0.nsys-rep to SQLite..." | tee -a "$RESULTS_DIR/output.txt"
        nsys export --type=sqlite "${NSYS_DIR}/nsys_rank0.nsys-rep" --output "${NSYS_DIR}/nsys_rank0.sqlite" --force-overwrite true >/dev/null 2>&1
        mv "${NSYS_DIR}/nsys_rank0.nsys-rep" "$STATS_DIR/"
    else
        echo "  -> ERROR: nsys_rank0.nsys-rep not found in $NSYS_DIR" | tee -a "$RESULTS_DIR/output.txt"
    fi

    # Generate Stats from Rank 0 NSYS
    SQLITE_BASE="${NSYS_DIR}/nsys_rank0.sqlite"

    if [ -f "$SQLITE_BASE" ]; then
        NUM_RUNS=$(grep -v '^#' "$CONFIG" | grep '(' | wc -l)
        BACKENDS=($(grep -v '^#' "$CONFIG" | grep '(' | awk '{print $NF}'))
        echo "Generating NSYS CSV Stats..." | tee -a "$RESULTS_DIR/output.txt"
        for i in $(seq 0 $((NUM_RUNS-1))); do
            TAG="${BACKENDS[$i]}_Run$i"
            nsys stats --report nvtx_sum --format csv --filter-nvtx="Profiling_Overlapped_$TAG" \
                --output "$STATS_DIR/nsys_nvtx_run$i" "$SQLITE_BASE" >/dev/null 2>&1
        done
    fi
    ;;
esac

echo "Run Complete."
