#!/bin/bash -l
#SBATCH --nodes=2
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

# Unified Library Path
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$NVSHMEM_HOME/lib:$NVHPC_ROOT/cuda/13.0/lib64:$LD_LIBRARY_PATH

# --- 2. COMMUNICATION TUNING (NCCL & NVSHMEM) ---
# NCCL Optimized for Grace Hopper
export NCCL_NET_GDR_LEVEL=3
export NCCL_DMABUF_ENABLE=1
export NCCL_CROSS_NIC='1'
export NCCL_NET_GDR_LEVEL="PHB"
export NCCL_PROTO="simple"

# NVSHMEM Runtime Settings
export NVSHMEM_SYMMETRIC_SIZE=16G
export NVSHMEM_BOOTSTRAP=mpi
export NVSHMEM_REMOTE_TRANSPORT=ibrc

# MPI / CUDA Visibility
export OMPI_MCA_opal_cuda_support=1
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# --- 3. EXECUTION SETUP ---
NUM_TASKS=$SLURM_NTASKS
mkdir -p results/benchmark slurm_output slurm_errors

# Get mode from argument (1: Standard, 2: Nsys, 3: Ncu)
MODE=${1:-1}
BINARY="./build/benchmark"
CONFIG="runs.txt" # Ensure this file exists in your root

echo "Executing Benchmark on $NUM_TASKS ranks (Nodes: $SLURM_NNODES)"
echo "Mode: $MODE"

case $MODE in
  1)
    echo "Starting Standard Performance Run..."
    mpirun -np $NUM_TASKS --map-by ppr:1:node:PE=36 --report-bindings \
        $BINARY $CONFIG >> results/benchmark/output-${SLURM_JOB_ID}.out 2>&1
    ;;

  2)
    echo "Starting Nsight Systems Profile..."
    PROF_DIR="./results/benchmark/nsys_job_${SLURM_JOB_ID}"
    mkdir -p $PROF_DIR

    mpirun -np $NUM_TASKS --map-by ppr:1:node:PE=36 --report-bindings \
        nsys profile \
        --trace=cuda,mpi,nvtx,osrt \
        --stats=true \
        --output=$PROF_DIR/rank%q{OMPI_COMM_WORLD_RANK} \
        $BINARY $CONFIG >> results/benchmark/output-${SLURM_JOB_ID}.out 2>&1
    ;;

  3)
    echo "Starting Nsight Compute Profile (Rank 0 Only)..."
    # NCU Wrapper to avoid profiling all ranks simultaneously
    echo '#!/bin/bash
    if [ "$OMPI_COMM_WORLD_RANK" -eq "0" ]; then
        ncu --set full \
            --target-processes all \
            --force-overwrite \
            -o ./results/benchmark/ncu_job_'${SLURM_JOB_ID}' \
            "$@"
    else
        "$@"
    fi' > ./ncu_wrapper.sh
    chmod +x ./ncu_wrapper.sh

    mpirun -np $NUM_TASKS --map-by ppr:1:node:PE=36 --report-bindings \
        ./ncu_wrapper.sh $BINARY $CONFIG >> results/benchmark/output-${SLURM_JOB_ID}.out 2>&1
    ;;

  *)
    echo "Invalid mode. Use 1 (Standard), 2 (nsys), or 3 (ncu)."
    exit 1
    ;;
esac

echo "Run Complete."
