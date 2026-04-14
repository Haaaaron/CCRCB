# 4D Lattice Resource Contention Benchmark

High-precision performance tool for quantifying SM and interconnect contention on NVIDIA Grace Hopper (GH200) architectures. Evaluates the performance penalties of overlapping heavy GPU compute kernels with **CUDA-aware MPI**, **NCCL**, and **NVSHMEM** backends.

---

## Build Instructions

The dimensionality is fixed at compile-time via the `NDIM` macro to optimize memory layout and `std::array` access. 

### Separable Compilation (RDC)
Separable compilation is disabled for host objects using `-gpu=nordc` to prevent linker conflicts with the C++ Standard Library (`__fatbinwrap` errors) in NVHPC 25.x.

---

## Configuration (`runs.txt`)

The execution loop is driven by `runs.txt`. The parser uses strict **stream exceptions** to validate data types and formatting.

### Column Ordering
`Volume | Iters | Spill | Warmup | Repeats | Comm_Mask | Backend`

* **Tuples:** `Volume` and `Mask` must be wrapped in `()` with **no spaces** (e.g., `(256,256,256,1)`).
* **Backends:** Accepted strings are `MPI`, `NCCL`, or `NVSHMEM`.

**Example:**
```text
# Volume          Iters  Spill  Warmup  Repeats  Mask                 Backend
(256,256,256,1)   1000   128    100     10       (1,1,1,1,1,1,1,1)    MPI
(256,256,256,1)   1000   128    100     10       (1,1,1,1,1,1,1,1)    NCCL
```

---

## Execution & Profiling

Runtimes require an active MPI environment to initialize communication bootstraps.

### Library Path
Ensure `LD_LIBRARY_PATH` includes the `comm_libs` for NCCL and NVSHMEM to resolve dynamic dependencies.

### Batch Modes
1.  **Standard:** `sbatch submit_benchmark.sh 1`
2.  **Nsight Systems:** `sbatch submit_benchmark.sh 2` (Timeline analysis)
3.  **Nsight Compute:** `sbatch submit_benchmark.sh 3` (Kernel-level metrics, Rank 0 only)

---

## Metrics

The benchmark measures three distinct timing phases to calculate interference:

* $T_{\text{comp}}$: Baseline compute time (isolated SM load).
* $T_{\text{comm}}$: Baseline communication time (isolated fabric load).
* $T_{\text{overlap}}$: Concurrent compute and communication.

### Overlap Efficiency
$$\eta = \frac{\max(T_{\text{comp}}, T_{\text{comm}})}{T_{\text{overlap}}} \times 100\%$$

---

> **Note: Build Environment (Thea)**
> 
> Required CMake configuration for the current environment:
> ```bash
> cmake .. \
>   -DNVHPC_ROOT=/global/exafs/groups/gh/spack-1.0.2-20251008/opt/linux-neoverse_v2/nvhpc-25.9-wzo57cr7ncwj6refsynd5b4bkjtz34dt/Linux_aarch64/25.9 \
>   -DCMAKE_CUDA_COMPILER=/global/exafs/groups/gh/spack-1.0.2-20251008/opt/linux-neoverse_v2/nvhpc-25.9-wzo57cr7ncwj6refsynd5b4bkjtz34dt/Linux_aarch64/25.9/cuda/13.0/bin/nvcc \
>   -DNDIM=4
> ```
