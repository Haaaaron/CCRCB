#pragma once

#include "gpu_type.h"
#include "comm_env.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <vector>

enum class CommBackend { CUDA_AWARE_MPI, NCCL, NVSHMEM };

// =====================================================================
// DEVICE KERNELS (Templated for ElementType)
// =====================================================================

template <typename T>
__global__ void kernel_compute(T *d_bulk, const size_t *grid_size, size_t volume, int iters,
                               int math_per_load) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= volume)
        return;

    // Load grid dimensions into registers
    size_t g0 = grid_size[0];
    size_t g1 = grid_size[1];
    size_t g2 = grid_size[2];

    // Initial load
    T res = d_bulk[idx];
    T factor = (T)1.000001;

    // Strides for 4D neighbors
    size_t s0 = 1;
    size_t s1 = g0;
    size_t s2 = g0 * g1;
    size_t s3 = g0 * g1 * g2;

    int load_count = 0;

    // HEAVY COMPUTE & MEMORY LOOP
    for (int i = 0; i < iters; ++i) {
        // 1. Math is ALWAYS executed
        res = (res * factor) + (T)0.0123;
        // Non-linear operation to prevent compiler from collapsing the loop
        if (res > 1e10 || res < -1e10)
            res *= 1e-10;

        // 2. Memory is ONLY fetched periodically based on the AI knob
        // Use a local copy to ensure no division by zero
        int m_load = (math_per_load > 0) ? math_per_load : 1;
        if (i % m_load == 0) {
            size_t neighbor_idx;
            int mode = load_count % 4;
            if (mode == 0)
                neighbor_idx = (idx + s0) % volume;
            else if (mode == 1)
                neighbor_idx = (idx + s1) % volume;
            else if (mode == 2)
                neighbor_idx = (idx + s2) % volume;
            else
                neighbor_idx = (idx + s3) % volume;

            res += d_bulk[neighbor_idx];
            load_count++;
        }
    }

    // Final dependent write
    d_bulk[idx] = res;
}

template <typename T>
__global__ void kernel_pack(const T *d_bulk, T *d_halo, const size_t *strides,
                            const size_t *face_dims, size_t f_vol, int dim_idx,
                            bool is_fwd, int ndim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= f_vol)
        return;

    size_t coords[4]; // Max supported dims for this stack array
    size_t rem = idx;
    for (int d = 0; d < ndim - 1; ++d) {
        coords[d] = rem % face_dims[d];
        rem /= face_dims[d];
    }

    size_t b_idx = 0;
    int f_ptr = 0;
    for (int d = 0; d < ndim; ++d) {
        if (d == dim_idx) {
            b_idx += (is_fwd ? (face_dims[ndim - 1] - 1) : 0) * strides[d];
        } else {
            b_idx += coords[f_ptr++] * strides[d];
        }
    }
    d_halo[idx] = d_bulk[b_idx];
}

template <typename T>
__global__ void kernel_unpack(T *d_bulk, const T *d_halo, const size_t *strides,
                              const size_t *face_dims, size_t f_vol,
                              int dim_idx, bool is_fwd, int ndim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= f_vol)
        return;

    size_t coords[4];
    size_t rem = idx;
    for (int d = 0; d < ndim - 1; ++d) {
        coords[d] = rem % face_dims[d];
        rem /= face_dims[d];
    }

    size_t b_idx = 0;
    int f_ptr = 0;
    for (int d = 0; d < ndim; ++d) {
        if (d == dim_idx) {
            b_idx += (is_fwd ? (face_dims[ndim - 1] - 1) : 0) * strides[d];
        } else {
            b_idx += coords[f_ptr++] * strides[d];
        }
    }
    d_bulk[b_idx] = d_halo[idx];
}

// =====================================================================
// LATTICE OBJECT
// =====================================================================

template <typename ElementType> class BenchmarkLattice {
  private:
    std::array<size_t, NDIM> grid_size;
    std::array<int, 2 * NDIM> active_mask;
    size_t local_volume;
    std::array<size_t, NDIM> face_volumes;
    int compute_iters, math_per_load;
    CommBackend backend;

    static constexpr size_t NUM_DIRS = 2 * NDIM;
    int rank, num_ranks;
    std::array<int, NUM_DIRS> neighbor_ranks;

    ElementType *d_bulk;
    std::array<ElementType *, NUM_DIRS> d_halo_send_bufs, d_halo_recv_bufs;

    size_t *d_strides, *d_grid_size;
    std::array<size_t *, NDIM> d_face_dims;

    gcclComm_t nccl_comm;
    MPI_Comm cart_comm;
    std::array<MPI_Request, NUM_DIRS> send_reqs, recv_reqs;

  public:
    BenchmarkLattice(const std::array<size_t, NDIM> &size, int iters, int m_per_load,
                     const std::vector<int> &mask, CommBackend mode,
                     MPI_Comm mpi_handle, gcclComm_t nccl_handle, int c_rank,
                     int t_ranks)
        : grid_size(size), compute_iters(iters), math_per_load(m_per_load > 0 ? m_per_load : 1),
          backend(mode), nccl_comm(nccl_handle), cart_comm(mpi_handle),
          rank(c_rank), num_ranks(t_ranks) {

        local_volume = 1;
        for (size_t d = 0; d < NDIM; ++d) {
            local_volume *= grid_size[d];
        }

        for (size_t d = 0; d < NDIM; ++d) {
            face_volumes[d] = local_volume / grid_size[d];
        }

        std::copy(mask.begin(), mask.end(), active_mask.begin());

        for (int d = 0; d < (int)NDIM; ++d) {
            int src, dst;
            MPI_Cart_shift(cart_comm, d, 1, &src, &dst);
            // If the neighbor is ourselves (happens in periodic dims of size 1), 
            // treat it as PROC_NULL to skip redundant self-communication.
            neighbor_ranks[2 * d] = (src == rank) ? MPI_PROC_NULL : src;
            neighbor_ranks[2 * d + 1] = (dst == rank) ? MPI_PROC_NULL : dst;
        }

        std::vector<size_t> h_strides(NDIM);
        h_strides[0] = 1;
        for (size_t d = 1; d < NDIM; ++d)
            h_strides[d] = h_strides[d - 1] * grid_size[d - 1];

        gpuMalloc(&d_strides, NDIM * sizeof(size_t));
        gpuMemcpy(d_strides, h_strides.data(), NDIM * sizeof(size_t),
                  gpuMemcpyHostToDevice);

        gpuMalloc(&d_grid_size, NDIM * sizeof(size_t));
        gpuMemcpy(d_grid_size, grid_size.data(), NDIM * sizeof(size_t),
                  gpuMemcpyHostToDevice);

        for (size_t d = 0; d < NDIM; ++d) {
            std::vector<size_t> f_dims;
            for (size_t i = 0; i < NDIM; ++i)
                if (i != d)
                    f_dims.push_back(grid_size[i]);
            f_dims.push_back(grid_size[d]);
            gpuMalloc(&d_face_dims[d], NDIM * sizeof(size_t));
            gpuMemcpy(d_face_dims[d], f_dims.data(), NDIM * sizeof(size_t),
                      gpuMemcpyHostToDevice);
        }

        gpuMalloc(&d_bulk, local_volume * sizeof(ElementType));
        gpuMemset(d_bulk, 0, local_volume * sizeof(ElementType));

        for (size_t i = 0; i < NUM_DIRS; ++i) {
            size_t b_bytes = face_volumes[i / 2] * sizeof(ElementType);
            if (backend == CommBackend::NVSHMEM) {
#ifdef USE_NVSHMEM
                d_halo_send_bufs[i] = (ElementType *)nvshmem_malloc(b_bytes);
                d_halo_recv_bufs[i] = (ElementType *)nvshmem_malloc(b_bytes);
#else
                if (rank == 0)
                    fprintf(stderr,
                            "Error: NVSHMEM requested but not compiled.\n");
                exit(1);
#endif
            } else {
                gpuMalloc(&d_halo_send_bufs[i], b_bytes);
                gpuMalloc(&d_halo_recv_bufs[i], b_bytes);
            }
        }
    }

    ~BenchmarkLattice() {
        gpuFree(d_bulk);
        gpuFree(d_strides);
        gpuFree(d_grid_size);
        for (auto ptr : d_face_dims)
            gpuFree(ptr);
        for (size_t i = 0; i < NUM_DIRS; ++i) {
            if (backend == CommBackend::NVSHMEM) {
#ifdef USE_NVSHMEM
                nvshmem_free(d_halo_send_bufs[i]);
                nvshmem_free(d_halo_recv_bufs[i]);
#endif
            } else {
                gpuFree(d_halo_send_bufs[i]);
                gpuFree(d_halo_recv_bufs[i]);
            }
        }
    }

    void pack_direction(int dir, gpuStream_t stream) {
        if (!active_mask[dir])
            return;
        int d_idx = dir / 2;
        bool is_fwd = dir % 2;
        size_t f_vol = face_volumes[d_idx];
        kernel_pack<<<(f_vol + 255) / 256, 256, 0, stream>>>(
            d_bulk, d_halo_send_bufs[dir], d_strides, d_face_dims[d_idx], f_vol,
            d_idx, is_fwd, (int)NDIM);
    }

    void unpack_direction(int dir, gpuStream_t stream) {
        if (!active_mask[dir])
            return;
        int d_idx = dir / 2;
        bool is_fwd = dir % 2;
        size_t f_vol = face_volumes[d_idx];
        kernel_unpack<<<(f_vol + 255) / 256, 256, 0, stream>>>(
            d_bulk, d_halo_recv_bufs[dir], d_strides, d_face_dims[d_idx], f_vol,
            d_idx, is_fwd, (int)NDIM);
    }

    void start_exchange(int dir, gpuStream_t stream) {
        if (!active_mask[dir])
            return;
        int d_idx = dir / 2;
        size_t b = face_volumes[d_idx] * sizeof(ElementType);
        int p = neighbor_ranks[dir];
        if (p == MPI_PROC_NULL)
            return;

        switch (backend) {
        case CommBackend::NCCL:
#ifdef USE_NCCL
            gcclGroupStart();
            gcclSend(d_halo_send_bufs[dir], b, gccl_type<uint8_t>::value, p,
                     nccl_comm, stream);
            gcclRecv(d_halo_recv_bufs[dir], b, gccl_type<uint8_t>::value, p,
                     nccl_comm, stream);
            gcclGroupEnd();
#else
            if (rank == 0)
                fprintf(stderr, "Error: NCCL not compiled.\n");
            exit(1);
#endif
            break;
        case CommBackend::CUDA_AWARE_MPI: {
            int matching_tag = (dir % 2 == 0) ? (dir + 1) : (dir - 1);
            MPI_Irecv(d_halo_recv_bufs[dir], b, MPI_BYTE, p, matching_tag, cart_comm,
                      &recv_reqs[dir]);
            gpuStreamSynchronize(stream);
            MPI_Isend(d_halo_send_bufs[dir], b, MPI_BYTE, p, dir, cart_comm,
                      &send_reqs[dir]);
            break;
        }
        case CommBackend::NVSHMEM:
            nvshmemx_putmem_nbi_on_stream(d_halo_recv_bufs[dir],
                                          d_halo_send_bufs[dir], b, p, stream);
            break;
        }
    }

    void wait_exchange(int dir, gpuStream_t stream) {
        if (!active_mask[dir] || neighbor_ranks[dir] == MPI_PROC_NULL)
            return;
        if (backend == CommBackend::CUDA_AWARE_MPI) {
            MPI_Wait(&send_reqs[dir], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_reqs[dir], MPI_STATUS_IGNORE);
        } else if (backend == CommBackend::NVSHMEM) {
            nvshmemx_quiet_on_stream(stream);
        }
    }

    void compute_bulk(gpuStream_t stream, int custom_iters = -1) {
        int iters = (custom_iters > 0) ? custom_iters : compute_iters;
        kernel_compute<<<(local_volume + 255) / 256, 256, 0, stream>>>(
            d_bulk, d_grid_size, local_volume, iters, math_per_load);
    }

    /**
     * @brief Global barrier synchronization using the active backend.
     */
    void sync_all() {
        gpuDeviceSynchronize();
        switch (backend) {
        case CommBackend::NVSHMEM:
#ifdef USE_NVSHMEM
            nvshmem_barrier_all();
#else
            MPI_Barrier(cart_comm);
#endif
            break;
        default:
            // For NCCL and MPI, we use the control plane barrier
            MPI_Barrier(cart_comm);
            break;
        }
    }

    /**
     * @brief Estimates the workload of the compute kernel.
     * @return pair of (Total FLOPs, Total Bytes Moved)
     */
    std::pair<double, double> get_workload() const {
        double ops_per_iter = 2.0; // 1 mult, 1 add in the inner loop
        double total_ops = (double)local_volume * compute_iters * ops_per_iter;
        
        // Initial load + Final store
        double base_bytes = (double)local_volume * 2 * sizeof(ElementType);
        // Loads inside the loop (only happen when i % math_per_load == 0)
        double mem_fetches = (double)compute_iters / math_per_load;
        double loop_bytes = (double)local_volume * mem_fetches * sizeof(ElementType);
        
        return {total_ops, base_bytes + loop_bytes};
    }

    void sync() { gpuDeviceSynchronize(); }
};
