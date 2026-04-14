#pragma once

#include "gpu_type.h"
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
__global__ void kernel_compute(T *d_bulk, size_t volume, int iters,
                               int spill_words) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume)
    return;

  T val = d_bulk[idx];
  T spill_stack[512];

#pragma unroll 4
  for (int i = 0; i < spill_words && i < 512; ++i) {
    spill_stack[i] = val + (T)i;
  }

#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    // Note: fma is specialized for double/float; using standard op for generic
    // T
    val = val * spill_stack[i % spill_words] + (T)1.000001;
  }
  d_bulk[idx] = val;
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
      b_idx += (is_fwd ? (face_dims[d] - 1) : 0) * strides[d];
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
      b_idx += (is_fwd ? (face_dims[d] - 1) : 0) * strides[d];
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
  size_t local_volume, face_volume;
  int compute_iters, spill_words;
  CommBackend backend;

  static constexpr size_t NUM_DIRS = 2 * NDIM;
  int rank, num_ranks;
  std::array<int, NUM_DIRS> neighbor_ranks;

  ElementType *d_bulk;
  std::array<ElementType *, NUM_DIRS> d_halo_send_bufs, d_halo_recv_bufs;

  size_t *d_strides;
  std::array<size_t *, NDIM> d_face_dims;

  gcclComm_t nccl_comm;
  MPI_Comm cart_comm;
  std::array<MPI_Request, NUM_DIRS> send_reqs, recv_reqs;

public:
  BenchmarkLattice(const std::array<size_t, NDIM> &size, int iters, int spill,
                   const std::vector<int> &mask, CommBackend mode,
                   MPI_Comm mpi_handle, gcclComm_t nccl_handle, int c_rank,
                   int t_ranks)
      : grid_size(size), compute_iters(iters), spill_words(spill),
        backend(mode), nccl_comm(nccl_handle), cart_comm(mpi_handle),
        rank(c_rank), num_ranks(t_ranks) {

    local_volume = 1;
    for (size_t d = 0; d < NDIM; ++d)
      local_volume *= grid_size[d];
    face_volume = local_volume / grid_size[0];
    std::copy(mask.begin(), mask.end(), active_mask.begin());

    for (int d = 0; d < (int)NDIM; ++d) {
      int src, dst;
      MPI_Cart_shift(cart_comm, d, 1, &src, &dst);
      neighbor_ranks[2 * d] = src;
      neighbor_ranks[2 * d + 1] = dst;
    }

    std::vector<size_t> h_strides(NDIM);
    h_strides[0] = 1;
    for (size_t d = 1; d < NDIM; ++d)
      h_strides[d] = h_strides[d - 1] * grid_size[d - 1];

    gpuMalloc(&d_strides, NDIM * sizeof(size_t));
    gpuMemcpy(d_strides, h_strides.data(), NDIM * sizeof(size_t),
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
    size_t b_bytes = face_volume * sizeof(ElementType);
    for (size_t i = 0; i < NUM_DIRS; ++i) {
      if (backend == CommBackend::NVSHMEM) {
#ifdef USE_NVSHMEM
        d_halo_send_bufs[i] = (ElementType *)nvshmem_malloc(b_bytes);
        d_halo_recv_bufs[i] = (ElementType *)nvshmem_malloc(b_bytes);
#else
        if (rank == 0)
          fprintf(stderr, "Error: NVSHMEM requested but not compiled.\n");
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
    kernel_pack<<<(face_volume + 255) / 256, 256, 0, stream>>>(
        d_bulk, d_halo_send_bufs[dir], d_strides, d_face_dims[d_idx],
        face_volume, d_idx, is_fwd, (int)NDIM);
  }

  void unpack_direction(int dir, gpuStream_t stream) {
    if (!active_mask[dir])
      return;
    int d_idx = dir / 2;
    bool is_fwd = dir % 2;
    kernel_unpack<<<(face_volume + 255) / 256, 256, 0, stream>>>(
        d_bulk, d_halo_recv_bufs[dir], d_strides, d_face_dims[d_idx],
        face_volume, d_idx, is_fwd, (int)NDIM);
  }

  void start_exchange(int dir, gpuStream_t stream) {
    if (!active_mask[dir])
      return;
    size_t b = face_volume * sizeof(ElementType);
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
    case CommBackend::CUDA_AWARE_MPI:
      MPI_Irecv(d_halo_recv_bufs[dir], b, MPI_BYTE, p, dir, cart_comm,
                &recv_reqs[dir]);
      gpuStreamSynchronize(stream);
      MPI_Isend(d_halo_send_bufs[dir], b, MPI_BYTE, p, dir, cart_comm,
                &send_reqs[dir]);
      break;
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

  void compute_bulk(gpuStream_t stream) {
    kernel_compute<<<(local_volume + 255) / 256, 256, 0, stream>>>(
        d_bulk, local_volume, compute_iters, spill_words);
  }

  void sync() { gpuDeviceSynchronize(); }
};
