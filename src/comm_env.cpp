#include "comm_env.h"
#include "gpu_type.h"
#include <iostream>
#include <vector>

void init_benchmark_env(int argc, char **argv, int Dims, MPI_Comm &cart_comm,
                        gcclComm_t &nccl_comm) {
  // 1. Initialize MPI with Thread Multiple
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

#ifdef USE_NVSHMEM
  if (provided < MPI_THREAD_MULTIPLE) {
    std::cerr << "Fatal Error: MPI_THREAD_MULTIPLE required for NVSHMEM but "
                 "not supported."
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
#endif

  int rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  // 2. Cartesian Topology Setup
  std::vector<int> mpi_dims(Dims, 0);
  MPI_Dims_create(num_ranks, Dims, mpi_dims.data());

  std::vector<int> periods(Dims, 1);
  MPI_Cart_create(MPI_COMM_WORLD, Dims, mpi_dims.data(), periods.data(), 1,
                  &cart_comm);

  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);

  // 3. GPU Affinity Setup
  MPI_Comm node_comm;
  MPI_Comm_split_type(cart_comm, MPI_COMM_TYPE_SHARED, cart_rank, MPI_INFO_NULL,
                      &node_comm);

  int local_rank;
  MPI_Comm_rank(node_comm, &local_rank);

  // Set Device using our abstraction
#if defined(USE_HIP)
  hipSetDevice(local_rank);
#else
  cudaSetDevice(local_rank);
#endif
  MPI_Comm_free(&node_comm);

  // 4. Initialize NCCL / RCCL
#ifdef USE_NCCL
  ncclUniqueId nccl_id;
  if (cart_rank == 0) {
    ncclGetUniqueId(&nccl_id);
  }
  MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, cart_comm);
  gcclCommInitRank(&nccl_comm, num_ranks, nccl_id, cart_rank);
#else
  nccl_comm = nullptr; // Safety for the void* stub
#endif

  // 5. Initialize NVSHMEM
#ifdef USE_NVSHMEM
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &cart_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif
}

void finalize_benchmark_env(MPI_Comm cart_comm, gcclComm_t nccl_comm) {
  // 1. Finalize NVSHMEM
#ifdef USE_NVSHMEM
  nvshmem_finalize();
#endif

  // 2. Finalize NCCL
#ifdef USE_NCCL
  gcclCommDestroy(nccl_comm);
#endif

  // 3. Finalize MPI
  if (cart_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm);
  }
  MPI_Finalize();
}
