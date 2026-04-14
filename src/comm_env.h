#ifndef COMM_ENV_H_
#define COMM_ENV_H_
#pragma once

#include "gpu_type.h"
#include <mpi.h>

/**
 * @brief Initializes the execution environment.
 * * Sets up MPI Cartesian topology, GPU affinity, and conditionally
 * initializes NCCL and NVSHMEM based on compile-time flags.
 */
void init_benchmark_env(int argc, char **argv, int Dims, MPI_Comm &cart_comm,
                        gcclComm_t &nccl_comm);

/**
 * @brief Tears down the environment in reverse initialization order.
 */
void finalize_benchmark_env(MPI_Comm cart_comm, gcclComm_t nccl_comm);

#endif // COMM_ENV_H_
