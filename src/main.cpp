#include "benchmark_lattice.h"
#include "comm_env.h"
#include "config_parser.h"
#include "gpu_type.h"
#include <algorithm>
#include <iostream>
#include <vector>

#ifndef NDIM
#define NDIM 4
#endif

void report_metrics(int rank, const RunConfig &cfg, double t_comp,
                    double t_comm, double t_overlap) {
  if (rank == 0) {
    double ideal = std::max(t_comp, t_comm);
    double efficiency = (ideal / t_overlap) * 100.0;
    double contention = t_overlap / ideal;

    printf("\n==========================================================\n");
    printf(" CONFIGURATION:\n");
    printf("  Backend:    %-10s | Dim: %dD\n", cfg.backend_name.c_str(),
           (int)NDIM);
    printf("  Iters (AI): %-10d | Spill (Words): %d\n", cfg.iters,
           cfg.spill_words);
    printf("  Repeats:    %-10d | Warmup: %d\n", cfg.repeats, cfg.warmup);
    printf("----------------------------------------------------------\n");
    printf(" PERFORMANCE:\n");
    printf("  T_compute:  %8.3f ms\n", t_comp);
    printf("  T_comm:     %8.3f ms\n", t_comm);
    printf("  T_overlap:  %8.3f ms\n", t_overlap);
    printf("----------------------------------------------------------\n");
    printf(" ANALYSIS:\n");
    printf("  Overlap Efficiency: %6.2f%%\n", efficiency);
    printf("  Contention Factor:  %6.2fx\n", contention);
    printf("==========================================================\n");
  }
}

bool is_backend_enabled(CommBackend backend) {
#ifndef USE_NCCL
  if (backend == CommBackend::NCCL)
    return false;
#endif
#ifndef USE_NVSHMEM
  if (backend == CommBackend::NVSHMEM)
    return false;
#endif
  return true;
}

int main(int argc, char *argv[]) {
  MPI_Comm cart_comm;
  gcclComm_t nccl_comm;

  init_benchmark_env(argc, argv, NDIM, cart_comm, nccl_comm);

  int rank, num_ranks;
  MPI_Comm_rank(cart_comm, &rank);
  MPI_Comm_size(cart_comm, &num_ranks);

  std::string config_file = (argc > 1) ? argv[1] : "runs.txt";
  auto runs = parse_run_file(config_file, rank);

  for (const auto &run : runs) {
    if (run.volume.size() != NDIM) {
      if (rank == 0)
        printf("[SKIP] %zuD run requested (Compiled for %dD)\n",
               run.volume.size(), (int)NDIM);
      continue;
    }

    if (!is_backend_enabled(run.backend)) {
      if (rank == 0)
        printf("[SKIP] Backend '%s' not enabled in this build.\n",
               run.backend_name.c_str());
      continue;
    }

    std::array<size_t, NDIM> vol;
    std::copy(run.volume.begin(), run.volume.end(), vol.begin());

    BenchmarkLattice<double> lattice(vol, run.iters, run.spill_words,
                                     run.comm_mask, run.backend, cart_comm,
                                     nccl_comm, rank, num_ranks);

    // Retrieve singletons from gpu_type
    gpuStream_t c_stream = streams::bulk_stream();
    gpuStream_t h_stream = streams::halo_stream();

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);

    auto timer = [&](auto func) {
      gpuDeviceSynchronize();
      gpuEventRecord(start, 0);
      for (int i = 0; i < run.repeats; ++i)
        func();
      gpuEventRecord(stop, 0);
      gpuEventSynchronize(stop);
      float ms;
      gpuEventElapsedTime(&ms, start, stop);
      return (double)ms / run.repeats;
    };

    // Warmup
    for (int i = 0; i < run.warmup; ++i) {
      for (int d = 0; d < 2 * NDIM; ++d) {
        lattice.pack_direction(d, h_stream);
        lattice.start_exchange(d, h_stream);
      }
      lattice.compute_bulk(c_stream);
      for (int d = 0; d < 2 * NDIM; ++d) {
        lattice.wait_exchange(d, h_stream);
        lattice.unpack_direction(d, h_stream);
      }
    }

    // Phase 1: Compute Baseline
    double t_comp = timer([&]() { lattice.compute_bulk(c_stream); });

    // Phase 2: Comm Baseline
    double t_comm = timer([&]() {
      for (int d = 0; d < 2 * NDIM; ++d) {
        lattice.pack_direction(d, h_stream);
        lattice.start_exchange(d, h_stream);
        lattice.wait_exchange(d, h_stream);
        lattice.unpack_direction(d, h_stream);
      }
    });

    // Phase 3: Overlapped
    double t_overlap = timer([&]() {
      for (int d = 0; d < 2 * NDIM; ++d) {
        lattice.pack_direction(d, h_stream);
        lattice.start_exchange(d, h_stream);
      }
      lattice.compute_bulk(c_stream);
      for (int d = 0; d < 2 * NDIM; ++d) {
        lattice.wait_exchange(d, h_stream);
        lattice.unpack_direction(d, h_stream);
      }
    });

    report_metrics(rank, run, t_comp, t_comm, t_overlap);

    gpuEventDestroy(start);
    gpuEventDestroy(stop);
  }

  finalize_benchmark_env(cart_comm, nccl_comm);
  return 0;
}
