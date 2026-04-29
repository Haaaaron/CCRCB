#include "benchmark_lattice.h"
#include "comm_env.h"
#include "config_parser.h"
#include "gpu_type.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <vector>

#ifndef NDIM
#define NDIM 4
#endif

void snoop_system_info(std::ostream &out, int rank, int num_ranks,
                       MPI_Comm cart_comm) {
    if (rank != 0)
        return;

    out << "==========================================================\n";
    out << " SYSTEM ARCHITECTURE INFO\n";
    out << "==========================================================\n";

    // 1. Topology Info
    out << "Ranks: " << num_ranks << "\n";
    int mpi_dims[NDIM], periods[NDIM], coords[NDIM];
    MPI_Cart_get(cart_comm, NDIM, mpi_dims, periods, coords);
    out << "Cartesian Grid: (";
    for (int i = 0; i < NDIM; ++i)
        out << mpi_dims[i] << (i == NDIM - 1 ? ")\n" : ", ");

    // 2. Software Stack Info
    out << "SOFTWARE STACK:\n";
    char mpi_ver[MPI_MAX_LIBRARY_VERSION_STRING];
    int mpi_ver_len;
    MPI_Get_library_version(mpi_ver, &mpi_ver_len);
    out << "  MPI: " << mpi_ver << "\n";

    // CUDA/HIP Runtime Version
    int runtime_ver = 0;
#if defined(USE_HIP)
    hipRuntimeGetVersion(&runtime_ver);
    out << "  HIP Runtime: " << runtime_ver << "\n";
#else
    cudaRuntimeGetVersion(&runtime_ver);
    out << "  CUDA Runtime: " << runtime_ver / 1000 << "."
        << (runtime_ver % 1000) / 10 << "\n";
#endif

    // NCCL Version
#ifdef USE_NCCL
    int nccl_ver = 0;
    ncclGetVersion(&nccl_ver);
    int n_major = nccl_ver / 10000;
    int n_minor = (nccl_ver % 10000) / 100;
    int n_patch = nccl_ver % 100;
    out << "  NCCL: " << n_major << "." << n_minor << "." << n_patch << "\n";
#endif

    // NVSHMEM Version
#ifdef USE_NVSHMEM
    int nv_major, nv_minor;
    nvshmem_info_get_version(&nv_major, &nv_minor);
    out << "  NVSHMEM: " << nv_major << "." << nv_minor << "\n";
#endif

    // 3. Hardware / Interconnect Info (Snoop via sysfs)
    out << "INTERCONNECT:\n";

    // 3a. InfiniBand (Mellanox/NVIDIA)
    std::string ib_path = "/sys/class/infiniband";
    if (std::filesystem::exists(ib_path)) {
        out << "  InfiniBand (Mellanox/NVIDIA):\n";
        auto read_sysfs = [&](const std::string &fname) {
            std::ifstream f(ib_path + "/mlx5_0/" + fname);
            std::string s;
            std::getline(f, s);
            return s;
        };
        for (const auto &entry : std::filesystem::directory_iterator(ib_path)) {
            out << "    Device: " << entry.path().filename().string() << "\n";
            out << "      HCA Type: " << read_sysfs("hca_type") << "\n";
            out << "      FW Ver:   " << read_sysfs("fw_ver") << "\n";
            std::string port_path = entry.path().string() + "/ports/1/rate";
            if (std::filesystem::exists(port_path)) {
                std::ifstream f(port_path);
                std::string rate;
                std::getline(f, rate);
                out << "      Rate:     " << rate << "\n";
            }
        }
    }

    // 3b. HPE Slingshot (CXI/Cassini)
    std::string cxi_path = "/sys/class/cxi";
    if (std::filesystem::exists(cxi_path)) {
        out << "  HPE Slingshot (Cassini/CXI):\n";
        for (const auto &entry :
             std::filesystem::directory_iterator(cxi_path)) {
            std::string dev_name = entry.path().filename().string();
            out << "    Device: " << dev_name << "\n";
            std::string hsn_name = "hsn" + dev_name.substr(3); // cxi0 -> hsn0
            std::string net_speed_path =
                "/sys/class/net/" + hsn_name + "/speed";
            if (std::filesystem::exists(net_speed_path)) {
                std::ifstream f(net_speed_path);
                std::string speed;
                std::getline(f, speed);
                out << "      Interface: " << hsn_name << "\n";
                out << "      Speed:     " << speed << " Mbps\n";
            }
        }
    }

    if (!std::filesystem::exists(ib_path) &&
        !std::filesystem::exists(cxi_path)) {
        out << "  No InfiniBand or Slingshot devices detected in /sys/class.\n";
    }

    out << "==========================================================\n\n";
}

struct TimingResult {
    double t_comp;
    double t_comm;
    double t_total_ovl;
    double t_comp_ovl;
    double t_comm_ovl;
};

void report_metrics(std::ostream &out, int rank, const RunConfig &cfg,
                    const TimingResult &res, MPI_Comm cart_comm,
                    gcclComm_t nccl_comm, int num_ranks) {
    if (rank == 0) {
        double t_serial = res.t_comp + res.t_comm;
        double t_max = std::max(res.t_comp, res.t_comm);

        double speedup = t_serial / res.t_total_ovl;
        double contention = res.t_total_ovl / t_max;

        // Retrieve topology
        int mpi_dims[NDIM], periods[NDIM], coords[NDIM];
        MPI_Cart_get(cart_comm, NDIM, mpi_dims, periods, coords);

        // Calculate volumes and message sizes
        size_t local_volume = 1;
        std::array<size_t, NDIM> local_vols;
        for (int i = 0; i < NDIM; ++i) {
            local_vols[i] = cfg.volume[i] / mpi_dims[i];
            local_volume *= local_vols[i];
        }

        // Format strings for local and global volume
        std::string vol_str = "(";
        std::string global_vol_str = "(";
        std::string grid_str = "(";
        for (size_t i = 0; i < cfg.volume.size(); ++i) {
            vol_str += std::to_string(local_vols[i]);
            global_vol_str += std::to_string(cfg.volume[i]);
            grid_str += std::to_string(mpi_dims[i]);
            if (i < cfg.volume.size() - 1) {
                vol_str += ", ";
                global_vol_str += ", ";
                grid_str += ", ";
            }
        }
        vol_str += ")";
        global_vol_str += ")";
        grid_str += ")";

        // Message sizes for all 2*NDIM directions
        std::string msg_size_str = "";
        for (size_t i = 0; i < 2 * NDIM; ++i) {
            int d = i / 2;
            bool is_fwd = i % 2;
            int src, dst, p;
            MPI_Cart_shift(cart_comm, d, 1, &src, &dst);
            p = is_fwd ? dst : src;

            msg_size_str += "d" + std::to_string(d) + (is_fwd ? "-f:" : "-b:");

            if (p != rank && p != MPI_PROC_NULL) {
                size_t f_vol = local_volume / local_vols[d];
                size_t msg_bytes = f_vol * sizeof(double);
                if (msg_bytes >= 1024 * 1024) {
                    msg_size_str +=
                        std::to_string(msg_bytes / (1024 * 1024)) + "MB";
                } else if (msg_bytes >= 1024) {
                    msg_size_str += std::to_string(msg_bytes / 1024) + "KB";
                } else {
                    msg_size_str += std::to_string(msg_bytes) + "B";
                }
            } else {
                msg_size_str += "SKIP";
            }
            if (i < 2 * NDIM - 1)
                msg_size_str += ", ";
        }

        char buffer[2048];
        snprintf(
            buffer, sizeof(buffer),
            "\n==========================================================\n"
            " CONFIGURATION:\n"
            "  Backend:    %-10s | Dim: %dD\n"
            "  Local Vol:  %-15s | Global Vol: %s\n"
            "  MPI Grid:   %-15s | Msg Size:   %s\n"
            "  Iters(Time):%-15d | Math_Load(AI):%d\n"
            "  Warmup:     %-15d | Repeats:    %d\n"
            "----------------------------------------------------------\n"
            " PERFORMANCE:\n"
            "  T_compute:  %8.3f ms (Isolated) | %8.3f ms (Overlap)\n"
            "  T_comm:     %8.3f ms (Isolated) | %8.3f ms (Overlap)\n"
            "  T_serial:   %8.3f ms (Sum)      | T_total: %8.3f ms\n"
            "----------------------------------------------------------\n"
            " ANALYSIS:\n"
            "  Overlap Speedup:    %6.2fx\n"
            "  Contention Factor:  %6.2fx\n"
            "==========================================================\n",
            cfg.backend_name.c_str(), (int)NDIM, vol_str.c_str(),
            global_vol_str.c_str(), grid_str.c_str(), msg_size_str.c_str(),
            cfg.iters, cfg.math_per_load, cfg.warmup, cfg.repeats, res.t_comp,
            res.t_comp_ovl, res.t_comm, res.t_comm_ovl, t_serial, res.t_total_ovl, speedup,
            contention);
        out << buffer << std::flush;

        // Detailed Workload Calculation
        double ops_per_iter = 2.0;
        double total_ops = (double)local_volume * cfg.iters * ops_per_iter;
        double base_bytes = (double)local_volume * 2 * sizeof(double);
        double mem_fetches =
            (double)cfg.iters /
            (double)(cfg.math_per_load > 0 ? cfg.math_per_load : 1);
        double loop_bytes = (double)local_volume * mem_fetches * sizeof(double);

        double ops = total_ops;
        double bytes = base_bytes + loop_bytes;

        double gflops = (ops / (res.t_comp / 1000.0)) / 1e9;
        double bw_gbs = (bytes / (res.t_comp / 1000.0)) / 1e9;
        double ai = ops / bytes;

        snprintf(buffer, sizeof(buffer),
                 "  Est. Compute: %8.2f GFLOPS\n"
                 "  Est. Bandwidth:%8.2f GB/s\n"
                 "  Actual AI:     %8.2f Ops/Byte\n"
                 "==========================================================\n",
                 gflops, bw_gbs, ai);
        out << buffer << std::flush;
    }
}

TimingResult run_timing_phase(BenchmarkLattice<double> &lattice, const RunConfig &cfg, int rank) {
    gpuStream_t c_stream = streams::bulk_stream();
    gpuStream_t h_stream = streams::halo_stream();

    auto timer = [&](auto func, gpuStream_t stream = 0, bool sync = true) {
        if (sync)
            gpuDeviceSynchronize();
        gpuEvent_t s, e;
        gpuEventCreate(&s);
        gpuEventCreate(&e);

        gpuEventRecord(s, stream);
        for (int i = 0; i < cfg.repeats; ++i) {
            func();
        }
        gpuEventRecord(e, stream);

        if (sync) {
            gpuEventSynchronize(e);
            float ms;
            gpuEventElapsedTime(&ms, s, e);
            double avg = (double)ms / cfg.repeats;
            gpuEventDestroy(s);
            gpuEventDestroy(e);
            return avg;
        } else {
            gpuEventDestroy(s);
            gpuEventDestroy(e);
            return 0.0;
        }
    };

    // Warmup
    gpuRangePush("Warmup");
    lattice.sync_all();
    if (rank == 0)
        printf("  -> Warmup (%d iters): ", cfg.warmup);
    fflush(stdout);
    for (int i = 0; i < cfg.warmup; ++i) {
        if (rank == 0 && (i % 5 == 0 || i == cfg.warmup - 1)) {
            printf("\r  -> Warmup (%d/%d)... ", i + 1, cfg.warmup);
            fflush(stdout);
        }
        for (int d = 0; d < 2 * NDIM; ++d) {
            lattice.pack_direction(d, h_stream);
            lattice.start_exchange(d, h_stream);
        }
        lattice.compute_bulk(c_stream);
        for (int d = 0; d < 2 * NDIM; ++d) {
            lattice.wait_exchange(d, h_stream);
            lattice.unpack_direction(d, h_stream);
        }
        gpuDeviceSynchronize();
    }
    if (rank == 0)
        printf("Done.   \n");
    gpuRangePop();

    // Phase 1: Compute Baseline
    gpuRangePush("Phase 1: Compute Baseline");
    lattice.sync_all();
    if (rank == 0)
        printf("  -> Phase 1: Compute Baseline... ");
    fflush(stdout);
    double t_comp = timer([&]() { lattice.compute_bulk(c_stream); }, c_stream);
    if (rank == 0)
        printf("Done (%.3f ms)\n", t_comp);
    gpuRangePop();

    // Phase 2: Comm Baseline
    gpuRangePush("Phase 2: Comm Baseline");
    lattice.sync_all();
    if (rank == 0)
        printf("  -> Phase 2: Comm Baseline... ");
    fflush(stdout);
    double t_comm = timer(
        [&]() {
            for (int d = 0; d < 2 * NDIM; ++d) {
                lattice.pack_direction(d, h_stream);
                lattice.start_exchange(d, h_stream);
            }
            for (int d = 0; d < 2 * NDIM; ++d) {
                lattice.wait_exchange(d, h_stream);
                lattice.unpack_direction(d, h_stream);
            }
        },
        h_stream);
    if (rank == 0)
        printf("Done (%.3f ms)\n", t_comm);
    gpuRangePop();

    // Phase 3: Overlapped
    gpuRangePush("Phase 3: Overlap");
    lattice.sync_all();
    if (rank == 0)
        printf("  -> Phase 3: Overlap (%d repeats): ", cfg.repeats);
    fflush(stdout);
    double t_comp_ovl = 0;
    double t_comm_ovl = 0;
    double t_total_ovl = 0;

    for (int i = 0; i < cfg.repeats; ++i) {
        if (rank == 0) {
            printf("\r  -> Phase 3: Overlap (%d/%d)... ", i + 1, cfg.repeats);
            fflush(stdout);
        }
        gpuDeviceSynchronize();
        gpuEvent_t ev_total_start, ev_total_stop, ev_comp_start,
            ev_comp_stop, ev_comm_start, ev_comm_stop;
        gpuEventCreate(&ev_total_start);
        gpuEventCreate(&ev_total_stop);
        gpuEventCreate(&ev_comm_start);
        gpuEventCreate(&ev_comp_start);
        gpuEventCreate(&ev_comp_stop);
        gpuEventCreate(&ev_comm_stop);

        gpuEventRecord(ev_total_start, h_stream);
        gpuEventRecord(ev_comm_start, h_stream);
        for (int d = 0; d < 2 * NDIM; ++d) {
            lattice.pack_direction(d, h_stream);
            lattice.start_exchange(d, h_stream);
        }
        gpuEventRecord(ev_comp_start, c_stream);
        lattice.compute_bulk(c_stream);
        gpuEventRecord(ev_comp_stop, c_stream);
        for (int d = 0; d < 2 * NDIM; ++d) {
            lattice.wait_exchange(d, h_stream);
            lattice.unpack_direction(d, h_stream);
        }
        gpuEventRecord(ev_comm_stop, h_stream);
        gpuStreamWaitEvent(h_stream, ev_comp_stop, 0);
        gpuEventRecord(ev_total_stop, h_stream);

        gpuDeviceSynchronize();
        float ms_total, ms_comp, ms_comm;
        gpuEventElapsedTime(&ms_total, ev_total_start, ev_total_stop);
        gpuEventElapsedTime(&ms_comp, ev_comp_start, ev_comp_stop);
        gpuEventElapsedTime(&ms_comm, ev_comm_start, ev_comm_stop);

        t_total_ovl += ms_total;
        t_comp_ovl += ms_comp;
        t_comm_ovl += ms_comm;

        gpuEventDestroy(ev_total_start);
        gpuEventDestroy(ev_total_stop);
        gpuEventDestroy(ev_comm_start);
        gpuEventDestroy(ev_comp_start);
        gpuEventDestroy(ev_comp_stop);
        gpuEventDestroy(ev_comm_stop);
    }
    t_total_ovl /= cfg.repeats;
    t_comp_ovl /= cfg.repeats;
    t_comm_ovl /= cfg.repeats;

    if (rank == 0)
        printf("Done (Total: %.3f ms)\n", t_total_ovl);
    gpuRangePop();

    return {t_comp, t_comm, t_total_ovl, t_comp_ovl, t_comm_ovl};
}

void run_ncu_profiling_pass(BenchmarkLattice<double> &lattice, const RunConfig &cfg, int rank, int run_idx) {
    std::string tag = cfg.backend_name + "_Run" + std::to_string(run_idx);
    if (rank == 0) {
        printf("  -> [PROFILING] NCU Isolated Pass [%s]...\n", tag.c_str());
        fflush(stdout);
    }

    gpuStream_t p_stream = streams::bulk_stream();
    lattice.sync_all();
    
    gpuProfilerStart();

    // Compute
    gpuRangePush(("Profiling_Isolated_Compute_" + tag).c_str());
    lattice.compute_bulk(p_stream, std::min(cfg.iters, 2));
    gpuStreamSynchronize(p_stream);
    gpuRangePop();

    // Pack
    gpuRangePush(("Profiling_Isolated_Pack_" + tag).c_str());
    lattice.pack_direction(0, p_stream);
    gpuStreamSynchronize(p_stream);
    gpuRangePop();

    // Unpack
    gpuRangePush(("Profiling_Isolated_Unpack_" + tag).c_str());
    lattice.unpack_direction(0, p_stream);
    gpuStreamSynchronize(p_stream);
    gpuRangePop();

    gpuProfilerStop();
    lattice.sync_all();
}

void run_nsys_profiling_pass(BenchmarkLattice<double> &lattice, const RunConfig &cfg, int rank, int run_idx) {
    std::string tag = cfg.backend_name + "_Run" + std::to_string(run_idx);
    if (rank == 0) {
        printf("  -> [PROFILING] NSYS Overlap Pass [%s]...\n", tag.c_str());
        fflush(stdout);
    }

    gpuStream_t p_stream = streams::bulk_stream();
    gpuStream_t h_stream = streams::halo_stream();
    lattice.sync_all();

    gpuProfilerStart();
    
    // Isolated Compute Baseline
    gpuRangePush(("Profiling_Isolated_Compute_" + tag).c_str());
    lattice.compute_bulk(p_stream, 100);
    gpuDeviceSynchronize();
    gpuRangePop();

    lattice.sync_all();

    // Overlapped State
    gpuRangePush(("Profiling_Overlapped_" + tag).c_str());

    for (int p_iter = 0; p_iter < 100; ++p_iter) {
        for (int d = 0; d < 2 * NDIM; ++d) {
            lattice.pack_direction(d, h_stream);
            lattice.start_exchange(d, h_stream);
        }
        lattice.compute_bulk(p_stream);
        for (int d = 0; d < 2 * NDIM; ++d) {
            lattice.wait_exchange(d, h_stream);
            lattice.unpack_direction(d, h_stream);
        }
        gpuDeviceSynchronize();
    }

    gpuRangePop();
    gpuProfilerStop();
    lattice.sync_all();
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

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <config_file> [output_folder]\n";
        }
        finalize_benchmark_env(cart_comm, nccl_comm);
        return 1;
    }

    std::string config_file = argv[1];
    std::string output_dir;

    if (argc >= 3) {
        output_dir = argv[2];
    } else {
        auto now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::stringstream ss;
        ss << "results_" << num_ranks << "_ranks_"
           << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
        output_dir = ss.str();
    }

    if (rank == 0) {
        std::filesystem::create_directories(output_dir);
        printf("Output Directory: %s\n", output_dir.c_str());

        std::ofstream info_file(output_dir + "/system_info.txt");
        snoop_system_info(info_file, rank, num_ranks, cart_comm);
        info_file.close();
    }
    MPI_Barrier(cart_comm);

    std::ofstream res_file;
    if (rank == 0 && !getenv("CCRCB_PROFILING_ONLY")) {
        res_file.open(output_dir + "/results.txt");
    }

    auto runs = parse_run_file(config_file, rank);

    for (size_t run_idx = 0; run_idx < runs.size(); ++run_idx) {
        const auto &run = runs[run_idx];
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

        int mpi_dims[NDIM], periods[NDIM], coords[NDIM];
        MPI_Cart_get(cart_comm, NDIM, mpi_dims, periods, coords);

        std::array<size_t, NDIM> local_vol;
        bool possible = true;
        for (int i = 0; i < NDIM; ++i) {
            if (run.volume[i] % mpi_dims[i] != 0) {
                if (rank == 0) {
                    printf(
                        "[SKIP] Global volume dimension %d (%zu) not divisible "
                        "by grid dimension %d\n",
                        i, run.volume[i], mpi_dims[i]);
                }
                possible = false;
                break;
            }
            local_vol[i] = run.volume[i] / mpi_dims[i];
        }
        if (!possible)
            continue;

        if (rank == 0) {
            printf("[RUN %zu/%zu] Backend: %s, Iters: %d, Global Vol: (",
                   run_idx + 1, runs.size(), run.backend_name.c_str(),
                   run.iters);
            for (size_t i = 0; i < NDIM; ++i)
                printf("%zu%s", run.volume[i], (i == NDIM - 1 ? ")\n" : ", "));
        }

        BenchmarkLattice<double> lattice(
            local_vol, run.iters, run.math_per_load, run.comm_mask, run.backend,
            cart_comm, nccl_comm, rank, num_ranks);

        // ---------------------------------------------------------------------
        // 1. Profiling Passes
        // ---------------------------------------------------------------------

        if (getenv("CCRCB_NCU_PASS")) {
            run_ncu_profiling_pass(lattice, run, rank, run_idx);
        }

        if (getenv("CCRCB_NSYS_PASS")) {
            run_nsys_profiling_pass(lattice, run, rank, run_idx);
        }

        // ---------------------------------------------------------------------
        // 2. Timing Phase
        // ---------------------------------------------------------------------

        if (!getenv("CCRCB_PROFILING_ONLY")) {
            TimingResult res = run_timing_phase(lattice, run, rank);
            report_metrics(res_file, rank, run, res, cart_comm, nccl_comm, num_ranks);
        }

        lattice.sync_all();
    }

    if (rank == 0 && !getenv("CCRCB_PROFILING_ONLY"))
        res_file.close();

    finalize_benchmark_env(cart_comm, nccl_comm);
    return 0;
}
