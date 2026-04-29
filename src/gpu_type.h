#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// =====================================================================
// 1. BASE GPU ABSTRACTION (CUDA/HIP)
// =====================================================================

#if defined(__HIPCC__) || defined(USE_HIP)
#include <hip/hip_runtime.h>
using gpuError_t = hipError_t;
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#else
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
using gpuError_t = cudaError_t;
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#endif

// Define gpuAssert FIRST so it is visible to the GPU_CHECK macro
inline void gpuAssert(gpuError_t code, const char *cmd, const char *file,
                      int line, bool abort = true) {
  if (code != gpuSuccess) {
    fprintf(stderr, "GPU Error [%s]: %s\n%s : %d\n", cmd,
            gpuGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// Ensure the macro name is exactly GPU_CHECK
#define GPU_CHECK(cmd)                                                         \
  do {                                                                         \
    auto code = cmd;                                                           \
    gpuAssert(code, #cmd, __FILE__, __LINE__);                                 \
  } while (0)

#if defined(__HIPCC__) || defined(USE_HIP)
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuStreamNonBlocking hipStreamNonBlocking

#define gpuMalloc(ptr, size) GPU_CHECK(hipMalloc((void **)(ptr), size))
#define gpuFree(ptr) GPU_CHECK(hipFree(ptr))
#define gpuMemcpy(dst, src, count, kind)                                       \
  GPU_CHECK(hipMemcpy(dst, src, count, kind))
#define gpuMemcpyAsync(dst, src, count, k, s)                                  \
  GPU_CHECK(hipMemcpyAsync(dst, src, count, k, s))
#define gpuMemset(dst, val, count) GPU_CHECK(hipMemset(dst, val, count))

#define gpuStreamCreate(stream) GPU_CHECK(hipStreamCreate(stream))
#define gpuStreamCreateWithFlags(s, flags)                                     \
  GPU_CHECK(hipStreamCreateWithFlags(s, flags))
#define gpuStreamDestroy(stream) GPU_CHECK(hipStreamDestroy(stream))
#define gpuStreamSynchronize(stream) GPU_CHECK(hipStreamSynchronize(stream))
#define gpuStreamWaitEvent(stream, event, flags)                               \
  GPU_CHECK(hipStreamWaitEvent(stream, event, flags))
#define gpuDeviceSynchronize() GPU_CHECK(hipDeviceSynchronize())

#define gpuGetDeviceCount(count) GPU_CHECK(hipGetDeviceCount(count))
#define gpuGetDeviceProperties(prop, dev) GPU_CHECK(hipGetDeviceProperties(prop, dev))
#define gpuDeviceProp_t hipDeviceProp_t

#define gpuEventCreate(event) GPU_CHECK(hipEventCreate(event))
#define gpuEventRecord(event, stream) GPU_CHECK(hipEventRecord(event, stream))
#define gpuEventSynchronize(event) GPU_CHECK(hipEventSynchronize(event))
#define gpuEventDestroy(event) GPU_CHECK(hipEventDestroy(event))

inline void gpuEventElapsedTime(float *ms, gpuEvent_t start, gpuEvent_t stop) {
  GPU_CHECK(hipEventElapsedTime(ms, start, stop));
}

// Profiler stubs for HIP (can be extended if needed)
inline void gpuProfilerStart() {}
inline void gpuProfilerStop() {}
inline void gpuRangePush(const char* label) {}
inline void gpuRangePop() {}

#else // CUDA
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuStreamNonBlocking cudaStreamNonBlocking

#define gpuMalloc(ptr, size) GPU_CHECK(cudaMalloc((void **)(ptr), size))
#define gpuFree(ptr) GPU_CHECK(cudaFree(ptr))
#define gpuMemcpy(dst, src, count, kind)                                       \
  GPU_CHECK(cudaMemcpy(dst, src, count, kind))
#define gpuMemcpyAsync(dst, src, count, k, s)                                  \
  GPU_CHECK(cudaMemcpyAsync(dst, src, count, k, s))
#define gpuMemset(dst, val, count) GPU_CHECK(cudaMemset(dst, val, count))

#define gpuStreamCreate(stream) GPU_CHECK(cudaStreamCreate(stream))
#define gpuStreamCreateWithFlags(s, flags)                                     \
  GPU_CHECK(cudaStreamCreateWithFlags(s, flags))
#define gpuStreamDestroy(stream) GPU_CHECK(cudaStreamDestroy(stream))
#define gpuStreamSynchronize(stream) GPU_CHECK(cudaStreamSynchronize(stream))
#define gpuStreamWaitEvent(stream, event, flags)                               \
  GPU_CHECK(cudaStreamWaitEvent(stream, event, flags))
#define gpuDeviceSynchronize() GPU_CHECK(cudaDeviceSynchronize())

#define gpuGetDeviceCount(count) GPU_CHECK(cudaGetDeviceCount(count))
#define gpuGetDeviceProperties(prop, dev) GPU_CHECK(cudaGetDeviceProperties(prop, dev))
#define gpuDeviceProp_t cudaDeviceProp

#define gpuEventCreate(event) GPU_CHECK(cudaEventCreate(event))
#define gpuEventRecord(event, stream) GPU_CHECK(cudaEventRecord(event, stream))
#define gpuEventSynchronize(event) GPU_CHECK(cudaEventSynchronize(event))
#define gpuEventDestroy(event) GPU_CHECK(cudaEventDestroy(event))

inline void gpuEventElapsedTime(float *ms, gpuEvent_t start, gpuEvent_t stop) {
  GPU_CHECK(cudaEventElapsedTime(ms, start, stop));
}

// Profiler / NVTX for CUDA
inline void gpuProfilerStart() { cudaProfilerStart(); }
inline void gpuProfilerStop() { cudaProfilerStop(); }
inline void gpuRangePush(const char* label) { nvtxRangePushA(label); }
inline void gpuRangePop() { nvtxRangePop(); }

#endif

namespace streams {
gpuStream_t &halo_stream();
gpuStream_t &bulk_stream();
} // namespace streams

// =====================================================================
// 2. COMMUNICATION BACKENDS
// =====================================================================

#ifdef USE_NCCL
#if defined(USE_HIP)
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif

#define gcclComm_t ncclComm_t
#define gcclSuccess ncclSuccess
#define gcclGetErrorString ncclGetErrorString

#define GCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "GCCL Error: %s at %s:%d\n", ncclGetErrorString(res),    \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define gcclGroupStart() GCCL_CHECK(ncclGroupStart())
#define gcclGroupEnd() GCCL_CHECK(ncclGroupEnd())
#define gcclSend(ptr, count, type, peer, comm, stream)                         \
  GCCL_CHECK(ncclSend(ptr, count, type, peer, comm, stream))
#define gcclRecv(ptr, count, type, peer, comm, stream)                         \
  GCCL_CHECK(ncclRecv(ptr, count, type, peer, comm, stream))
#define gcclCommInitRank(comm, nranks, id, rank)                               \
  GCCL_CHECK(ncclCommInitRank(comm, nranks, id, rank))
#define gcclCommDestroy(comm) GCCL_CHECK(ncclCommDestroy(comm))

template <typename T> struct gccl_type;
template <> struct gccl_type<double> {
  static constexpr ncclDataType_t value = ncclDouble;
};
template <> struct gccl_type<float> {
  static constexpr ncclDataType_t value = ncclFloat;
};
template <> struct gccl_type<uint8_t> {
  static constexpr ncclDataType_t value = ncclUint8;
};
#else
typedef void *gcclComm_t;
#define gcclGroupStart()                                                       \
  do {                                                                         \
  } while (0)
#define gcclGroupEnd()                                                         \
  do {                                                                         \
  } while (0)
#define gcclSend(...)                                                          \
  do {                                                                         \
  } while (0)
#define gcclRecv(...)                                                          \
  do {                                                                         \
  } while (0)
#endif

// =====================================================================
// 3. NVSHMEM BACKEND
// =====================================================================

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#define gpuNvshmemMalloc(size) nvshmem_malloc(size)
#define gpuNvshmemFree(ptr) nvshmem_free(ptr)
#else
#define gpuNvshmemMalloc(size) (nullptr)
#define gpuNvshmemFree(ptr)                                                    \
  do {                                                                         \
  } while (0)
#define nvshmemx_putmem_nbi_on_stream(...)                                     \
  do {                                                                         \
  } while (0)
#define nvshmemx_quiet_on_stream(...)                                          \
  do {                                                                         \
  } while (0)
#endif
