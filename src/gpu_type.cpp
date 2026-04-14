#include "gpu_type.h"

gpuStream_t &streams::halo_stream() {
  static gpuStream_t instance = [] {
    gpuStream_t stream;
    gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking);
    return stream;
  }();
  return instance;
}

gpuStream_t &streams::bulk_stream() {
  static gpuStream_t instance = [] {
    gpuStream_t stream;
    gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking);
    return stream;
  }();
  return instance;
}
