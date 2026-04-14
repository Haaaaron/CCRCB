cmake_minimum_required(VERSION 3.18)
project(ResourceContentionBenchmark LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Grace Hopper Architecture
set(CMAKE_CUDA_ARCHITECTURES 90)

# DISABLE global RDC to prevent STL "fatbinwrap" errors
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# --- 1. SET THE ROOT PATH ---
if(NOT NVHPC_ROOT)
    set(NVHPC_ROOT "/global/exafs/groups/gh/spack-1.0.2-20251008/opt/linux-neoverse_v2/nvhpc-25.9-wzo57cr7ncwj6refsynd5b4bkjtz34dt/Linux_aarch64/25.9")
endif()

# --- 2. FIND AND FORCE DEPENDENCIES ---
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ${NVHPC_ROOT}/comm_libs/nccl PATH_SUFFIXES include REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl HINTS ${NVHPC_ROOT}/comm_libs/nccl PATH_SUFFIXES lib lib64 REQUIRED)

find_path(NVSHMEM_INCLUDE_DIR NAMES nvshmem.h HINTS ${NVHPC_ROOT}/comm_libs/nvshmem PATH_SUFFIXES include REQUIRED)
find_library(NVSHMEM_LIBRARY NAMES nvshmem HINTS ${NVHPC_ROOT}/comm_libs/nvshmem PATH_SUFFIXES lib lib64 REQUIRED)

# --- 3. TARGET SETUP ---
add_executable(benchmark
    src/main.cpp
    src/comm_env.cpp
    src/gpu_type.cpp # <--- Ensures streams symbols are present
)

# --- 4. COMPILER & LINKER FLAGS ---
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # -cuda: parses <<<>>>
    # -gpu=nordc: stops compiler from trying to GPU-ize the C++ STL
    target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-cuda -gpu=nordc>)

    # Apply -cuda during linking to pull in NVHPC CUDA runtimes
    target_link_options(benchmark PRIVATE $<$<LINK_LANGUAGE:CXX>:-cuda>)

    # Suppress remarks
    target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress=cuda_compile>)
endif()

# Define USE_NCCL/USE_NVSHMEM by default as requested
option(USE_NCCL "Build with NCCL" ON)
option(USE_NVSHMEM "Build with NVSHMEM" ON)

target_compile_definitions(benchmark PRIVATE NDIM=${NDIM})

target_include_directories(benchmark PRIVATE
    src
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(benchmark PRIVATE MPI::MPI_CXX CUDA::cudart)

if(USE_NCCL)
    target_compile_definitions(benchmark PRIVATE USE_NCCL)
    target_include_directories(benchmark PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(benchmark PRIVATE ${NCCL_LIBRARY})
endif()

if(USE_NVSHMEM)
    target_compile_definitions(benchmark PRIVATE USE_NVSHMEM)
    target_include_directories(benchmark PRIVATE ${NVSHMEM_INCLUDE_DIR})
    target_link_libraries(benchmark PRIVATE ${NVSHMEM_LIBRARY} cuda rt)
endif()

