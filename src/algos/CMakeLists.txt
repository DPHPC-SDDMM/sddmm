cmake_minimum_required(VERSION 3.22)

file(GLOB SOURCES
    ./*.cpp
    ./cpu_sddmm/*.cpp
    ./sample/*.cpp
    ./cuda_sddmm/*.cpp
    ./sm-l2-sddmm/*.cpp
    ./cuda_mat_mult/*.cpp
)

add_library(AlgoLib STATIC ${SOURCES})

file(GLOB CU_SOURCES
    ./cuda_sddmm/*.cuh
    ./cuda_sddmm/*.cu
    ./cuda_sample/*.cuh
    ./cuda_sample/*.cu
    ./sm-l2-sddmm/*.cuh
    ./sm-l2-sddmm/*.cu
    ./cuda_mat_mult/*.cuh
    ./cuda_mat_mult/*.cu
)

add_library(CudaLib STATIC ${CU_SOURCES})
set_target_properties(CudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CudaLib PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
