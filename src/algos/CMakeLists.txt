cmake_minimum_required(VERSION 3.22)

file(GLOB SOURCES
    ./*.cpp
)

add_library(AlgoLib STATIC ${SOURCES})

file(GLOB CU_SOURCES
    ./*.cuh
    ./*.cu
)

add_library(CudaLib STATIC ${CU_SOURCES})
set_target_properties(CudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)