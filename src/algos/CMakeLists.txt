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

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
message("========================= Available CUDA architecture =========================")
message("${CUDA_ARCH_LIST}")
message(">>>>>>>>>>>>>>>>>>>> This is a hack but it seems to work.. <<<<<<<<<<<<<<<<<<<<")
message("========================= --------------------------- =========================")

set_property(TARGET CudaLib PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})