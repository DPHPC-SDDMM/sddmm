# https://code.visualstudio.com/docs/cpp/cmake-linux
# https://code.visualstudio.com/docs/cpp/config-linux
cmake_minimum_required(VERSION 3.22)
project(sddmm VERSION 0.1.0 LANGUAGES C CXX CUDA)

file(READ "./.cuda_arch" CUDA_ARCH)

if("${CUDA_ARCH}" STREQUAL "")
    message(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    message("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< :-) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    message(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    message("   .cuda_arch doesn't exists. Visit")
    message("   https://developer.nvidia.com/cuda-gpus")
    message("   to find your GPU and correct architecture number. Then create file .cuda_arch")
    message("   inside toplevel folder and write the architecture number in it WITHOUT! spaces")
    message("   or newlines")
    message("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    message(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (-: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    message("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    message("")

    ############################################################################################
    # if everything else fails, use this cmake-"code" snipsel and check the output. Maybe
    # you get a bunch of possibilities for compatible cuda architectures
    ############################################################################################
    # read .cuda_arch to get number for achitecture
    # include(FindCUDA/select_compute_arch)
    # CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    # string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    # string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    # string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    # message("========================= Available CUDA architecture =========================")
    # message("${CUDA_ARCH_LIST}")
    # message(">>>>>>>>>>>>>>>>>>>> This is a hack but it seems to work.. <<<<<<<<<<<<<<<<<<<<")
    # message("========================= --------------------------- =========================")

else()
    message("========================= --------------------------- =========================")
    message("Using cuda arch ${CUDA_ARCH}")
    message("===============================================================================")
    message("")
endif()


find_package(OpenMP)

find_package(CUDAToolkit REQUIRED)

set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(LIBRARY_OUTPUT_PATH  "${CMAKE_BINARY_DIR}")

include_directories("/usr/local/cuda/include" "${CMAKE_SOURCE_DIR}/matplotplusplus/source" "${CMAKE_SOURCE_DIR}/libs" "${CMAKE_SOURCE_DIR}/algos")
add_subdirectory(matplotplusplus)
add_subdirectory(src/algos)
add_subdirectory(src/libs)
add_subdirectory(src/data_structures)
add_subdirectory(tests)

# Mark only one
# add_compile_definitions(NONE)
# add_compile_definitions(SAMPLE_ALGO)
# add_compile_definitions(CSR_COO)
# add_compile_definitions(CUDA_SAMPLE)

add_executable(main src/main.cpp)
target_link_libraries(main OpenMP::OpenMP_CXX)
target_link_libraries(main CUDA::cudart)
target_link_libraries(main CUDA::cuda_driver)
# target_link_libraries(main CudaLib)
target_link_libraries(main matplot)
# target_link_libraries(main LibsLib)
target_link_libraries(main DataStructuresLib)
target_link_libraries(main AlgoLib)

add_executable(test1 tests/tests.cpp)
target_link_libraries(test1 OpenMP::OpenMP_CXX)
target_link_libraries(test1 CUDA::cudart)
target_link_libraries(test1 CUDA::cuda_driver)
# target_link_libraries(test1 CudaLib)
target_link_libraries(test1 matplot)
# target_link_libraries(main LibsLib)
target_link_libraries(test1 DataStructuresLib)
target_link_libraries(test1 AlgoLib)

# Check
# https://forums.developer.nvidia.com/t/nvlink-fatal-could-not-open-input-file-when-linking-with-empty-static-library/208517/2
# for [build] nvlink fatal   : Could not open input file '/usr/lib/x86_64-linux-gnu/libpthread.a'
#  => use the dummy-symbols method. Example: add libpthread.a inside /usr/lib/x86_64-linux-gnu/libpthread.a containing
#       /* Weak references in glibc that must be filled if glibc is to be
#          thread safe.  */
#       EXTERN(dummy_1)
#       EXTERN(dummy_2)
#       EXTERN(dummy_3)
add_executable(cuda_tiled_sddmm_tests tests/cuda_tiled_sddmm_tests.cpp)
target_link_libraries(cuda_tiled_sddmm_tests OpenMP::OpenMP_CXX)
target_link_libraries(cuda_tiled_sddmm_tests CUDA::cudart)
target_link_libraries(cuda_tiled_sddmm_tests CUDA::cuda_driver)
target_link_libraries(cuda_tiled_sddmm_tests CudaLib)
target_link_libraries(cuda_tiled_sddmm_tests matplot)
target_link_libraries(cuda_tiled_sddmm_tests DataStructuresLib)
target_link_libraries(cuda_tiled_sddmm_tests AlgoLib)

# https://linuxhint.com/install-latest-version-nvidia-cuda-ubuntu-22-04-lts/