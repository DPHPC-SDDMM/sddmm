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
    message("   or newlines OR decimal number separators (e.g. '5.0 -> 50', '5,0 -> 50')")
    message("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    message(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> (-: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    message("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    message("")

    ############################################################################################
    # if everything else fails, use this cmake-"code" snipsel and check the output. Maybe
    # you get a bunch of possibilities for compatible cuda architectures
    ############################################################################################
    # read .cuda_arch to get number for achitecture
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    message("========================= Available CUDA architecture =========================")
    message("${INSTALLED_GPU_CCS_1}")
    message(">>>>>>>>>>>>>>>>>>>> This is a hack but it seems to work.. <<<<<<<<<<<<<<<<<<<<")
    message("========================= --------------------------- =========================")

else()
    message("========================= --------------------------- =========================")
    message("Using cuda arch ${CUDA_ARCH}")
    message("Using cuSPARSE lib ${CUDA_cusparse_LIBRARY}")
    message("===============================================================================")
    message("")
endif()

add_compile_options(-mavx2)
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)

set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(LIBRARY_OUTPUT_PATH  "${CMAKE_BINARY_DIR}")

include_directories("/usr/local/cuda/include" "/usr/include")

add_subdirectory(src/algos)
add_subdirectory(src/data_structures)
add_subdirectory(src/cuda_examples)
add_subdirectory(tests)
add_subdirectory(src/experiments)
add_subdirectory(src/matrix_file_reading)

# ======================================================================================
# NOTE: you MUST enable SDDMM_PARALLEL_CPU_ZERO_FILTER for all the tests to work!!!
add_compile_definitions(SDDMM_PARALLEL_CPU_ZERO_FILTER)
# ======================================================================================
# low precision requirements for floating point operations
# GPU tests will likely fail if this is not enabled because the GPU  is not built for
# precision (as it seems)
add_compile_definitions(USE_LOW_PRECISION)
# ======================================================================================

# Mark only one
# add_compile_definitions(NONE)
# add_compile_definitions(SAMPLE_ALGO)
# add_compile_definitions(CSR_COO)CUDA_cusparse_LIBRARY
# add_compile_definitions(USE_LOW_PRECISION)
# add_compile_definitions(GPU_CACHE_BENCHMARK)

# add_compile_definitions(USE_LOW_PRECISION)
# add_compile_definitions(UNROLLING_BENCHMARK)
# add_compile_definitions(UNROLLING_BENCHMARK_2)
# add_compile_definitions(MATRIX_MARKET_FILE_READ)

# add_compile_definitions(COMPARISON_BENCHMARK)
# add_compile_definitions(COMPARISON_BENCHMARK_SPARSITY)
add_compile_definitions(COMPARISON_BENCHMARK_SIZE)

add_executable(main src/main.cpp)
target_link_libraries(main OpenMP::OpenMP_CXX)
target_link_libraries(main CUDA::cudart)
target_link_libraries(main CUDA::cuda_driver)
target_link_libraries(main CudaLib)
target_link_libraries(main DataStructuresLib)
target_link_libraries(main AlgoLib)
target_link_libraries(main ExperimentLib)
target_link_libraries(main MatrixFileReadingLib)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET main PROPERTY CXX_STANDARD 17)

add_executable(test1 tests/tests.cpp)
target_link_libraries(test1 OpenMP::OpenMP_CXX)
target_link_libraries(test1 CUDA::cudart)
target_link_libraries(test1 CUDA::cuda_driver)
# target_link_libraries(test1 CudaLib)
target_link_libraries(test1 DataStructuresLib)
target_link_libraries(test1 AlgoLib)
set_property(TARGET test1 PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET test1 PROPERTY CXX_STANDARD 17)

# Check
# https://forums.developer.nvidia.com/t/nvlink-fatal-could-not-open-input-file-when-linking-with-empty-static-library/208517/2
# for [build] nvlink fatal   : Could not open input file '/usr/lib/x86_64-linux-gnu/libpthread.a'
#  => use the dummy-symbols method. Example: add libpthread.a inside /usr/lib/x86_64-linux-gnu/libpthread.a containing
#       /* Weak references in glibc that must be filled if glibc is to be
#          thread safe.  */
#       EXTERN(dummy_1)
#       EXTERN(dummy_2)
#       EXTERN(dummy_3)
add_executable(cuda_tests tests/cuda_tests.cpp)
target_link_libraries(cuda_tests OpenMP::OpenMP_CXX)
target_link_libraries(cuda_tests CUDA::cudart)
target_link_libraries(cuda_tests CUDA::cuda_driver)
target_link_libraries(cuda_tests CudaLib)
target_link_libraries(cuda_tests DataStructuresLib)
target_link_libraries(cuda_tests AlgoLib)
target_link_libraries(cuda_tests ${CUDA_cusparse_LIBRARY})
set_property(TARGET cuda_tests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET cuda_tests PROPERTY CXX_STANDARD 17)

# https://linuxhint.com/install-latest-version-nvidia-cuda-ubuntu-22-04-lts/

add_executable(cuda_example_tests tests/cuda_example_tests.cpp)
target_link_libraries(cuda_example_tests OpenMP::OpenMP_CXX)
target_link_libraries(cuda_example_tests CUDA::cudart)
target_link_libraries(cuda_example_tests CUDA::cuda_driver)
target_link_libraries(cuda_example_tests CudaExpCu)
target_link_libraries(cuda_example_tests CudaExp)
target_link_libraries(cuda_example_tests DataStructuresLib)
target_link_libraries(cuda_example_tests AlgoLib)
target_link_libraries(cuda_example_tests ${CUDA_cusparse_LIBRARY})
set_property(TARGET cuda_example_tests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET cuda_example_tests PROPERTY CXX_STANDARD 17)

add_executable(sml2_tests tests/sm_l2_tests.cpp)
target_link_libraries(sml2_tests CUDA::cudart)
target_link_libraries(sml2_tests CUDA::cuda_driver)
target_link_libraries(sml2_tests CudaLib)
target_link_libraries(sml2_tests DataStructuresLib)
target_link_libraries(sml2_tests AlgoLib)
target_link_libraries(sml2_tests ${CUDA_cusparse_LIBRARY})
set_property(TARGET sml2_tests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET sml2_tests PROPERTY CXX_STANDARD 17)

add_executable(random_tests tests/random_tests.cpp)
target_link_libraries(random_tests OpenMP::OpenMP_CXX)
target_link_libraries(random_tests CUDA::cudart)
target_link_libraries(random_tests CUDA::cuda_driver)
target_link_libraries(random_tests DataStructuresLib)
target_link_libraries(random_tests AlgoLib)
target_link_libraries(random_tests ${CUDA_cusparse_LIBRARY})
set_property(TARGET random_tests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET random_tests PROPERTY CXX_STANDARD 17)

add_executable(file_storage_tests tests/file_storage_tests.cpp)
target_link_libraries(file_storage_tests OpenMP::OpenMP_CXX)
target_link_libraries(file_storage_tests CUDA::cudart)
target_link_libraries(file_storage_tests CUDA::cuda_driver)
target_link_libraries(file_storage_tests DataStructuresLib)
target_link_libraries(file_storage_tests AlgoLib)
target_link_libraries(file_storage_tests ${CUDA_cusparse_LIBRARY})
set_property(TARGET file_storage_tests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET file_storage_tests PROPERTY CXX_STANDARD 17)
