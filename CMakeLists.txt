# https://code.visualstudio.com/docs/cpp/cmake-linux
# https://code.visualstudio.com/docs/cpp/config-linux
cmake_minimum_required(VERSION 3.22)
project(sddmm VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(OpenMP)

find_package(CUDAToolkit REQUIRED)

set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(LIBRARY_OUTPUT_PATH  "${CMAKE_BINARY_DIR}")

include_directories("${CMAKE_SOURCE_DIR}/matplotplusplus/source" "${CMAKE_SOURCE_DIR}/libs" "${CMAKE_SOURCE_DIR}/algos")
add_subdirectory(matplotplusplus)
add_subdirectory(src/algos)
add_subdirectory(src/libs)
add_subdirectory(src/data_structures)
add_subdirectory(tests)

# Mark only one
# add_compile_definitions(NONE)
# add_compile_definitions(SAMPLE_ALGO)
# add_compile_definitions(CSR_COO)
add_compile_definitions(CUDA_SAMPLE)

add_executable(main src/main.cpp)
target_link_libraries(main CUDA::cudart)
target_link_libraries(main CUDA::cuda_driver)
target_link_libraries(main CudaLib)
target_link_libraries(main OpenMP::OpenMP_CXX)
target_link_libraries(main matplot)
# target_link_libraries(main LibsLib)
target_link_libraries(main DataStructuresLib)
target_link_libraries(main AlgoLib)

add_executable(test1 tests/tests.cpp)
target_link_libraries(test1 CUDA::cudart)
target_link_libraries(test1 CUDA::cuda_driver)
target_link_libraries(test1 CudaLib)
target_link_libraries(test1 OpenMP::OpenMP_CXX)
target_link_libraries(test1 matplot)
# target_link_libraries(main LibsLib)
target_link_libraries(test1 DataStructuresLib)
target_link_libraries(test1 AlgoLib)