cmake_minimum_required(VERSION 3.9.1)
project(sddmm VERSION 0.1.0 LANGUAGES C CXX)

set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(LIBRARY_OUTPUT_PATH  "${CMAKE_BINARY_DIR}")

include_directories("${CMAKE_SOURCE_DIR}/libs" "${CMAKE_SOURCE_DIR}/matplotplusplus/source")
add_subdirectory(matplotplusplus)
add_subdirectory(src/algos)

add_executable(basic_main src/main.cpp)
target_link_libraries(basic_main AlgoLib matplot)

# add a new target here
add_executable(plot_sample src/plot_sample.cpp)
target_link_libraries(plot_sample AlgoLib matplot)