cmake_minimum_required(VERSION 3.8)
project(sddmm VERSION 0.1.0 LANGUAGES C CXX)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(PythonLibs REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES})

# find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})
# target_link_libraries(<your exe or lib> ${PYTHON_LIBRARIES})

add_subdirectory(src)
add_executable(main ./src/main.cpp)
target_link_libraries(main SrcLib)
