cmake_minimum_required(VERSION 3.22)

file(GLOB Cu_Exp_SOURCES_cpp
    ./*.cpp
)

add_library(CudaExp STATIC ${Cu_Exp_SOURCES_cpp})

file(GLOB Cu_Exp_SOURCES
    ./*.cuh
    ./*.cu
)

add_library(CudaExpCu STATIC ${Cu_Exp_SOURCES})