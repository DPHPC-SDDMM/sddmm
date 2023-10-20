cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE SOURCES
    ./*.h
    ./*.cpp
)

add_library(SrcLib ${SOURCES})
set_target_properties(SrcLib PROPERTIES LINKER_LANGUAGE CXX)
