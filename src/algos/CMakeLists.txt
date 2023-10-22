cmake_minimum_required(VERSION 3.27)

file(GLOB SOURCES
    ./*.cpp
)

add_library(AlgoLib STATIC ${SOURCES})
