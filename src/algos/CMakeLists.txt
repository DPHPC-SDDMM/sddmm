cmake_minimum_required(VERSION 3.9.1)

file(GLOB SOURCES
    ./*.cpp
)

add_library(AlgoLib STATIC ${SOURCES})
