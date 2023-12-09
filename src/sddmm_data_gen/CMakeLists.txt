cmake_minimum_required(VERSION 3.22)

file(GLOB SOURCES
    ./*.cpp
)

add_library(GenLib STATIC ${SOURCES})
