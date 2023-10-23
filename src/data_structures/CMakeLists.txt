cmake_minimum_required(VERSION 3.22)

file(GLOB SOURCES
    ./coo/*.cpp
    ./csr/*.cpp
    ./matrix/*.cpp
)

add_library(DataStructuresLib STATIC ${SOURCES})
