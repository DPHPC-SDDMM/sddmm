cmake_minimum_required(VERSION 3.22)

file(GLOB SOURCES
    *.h
    *.cpp
)

add_library(MatrixFileReadingLib STATIC ${SOURCES})