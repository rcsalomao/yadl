# Works with 3.11 and tested through 3.23
cmake_minimum_required(VERSION 3.11...3.23)

project(
    yadl
    DESCRIPTION "Yet Another Distribution Library"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(
    main
    ./src/main.cpp
)

target_include_directories(
    main
    PRIVATE
    ./src/yadl
)

target_compile_options(
    main
    PRIVATE
    -fdiagnostics-color=always
    -Wall
    -Wextra
    -std=c++2a
    -fconcepts
    -O3
    -march=native
    -pipe
)

target_link_libraries(
    main
    PRIVATE
    -lgslcblas
    -lgsl
)
