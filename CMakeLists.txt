cmake_minimum_required(VERSION 4.0.0)

project(
    yadl
    DESCRIPTION "Yet Another Distribution Library"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)

add_executable(
    main
    example/main.cpp
)

target_include_directories(
    main
    PRIVATE
    src/
)

target_compile_options(
    main
    PRIVATE
    -fdiagnostics-color=always
    -Wall
    -Wextra
    -O3
    -march=native
)

target_link_libraries(
    main
    PRIVATE
    -lgsl
    -lgslcblas
)
