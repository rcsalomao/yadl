project('test', 'cpp')

sourceList = [
    './src/main.cpp',
]

includeDir = include_directories([
    './src/yadl',
])

compilerFlags = [
    '-Wall',
    '-Wextra',
    '-std=c++2a',
    '-O3',
    # '-ftime-report',
    '-march=native',
    '-pipe',
    '-fopenmp',
]

linkerArgs = [
    # '-L/opt/intel/mkl/lib/intel64',
    # '-lmkl_rt',
    '-lgslcblas',
    '-lgomp',
    '-lgsl'
]

executable(
    'main',
    sources : sourceList,
    cpp_args : compilerFlags,
    # cpp_pch : 'pch/pch.hpp',
    link_args : linkerArgs,
    include_directories : includeDir
)
