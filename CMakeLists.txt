cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(matrixhao)

#For find module path
set(CMAKE_MODULE_PATH "${MODULE_EXTRA_PATH}" ${CMAKE_MODULE_PATH})

#For definitions
add_definitions(${COMPILER_EXTRA_DEF})

#For c++ flag
set(CMAKE_CXX_FLAGS "${COMPILER_EXTRA_FLAG}" ${CMAKE_CXX_FLAGS})

# For lapackblas library
#-DMKL_ILP64 -fopenmp -m64 are needed for mkl
find_package(lapackblas REQUIRED)
include_directories(${lapackblas_include_dirs})

include_directories(inc)

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)
