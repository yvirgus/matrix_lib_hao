cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(matrixhao)

if(NOT DEFINED MPI_HAO)
else()
  add_definitions(-DMPI_HAO=${MPI_HAO})
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   set(CMAKE_CXX_FLAGS " ")
   set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++11 ${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
endif()


# For mkl library
add_definitions(-DMKL_ILP64)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64")
set(lapack_blas_dir "/opt/intel/mkl")
set(lapack_blas_include "${lapack_blas_dir}/include")
set(lapack_blas_path    "${lapack_blas_dir}/lib/intel64")
find_library(lp64_lib NAMES mkl_intel_ilp64 PATHS ${lapack_blas_path})
find_library(core_lib NAMES mkl_core PATHS ${lapack_blas_path})
find_library(gnu_thread_lib NAMES mkl_gnu_thread PATHS ${lapack_blas_path})
set(lapack_blas_lib ${lp64_lib} ${core_lib} ${gnu_thread_lib})
include_directories(${lapack_blas_include})



include_directories(inc)

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)

