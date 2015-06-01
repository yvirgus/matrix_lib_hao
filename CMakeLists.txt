cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(matrixhao)

#For find module path
set(CMAKE_MODULE_PATH "${MODULE_EXTRA_PATH}" ${CMAKE_MODULE_PATH})

# Default settings
set(USE_MAGMA off CACHE BOOL "Determines whether we want to use MAGMA library.")
set(DEBUG_MY_CMAKE on CACHE BOOL "Enables verbose debugging of this cmake script.")

#For definitions
add_definitions(${COMPILER_EXTRA_DEF})

#For c++ flag
set(CMAKE_CXX_FLAGS "${COMPILER_EXTRA_FLAG}" ${CMAKE_CXX_FLAGS})

if (DEBUG_MY_CMAKE)
  message("")
  message("DEBUG_MY_CMAKE::")
  message("  MODULE_EXTRA_PATH=${MODULE_EXTRA_PATH}")
  message("  CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
  message("  COMPILER_EXTRA_DEF=${COMPILER_EXTRA_DEF}")
  message("  COMPILER_EXTRA_FLAG=${COMPILER_EXTRA_FLAG}")
  message("  CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
  message("")
endif()
 
#For magma and Cuda
#set(lapackblas_libraries "-L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lm -lgfortran")
#set(cuda_lapackblas_libraries "-L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lm")
#set(cuda_lapackblas_libraries "-L/usr/local/cuda-5.5/lib64 -lcublas -lcudart")

#set(gfortran_library "-L/usr/lib/gcc/x86_64-redhat-linux/4.4.7/ -lgfortran")
#include_directories("/usr/lib/gcc/x86_64-redhat-linux/4.4.7/include")

#include_directories("/sciclone/home00/yvirgus/build/magma-1.6.1/include")
#include_directories("/sciclone/home00/yvirgus/build/magma-1.6.1/lib")
#include_directories("/usr/local/cuda-5.5/include")

#target_link_libraries(matrixhao -L/usr/local/magma-1.6.1/lib -lmagma)
#target_link_libraries(matrixhao -L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lcublas -lcudart -lm -lgfortran)

# For lapackblas library
find_package(mkl)
if(MKL_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64")
#  add_definitions(-DMKL_ILP64 -DUSE_MKL)
  add_definitions( -DUSE_MKL)
  include_directories(${mkl_include_dirs})
  set(lapackblas_libraries "${mkl_libraries}")
endif()

find_package(acml)
if(ACML_FOUND)
  add_definitions(-DUSE_ACML)
  set(lapackblas_libraries "${acml_libraries}")
endif()

if(USE_MAGMA)
  # Use standard FindCUDA
  find_package(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  find_package(MAGMA)
  add_definitions(-DUSE_MAGMA)
  include_directories(${magma_include_dirs})
#  set(magma_libraries "${magma_libraries}" "${CUDA_CUBLAS_LIBRARIES}" "${CUDA_LIBRARIES}" -L/usr/local/intel64/nehalem/gcc/cblas-20100304/lib -lcblas)
  set(magma_libraries "${magma_libraries}" "${CUDA_CUBLAS_LIBRARIES}" "${CUDA_LIBRARIES}")
  message("all libs: CUDA_LIBRARIES = ${CUDA_LIBRARIES} ; CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}")
else()
  MESSAGE("MAGMA support was disabled.")
endif()

include_directories(inc)

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)
