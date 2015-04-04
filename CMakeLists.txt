cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(matrixhao)

#For find module path
set(CMAKE_MODULE_PATH "${MODULE_EXTRA_PATH}" ${CMAKE_MODULE_PATH})

#For definitions
add_definitions(${COMPILER_EXTRA_DEF})

#For c++ flag
set(CMAKE_CXX_FLAGS "${COMPILER_EXTRA_FLAG}" ${CMAKE_CXX_FLAGS})

# For lapackblas library
find_package(mkl)
if(MKL_FOUND)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64")
#  add_definitions(-DMKL_ILP64 -DUSE_MKL)
#  add_definitions( -DUSE_MKL)
  include_directories(${mkl_include_dirs})
#  set(lapackblas_libraries "${mkl_libraries}")
endif()


find_package(acml)
if(ACML_FOUND)
  add_definitions(-DUSE_ACML)
  set(lapackblas_libraries "${acml_libraries}")
endif()


#set(lapackblas_libraries "-L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lm -lgfortran")
set(cuda_lapackblas_libraries "-L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lm")

#set(gfortran_library "-L/usr/lib/gcc/x86_64-redhat-linux/4.4.7/ -lgfortran")
#include_directories("/usr/lib/gcc/x86_64-redhat-linux/4.4.7/include")

include_directories("/usr/local/magma-1.6.1/include")
include_directories("/usr/local/magma-1.6.1/lib")
include_directories("/usr/local/cuda-7.0/include")

#target_link_libraries(matrixhao -L/usr/local/magma-1.6.1/lib -lmagma)
#target_link_libraries(matrixhao -L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lcublas -lcudart -lm -lgfortran)

include_directories(inc)

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)
