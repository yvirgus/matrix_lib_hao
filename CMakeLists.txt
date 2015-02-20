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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -m64")
  add_definitions(-DMKL_ILP64)
  include_directories(${mkl_include_dirs})
  set(lapackblas_libraries "${mkl_libraries}")
endif()


find_package(acml)
if(ACML_FOUND)
  set(lapackblas_libraries "${acml_libraries}")
endif()


include_directories(inc)

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)
