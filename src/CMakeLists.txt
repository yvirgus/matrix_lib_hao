#set(matrix_srcs matrix_2d_blas_lapack.cpp matrix_mpi.cpp)
set(matrix_srcs matrix_mpi.cpp)

add_library(matrixhao STATIC ${matrix_srcs})

install(TARGETS matrixhao  DESTINATION lib)
