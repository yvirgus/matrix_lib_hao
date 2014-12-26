set(matrix_test_srcs matrix_2d_blas_lapack_test.cpp matrix_class_test.cpp matrix_mpi_test.cpp test_all.cpp)

add_executable(matrixtest ${matrix_test_srcs})

target_link_libraries(matrixtest matrixhao)

target_link_libraries(matrixtest ${lapack_blas_lib})

install(TARGETS matrixtest DESTINATION test)
