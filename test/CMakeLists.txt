set(matrix_test_srcs matrix_2d_blas_lapack_test.cpp matrix_class_test.cpp matrix_mpi_test.cpp test_all.cpp matrix_linalg_test.cpp matrix_size_linalg_test.cpp zheevd_test.cpp)

add_executable(matrixtest ${matrix_test_srcs})

target_link_libraries(matrixtest matrixhao)

#target_link_libraries(matrixtest -L/usr/local/magma-1.6.1/lib -lmagma)
#target_link_libraries(matrixtest -L/sciclone/home00/yvirgus/build/magma-1.6.1/lib -lmagma)
target_link_libraries(matrixtest ${magma_libraries})

#target_link_libraries(matrixtest -L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lcublas -lcudart -lm -lgfortran)
#target_link_libraries(matrixtest -L/usr/local/cuda-7.0/lib64 -L/usr/lib64/atlas-sse3 -L/usr/lib64/atlas-sse3/lib -llapack -lf77blas -lcblas -latlas -lcublas -lcudart -lm)
target_link_libraries(matrixtest ${cuda_lapackblas_libraries})

#target_link_libraries(matrixtest ${gfortran_library})

target_link_libraries(matrixtest ${lapackblas_libraries})


install(TARGETS matrixtest DESTINATION test)
