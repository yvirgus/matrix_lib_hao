set(header matrix_define.h matrix_base.h matrix_1d.h matrix_2d.h matrix_3d.h matrix_element_wise.h
    matrix_mpi.h matrix_2d_blas_lapack.h matrix_all.h acml.h lib_hao/blas_lapack_traits.h lib_hao/f77lapack_traits.h lib_hao/magma_traits.h lib_hao/matrix_linalg.h lib_hao/flops.h)

install(FILES ${header}  DESTINATION include)
