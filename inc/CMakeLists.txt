set(header matrix_define.h matrix_base.h matrix_1d.h matrix_2d.h matrix_3d.h matrix_element_wise.h
    matrix_mpi.h matrix_2d_blas_lapack.h matrix_all.h)

install(FILES ${header}  DESTINATION include)
