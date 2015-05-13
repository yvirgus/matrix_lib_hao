set(header matrix_all.h acml.h)

#set(header2 lib_hao/matrix_define.h lib_hao/matrix_base.h lib_hao/matrix_2d.h lib_hao/matrix_2d_blas_lapack.h 
#            lib_hao/blas_lapack_traits.h lib_hao/f77lapack_traits.h lib_hao/magma_traits.h lib_hao/matrix_linalg.h lib_hao/flops.h)

set(header2 lib_hao/matrix_1d.h lib_hao/matrix_3d.h lib_hao/matrix_element_wise.h lib_hao/matrix_mpi.h 
            lib_hao/matrix_define.h lib_hao/matrix_base.h lib_hao/matrix_2d.h lib_hao/blas_lapack_traits.h 
            lib_hao/f77lapack_traits.h lib_hao/magma_traits.h lib_hao/matrix_linalg.h lib_hao/flops.h)

install(FILES ${header}  DESTINATION include)

install(FILES ${header2}  DESTINATION include/lib_hao)
