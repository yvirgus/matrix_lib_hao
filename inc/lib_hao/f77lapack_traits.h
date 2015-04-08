#ifndef _LIB_HAO_F77LAPACK_TRAITS
#define _LIB_HAO_F77LAPACK_TRAITS

#include <complex>

#ifdef USE_MKL
#include "mkl.h"
#endif

#ifdef USE_ACML
#include "acml.h"
#endif

//#include "lib_hao/blas_lapack_traits.h"
#include "blas_lapack_traits.h"
#include "matrix_define.h"


namespace matrix_hao_lib
{

template <typename _Int_t=int>
class f77lapack_traits: public blas_lapack_traits<_Int_t>
//  This is the wrapper to the REAL implementation BLAS/LAPACK implementation
//  as defined in Fortran 77 style.
{
public:
    typedef float single_t;
    typedef double double_t;
    // define ALL numerical datatypes and their pointer counterparts
#ifdef USE_MKL
    typedef MKL_Complex8 ccomplex_t;
    typedef MKL_Complex16 zcomplex_t;
#elif USE_ACML
    typedef ccomplex       ccomplex_t;
    typedef doubleccomplex zcomplex_t;
#else
    typedef std::complex<single_t> ccomplex_t;
    typedef std::complex<double_t> zcomplex_t;
#endif

    typedef single_t *single_ptr_t;
    typedef double_t *double_ptr_t;
    typedef ccomplex_t *ccomplex_ptr_t;
    typedef zcomplex_t *zcomplex_ptr_t;

    static single_ptr_t _cast_Sptr(const float *A)
    {
        return const_cast<single_ptr_t>(A);
    }
    static double_ptr_t _cast_Dptr(const double *A)
    {
        return const_cast<double_ptr_t>(A);
    }
    static ccomplex_ptr_t _cast_Cptr(const std::complex<float> *A)
    {
        return reinterpret_cast<ccomplex_ptr_t>(A);
    }
    static zcomplex_ptr_t _cast_Zptr(const std::complex<double> *A)
    {
        return reinterpret_cast<zcomplex_ptr_t>(A);
    }

    // Dispatch functions: all of these define the are pure virtual
    // already declared in the base class
    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              float alpha, const float *A, int_t lda,
              const float *B, int_t ldb,
              float beta, float *C, int_t ldc) // virtual
    {
        FORTRAN_NAME(sgemm)(&trans_A, &trans_B,
                            &M, &N, &K,
                            _cast_Sptr(&alpha), _cast_Sptr(A), &lda,
                            _cast_Sptr(B), &ldb,
                            _cast_Sptr(&beta), _cast_Sptr(C), &ldc);
    }

    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              double alpha, const double *A, int_t lda,
              const double *B, int_t ldb,
              double beta, double *C, int_t ldc) // virtual
    {
        FORTRAN_NAME(dgemm)(&trans_A, &trans_B,
                            &M, &N, &K,
                            _cast_Dptr(&alpha), _cast_Dptr(A), &lda,
                            _cast_Dptr(B), &ldb,
                            _cast_Dptr(&beta), _cast_Dptr(C), &ldc);
    }

    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              std::complex<float> alpha, const std::complex<float> *A, int_t lda,
              const std::complex<float> *B, int_t ldb,
              std::complex<float> beta, std::complex<float> *C, int_t ldc) // virtual
    {
        FORTRAN_NAME(cgemm)(&trans_A, &trans_B,
                            &M, &N, &K,
                            _cast_Cptr(&alpha), _cast_Cptr(A), &lda,
                            _cast_Cptr(B), &ldb,
                            _cast_Cptr(&beta), _cast_Cptr(C), &ldc);
    }

    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              std::complex<double> alpha, const std::complex<double> *A, int_t lda,
              const std::complex<double> *B, int_t ldb,
              std::complex<double> beta, std::complex<double> *C, int_t ldc) // virtual
    {
        FORTRAN_NAME(zgemm)(&trans_A, &trans_B,
                            &M, &N, &K,
                            _cast_Zptr(&alpha), _cast_Zptr(A), &lda,
                            _cast_Zptr(B), &ldb,
                            _cast_Zptr(&beta), _cast_Zptr(C), &ldc);
    }

    // Fill IN FOR S, C, Z types
    void getrf(int_t M, int_t N, double *A, int_t lda,
               int_ptr_t ipiv, int_ptr_t info) // virtual
    {
        FORTRAN_NAME(dgetrf)(&M, &N,
                             _cast_Dptr(A), &lda,
                             ipiv, info);
    }
    void getrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, int_ptr_t info) // virtual
    {
        // FILL THIS IN
    }
    // ... ALL OTHER FUNCTIONS
};

} //end namespace matrix_hao_lib

#endif 
