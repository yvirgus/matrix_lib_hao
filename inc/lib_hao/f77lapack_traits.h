#ifndef _LIB_HAO_F77LAPACK_TRAITS
#define _LIB_HAO_F77LAPACK_TRAITS

#include <complex>

#ifdef USE_MKL
#include "mkl.h"
#endif

#ifdef USE_ACML
#include "acml.h"
#endif

#include "lib_hao/blas_lapack_traits.h"
#include "matrix_define.h"


namespace matrix_hao_lib
{

template <typename _Int_t=int>
class f77lapack_traits: public blas_lapack_traits<_Int_t>
//  This is the wrapper to the REAL implementation BLAS/LAPACK implementation
//  as defined in Fortran 77 style.
{
public:
    typedef typename blas_lapack_traits<_Int_t>::int_t int_t;
    typedef typename blas_lapack_traits<_Int_t>::int_ptr_t int_ptr_t;

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
        return reinterpret_cast<ccomplex_ptr_t>(const_cast<std::complex<float> *>(A));
    }
    static zcomplex_ptr_t _cast_Zptr(const std::complex<double> *A)
    {
        return reinterpret_cast<zcomplex_ptr_t>(const_cast<std::complex<double> *>(A));
    }

    // Dispatch functions: all of these define the are pure virtual
    // already declared in the base class

    // Matrix Multiplication
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

    // Eigen values and eigen vectors 
    void heevd(char jobz, char uplo, int_t N, std::complex<double> *A,       
               int_t lda, double *W, int_ptr_t info) //virtual
    {
        int_t lwork=-1, lrwork=-1, aux_iwork[1], liwork=-1;
        std::complex<double> aux_work[1];
        double aux_rwork[1];

        FORTRAN_NAME(zheevd)(&jobz, &uplo, &N, _cast_Zptr(A), &lda, _cast_Dptr(W),
                             _cast_Zptr(aux_work), &lwork, aux_rwork, &lrwork,
                             aux_iwork, &liwork, info);

        lwork = lround(aux_work[0].real());
        std::complex<double> *work = new std::complex<double>[lwork];

        lrwork = lround(aux_rwork[0]);
        double *rwork = new double[lrwork];

        liwork = *aux_iwork;
        int_t *iwork = new int_t[liwork];

        FORTRAN_NAME(zheevd)(&jobz, &uplo, &N, _cast_Zptr(A), &lda, _cast_Dptr(W),
                             _cast_Zptr(work), &lwork, rwork, &lrwork,
                             iwork, &liwork, info);

        delete[] work;
        delete[] rwork;
        delete[] iwork;
    }

    // LU decomposition
    void getrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, int_ptr_t info) // virtual
    {
        FORTRAN_NAME(zgetrf)(&M, &N, _cast_Zptr(A), &lda, ipiv, info);
    }

    // Inverse matrix 
    void getri(int_t N, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, int_ptr_t info)  //virtual 
    {
        int_t lwork = -1;
        std::complex<double> aux_work[1];

        FORTRAN_NAME(zgetri)(&N, _cast_Zptr(A), &lda, ipiv, 
                             _cast_Zptr(aux_work), &lwork, info);

        lwork = lround(aux_work[0].real());
        std::complex<double> *work = new std::complex<double>[lwork];

        FORTRAN_NAME(zgetri)(&N, _cast_Zptr(A), &lda, ipiv, 
                             _cast_Zptr(work), &lwork, info);
        
        delete[] work;
    }


    // Solve Linear Equation 
    void getrs(char trans, int_t N, int_t NRHS, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, std::complex<double> *B, int_t ldb, int_ptr_t info) //virtual
    {
        FORTRAN_NAME(zgetrs)(&trans, &N, &NRHS, _cast_Zptr(A), &lda, 
                             ipiv, _cast_Zptr(B), &ldb, info);
    }

};

} //end namespace matrix_hao_lib

#endif 
