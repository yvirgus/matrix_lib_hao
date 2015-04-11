#ifndef MATRIX_HAO_LINALG
#define MATRIX_HAO_LINALG

/*#define "lib_hao/matrix_2d.h"
  #define "lib_hao/blas_lapack_trait.h"*/

#include "matrix_2d.h"
#include "lib_hao/blas_lapack_traits.h"

namespace matrix_hao_lib
{

template <typename _Int_t>
class linalg
{
public:
    // This encapsulates the real linear algebra library implementation
    typedef blas_lapack_traits<_Int_t> linalg_t;
    typedef typename blas_lapack_traits<_Int_t>::int_t int_t;

protected:
    linalg_t *_impl;

public:
    linalg(blas_lapack_traits<_Int_t> *linalg_impl)
        : _impl(linalg_impl)
    {}

    /**************************************/
    /* Matrix Multiply C=alpha*A.B+beta*C */
    /**************************************/

    // _T can be single, double, single complex, or double complex
    template <typename _T>
    void gmm(const Matrix<_T,2>& A, const Matrix<_T,2>& B,
             Matrix<_T,2>& C,
             char TRANSA='N', char TRANSB='N',
             _T alpha=1.0, _T beta=0.0)
    {
        int_t M, N, K, LDA, LDB, LDC;
        M = (TRANSA=='N') ? A.L1 : A.L2;
        K = (TRANSA=='N') ? A.L2 : A.L1;
        N = (TRANSB=='N') ? B.L2 : B.L1;
        LDA = A.L1;
        LDB = B.L1;
        LDC = C.L1;

        _impl->gemm(TRANSA, TRANSB, M, N, K,
                    alpha, A.base_array, LDA,
                    B.base_array, LDB,
                    beta, C.base_array, LDC);
    }

    /**************************************/
    /*****Diagonalize Hermitian Matrix*****/
    /**************************************/
    //template <typename _T>
    void eigen(Matrix<std::complex<double>,2>& A, Matrix<double,1>& W, char JOBZ='V', char UPLO='U')
    {
        
        if(A.L1!=A.L2) throw std::invalid_argument("Input matrix is not a square matrix!");
        int_t N=A.L1, LDA, lwork=-1, lrwork=-1, iwork[1], liwork=-1, info=-1;
        std::complex<double> work[1];
        double rwork[1];
        LDA = N;

        _impl->heevd(JOBZ, UPLO, N, A.base_array, 
                     LDA, W.base_array, work, 
                     lwork, rwork, lrwork, iwork,
                     liwork, info);
    }     

};

}//end namespace matrix_hao_lib

#endif
