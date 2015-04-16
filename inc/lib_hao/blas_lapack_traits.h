#ifndef _LIB_HAO_BLAS_LAPACK_TRAITS
#define _LIB_HAO_BLAS_LAPACK_TRAITS

#include <complex>

namespace matrix_hao_lib
{

template <typename _Int_t=int>
class blas_lapack_traits
{
public:
    typedef _Int_t int_t;  // integer datatype
    typedef int_t *int_ptr_t;
    /* -- put this in detailed implementation below 
    typedef float single_t;
    typedef single_t *single_ptr_t;
    typedef double double_t;
    typedef ...;
    // define ALL numerical datatypes and their pointer counterparts
    */

    // Dispatch functions: all of these are pure virtual and must be defined
    // in the specific implementation

    /* Matrix Multiplication C = alpha*A*B + beta*C */
    virtual void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
                      float alpha, const float *A, int_t lda,
                      const float *B, int_t ldb,
                      float beta, float *C, int_t ldc) = 0;
    virtual void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
                      double alpha, const double *A, int_t lda,
                      const double *B, int_t ldb,
                      double beta, double *C, int_t ldc) = 0;
    virtual void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
                      std::complex<float> alpha, const std::complex<float> *A, int_t lda,
                      const std::complex<float> *B, int_t ldb,
                      std::complex<float> beta, std::complex<float> *C, int_t ldc) = 0;
    virtual void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
                      std::complex<double> alpha, const std::complex<double> *A, int_t lda,
                      const std::complex<double> *B, int_t ldb,
                      std::complex<double> beta, std::complex<double> *C, int_t ldc) = 0;


    /* Diagonalize Hermitian Matrix */
/*    virtual void heevd(char jobz, char uplo, int_t N, std::complex<double> *A, 
                       int_t lda, double *W, std::complex<double> *work, int_t lwork,
                       double *rwork, int_t lrwork, int_t *iwork, int_t liwork, int__ptr_t info) = 0;*/

   virtual void heevd(char jobz, char uplo, int_t N, std::complex<double> *A, 
                       int_t lda, double *W, int_ptr_t info) = 0;

    /*  Diagonalize Hermitian Matrix
       virtual void heev(char jobz, char uplo, int_t N, std::complex<double> *A, 
                       int_t lda, double *W, std::complex<double> *work, int_t lwork,
                       double *rwork, int_ptr_t info) = 0;    
    */

    /* LU decomposition  */

    virtual void getrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
                       int_ptr_t ipiv, int_ptr_t info) = 0;

    /* Inverse Matrix  */

    virtual void getri(int_t N, std::complex<double> *A, int_t lda,
                       int_ptr_t ipiv, int_ptr_t info) = 0;

    /* Solve Linear Equation  */

    virtual void getrs(char trans, int_t N, int_t NRHS, std::complex<double> *A, int_t lda,
                       int_ptr_t ipiv, std::complex<double> *B, int_t ldb, int_ptr_t info) = 0;

    /* QR Decomposition */    
    virtual void geqrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
                       std::complex<double> *tau, int_ptr_t info) = 0;
    
    virtual void ungqr(int_t M, int_t N, int_t K, std::complex<double> *A,
                       int_t lda, std::complex<double> *tau, int_ptr_t info) = 0;

};

} //end namespace matrix_hao_lib
#endif 
