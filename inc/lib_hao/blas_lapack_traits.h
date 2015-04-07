

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
    virtual void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
                      double alpha, const double *A, int_t lda,
                      const double *B, int_t ldb,
                      double beta, double *C, int_t ldc) = 0;
    virtual void getrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
                       int_ptr_t ipiv, int_ptr_t info) = 0;
    virtual void getrf(int_t M, int_t N, double *A, int_t lda,
                       int_ptr_t ipiv, int_ptr_t info) = 0;
    // ... ALL OTHER FUNCTIONS
};

} //end namespace matrix_hao_lib
