#include <complex>

#include "magma.h"

#include "lib_hao/blas_lapack_traits.h"
#include "matrix_define.h"


namespace matrix_hao_lib
{

template <typename _Int_t=int>
class magma_traits: public blas_lapack_traits<_Int_t>
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
    static ccomplex_t _cast_Cptr(const std::complex<float> *A)
    {
        return reinterpret_cast<ccomplex_t>(A);
    }
    static zcomplex_t _cast_Zptr(const std::complex<double> *A)
    {
        return reinterpret_cast<zcomplex_t>(A);
    }

    // Dispatch functions: all of these define the are pure virtual
    // already declared in the base class
    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              double alpha, const double *A, int_t lda,
              const double *B, int_t ldb,
              double beta, double *C, int_t ldc) // virtual
    {
        magma_int_t M, N, K;
        magma_int_t LDA, LDB, LDC, LDDA, LDDB, LDDC;
        magma_int_t Am, An, Bm, Bn;
        magma_trans_t transA = magma_trans_const(trans_A),
                      transB = magma_trans_const(trans_B);
        magmaFloat_ptr d_A, d_B, d_C;
   
        M=(transA==MagmaNoTrans) ? A.L1:A.L2;
        K=(transA==MagmaNoTrans) ? A.L2:A.L1;
        N=(transB==MagmaNoTrans) ? B.L2:B.L1;
        //LDA=A.L1;
        //LDB=B.L1;
        //LDC=C.L1;
        if ( transA == MagmaNoTrans ) {
          LDA = Am = M;
          An = K;
        } else {
          LDA = Am = K;
          An = M;
        }
   
        if ( transB == MagmaNoTrans ) {
          LDB = Bm = K;
          Bn = N;
        } else {
          LDB = Bm = N;
          Bn = K;
        }
        LDC = M;
   
        LDDA = ((LDA+31)/32)*32;
        LDDB = ((LDB+31)/32)*32;
        LDDC = ((LDC+31)/32)*32;
        
        // Allocate memory for the matrices on GPU
        magma_smalloc(&d_A, LDDA*An );
        magma_smalloc(&d_B, LDDB*Bn );
        magma_smalloc(&d_C, LDDC*N );
   
        // Copy data from host (CPU) to device (GPU)
        magma_ssetmatrix( Am, An, A.base_array, LDA, d_A, LDDA );
        magma_ssetmatrix( Bm, Bn, B.base_array, LDB, d_B, LDDB );
        magma_ssetmatrix( M, N, C.base_array, LDC, d_C, LDDC );
        
        magma_sgemm(transA, transB, M, N, K,
                    alpha, d_A, LDDA,
                           d_B, LDDB,
                    beta,  d_C, LDDC);
   
        // Copy solution from device (GPU) to host (CPU)
        magma_sgetmatrix( M, N, d_C, LDDC, C.base_array, LDC );
   
        // Free memory on GPU
        magma_free(d_A);
        magma_free(d_B);
        magma_free(d_C);
    }
    // FILL IN FOR S, C, Z types
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
