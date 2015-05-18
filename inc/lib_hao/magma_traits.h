#ifndef _LIB_HAO_MAGMA_TRAITS
#define _LIB_HAO_MAGMA_TRAITS

#ifdef USE_MAGMA

#include <complex>
#include <cstdio>

#include "magma.h"

#include "lib_hao/blas_lapack_traits.h"

namespace matrix_hao_lib
{

template <typename _Int_t=int>
class magma_traits: public blas_lapack_traits<_Int_t>
//  This is the wrapper to the REAL implementation of MAGMA
{
public:
    real_Double_t tm_transfer_in, tm_transfer_out, tm_blas;
    real_Double_t tm_query;
    typedef typename blas_lapack_traits<_Int_t>::int_t int_t;
    typedef typename blas_lapack_traits<_Int_t>::int_ptr_t int_ptr_t;

    typedef float single_t;
    typedef double double_t;
    // define ALL numerical datatypes and their pointer counterparts

    typedef magmaFloatComplex ccomplex_t;
    typedef magmaDoubleComplex zcomplex_t;

    typedef single_t *single_ptr_t;
    typedef double_t *double_ptr_t;
    typedef ccomplex_t *ccomplex_ptr_t;
    typedef zcomplex_t *zcomplex_ptr_t;
    typedef const single_t *const_single_ptr_t;
    typedef const double_t *const_double_ptr_t;
    typedef const ccomplex_t *const_ccomplex_ptr_t;
    typedef const zcomplex_t *const_zcomplex_ptr_t;

    static single_ptr_t _cast_Sptr(float *A)
    {
        return A;
    }
    static double_ptr_t _cast_Dptr(double *A)
    {
        return A;
    }
    static ccomplex_ptr_t _cast_Cptr(std::complex<float> *A)
    {
        return reinterpret_cast<ccomplex_ptr_t>(A);
    }
    static zcomplex_ptr_t _cast_Zptr(std::complex<double> *A)
    {
        return reinterpret_cast<zcomplex_ptr_t>(A);
    }
    static const_single_ptr_t _cast_Sptr(const float *A)
    {
        return A;
    }
    static const_double_ptr_t _cast_Dptr(const double *A)
    {
        return A;
    }
    static const_ccomplex_ptr_t _cast_Cptr(const std::complex<float> *A)
    {
        return reinterpret_cast<const_ccomplex_ptr_t>(A);
    }
    static const_zcomplex_ptr_t _cast_Zptr(const std::complex<double> *A)
    {
        return reinterpret_cast<const_zcomplex_ptr_t>(A);
    }
    static ccomplex_t _cast_C(const std::complex<float> &Z)
    {  
        using std::real;
        using std::imag;
        return MAGMA_C_MAKE(real(Z), imag(Z));
    }
    static zcomplex_t _cast_Z(const std::complex<double> &Z)
    {  
        using std::real;
        using std::imag;
        return MAGMA_Z_MAKE(real(Z), imag(Z));
    }

    // Dispatch functions: all of these define the are pure virtual
    // already declared in the base class

    // Matrix multiplication
    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              float alpha, const float *A, int_t lda,
              const float *B, int_t ldb,
              float beta, float *C, int_t ldc) // virtual
    {
        magma_int_t LDDA, LDDB, LDDC;
        magma_int_t Am, An, Bm, Bn, Cm, Cn;
        magma_trans_t transA = magma_trans_const(trans_A), transB = magma_trans_const(trans_B);
        magmaFloat_ptr d_A, d_B, d_C;
        real_Double_t t1, t2, t3;

        // Note: Am = number of physical rows of A matrix
        // Note: An = number of physical columns of A matrix
        Am = (transA == MagmaNoTrans) ? M : K;
        An = (transA == MagmaNoTrans) ? K : M;
        Bm = (transB == MagmaNoTrans) ? K : N;
        Bn = (transB == MagmaNoTrans) ? N : K;
        Cm = M;
        Cn = N;

   
        LDDA = ((Am+31)/32)*32;
        LDDB = ((Bm+31)/32)*32;
        LDDC = ((Cm+31)/32)*32;
        
        // Allocate memory for the matrices on GPU
        t1 = magma_sync_wtime(nullptr);
        magma_smalloc(&d_A, LDDA*An );
        magma_smalloc(&d_B, LDDB*Bn );
        magma_smalloc(&d_C, LDDC*Cn );
   
        // Copy data from host (CPU) to device (GPU)
        magma_ssetmatrix( Am, An, A, lda, d_A, LDDA );
        magma_ssetmatrix( Bm, Bn, B, ldb, d_B, LDDB );
        magma_ssetmatrix( Cm, Cn, C, ldc, d_C, LDDC );
        t2 = magma_sync_wtime(nullptr);
        tm_transfer_in = t2 - t1;
        
        magma_sgemm(transA, transB, M, N, K,
                    alpha, d_A, LDDA,
                           d_B, LDDB,
                    beta,  d_C, LDDC);

        t3 = magma_sync_wtime(nullptr);
        tm_blas = t3 - t2;
   
        // Copy solution from device (GPU) to host (CPU)
        magma_sgetmatrix( Cm, Cn, d_C, LDDC, C, ldc );

        tm_transfer_out = magma_sync_wtime(nullptr) - t3;

        // Free memory on GPU
        magma_free(d_A);
        magma_free(d_B);
        magma_free(d_C);
    }


    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              double alpha, const double *A, int_t lda,
              const double *B, int_t ldb,
              double beta, double *C, int_t ldc) // virtual
    {
        magma_int_t LDDA, LDDB, LDDC;
        magma_int_t Am, An, Bm, Bn, Cm, Cn;
        magma_trans_t transA = magma_trans_const(trans_A), transB = magma_trans_const(trans_B);
        magmaDouble_ptr d_A, d_B, d_C;
        real_Double_t t1, t2, t3;

        // Note: Am = number of physical rows of A matrix
        // Note: An = number of physical columns of A matrix
        Am = (transA == MagmaNoTrans) ? M : K;
        An = (transA == MagmaNoTrans) ? K : M;
        Bm = (transB == MagmaNoTrans) ? K : N;
        Bn = (transB == MagmaNoTrans) ? N : K;
        Cm = M;
        Cn = N;
        /*
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
        */
        LDDA = ((Am+31)/32)*32;
        LDDB = ((Bm+31)/32)*32;
        LDDC = ((Cm+31)/32)*32;

        // Allocate memory for the matrices on GPU
        t1 = magma_sync_wtime(nullptr);
        magma_dmalloc(&d_A, LDDA*An);
        magma_dmalloc(&d_B, LDDB*Bn);
        magma_dmalloc(&d_C, LDDC*Cn);

        // Copy data from host (CPU) to device (GPU)
        magma_dsetmatrix(Am, An, A, lda, d_A, LDDA);
        magma_dsetmatrix(Bm, Bn, B, ldb, d_B, LDDB);
        // FIXME: If beta == 0, no need to do dsetmatrix-->just zero out "*d_C"
        magma_dsetmatrix(Cm, Cn, C, ldc, d_C, LDDC);
        t2 = magma_sync_wtime(nullptr);
        tm_transfer_in = t2 - t1;

        magma_dgemm(transA, transB, M, N, K,
                    alpha, d_A, LDDA,
                    d_B, LDDB,
                    beta,  d_C, LDDC);

        t3 = magma_sync_wtime(nullptr);
        tm_blas = t3 - t2;
        // Copy solution from device (GPU) to host (CPU)
        magma_dgetmatrix(Cm, Cn, d_C, LDDC, C, ldc);

        tm_transfer_out = magma_sync_wtime(nullptr) - t3;

        // Free memory on GPU
        magma_free(d_A);
        magma_free(d_B);
        magma_free(d_C);
    }


    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              std::complex<float> alpha, const std::complex<float> *A, int_t lda,
              const std::complex<float> *B, int_t ldb,
              std::complex<float> beta, std::complex<float> *C, int_t ldc) // virtual
    {
        magma_int_t LDDA, LDDB, LDDC;
        magma_int_t Am, An, Bm, Bn, Cm, Cn;
        magma_trans_t transA = magma_trans_const(trans_A), transB = magma_trans_const(trans_B);
        magmaFloatComplex_ptr d_A, d_B, d_C;
        real_Double_t t1, t2, t3;

        // Note: Am = number of physical rows of A matrix
        // Note: An = number of physical columns of A matrix
        Am = (transA == MagmaNoTrans) ? M : K;
        An = (transA == MagmaNoTrans) ? K : M;
        Bm = (transB == MagmaNoTrans) ? K : N;
        Bn = (transB == MagmaNoTrans) ? N : K;
        Cm = M;
        Cn = N;

        LDDA = ((Am+31)/32)*32;
        LDDB = ((Bm+31)/32)*32;
        LDDC = ((Cm+31)/32)*32;
        
        // Allocate memory for the matrices on GPU     
        t1 = magma_sync_wtime(nullptr);
        magma_cmalloc(&d_A, LDDA*An );
        magma_cmalloc(&d_B, LDDB*Bn );
        magma_cmalloc(&d_C, LDDC*Cn );

        // Copy data from host (CPU) to device (GPU)
        // Casting is required from std:complex<float> to  magmaFloatComplex;
        magma_csetmatrix( Am, An, _cast_Cptr(A), lda, d_A, LDDA );
        magma_csetmatrix( Bm, Bn, _cast_Cptr(B), ldb, d_B, LDDB );
        magma_csetmatrix( Cm, Cn, _cast_Cptr(C), ldc, d_C, LDDC );
        t2 = magma_sync_wtime(nullptr);
        tm_transfer_in = t2 - t1;
      
        magma_cgemm(transA, transB, M, N, K,
                    _cast_C(alpha), d_A, LDDA,
                    d_B, LDDB,
                    _cast_C(beta),  d_C, LDDC);

        t3 = magma_sync_wtime(nullptr);
        tm_blas = t3 - t2;

        // Copy solution from device (GPU) to host (CPU)
        magma_cgetmatrix( Cm, Cn, d_C, LDDC, _cast_Cptr(C), ldc );

        tm_transfer_out = magma_sync_wtime(nullptr) - t3;

        // Free memory on GPU
        magma_free(d_A);
        magma_free(d_B);
        magma_free(d_C);
    }

    void gemm(char trans_A, char trans_B, int_t M, int_t N, int_t K,
              std::complex<double> alpha, const std::complex<double> *A, int_t lda,
              const std::complex<double> *B, int_t ldb,
              std::complex<double> beta, std::complex<double> *C, int_t ldc) // virtual
    {
        magma_int_t LDDA, LDDB, LDDC;
        magma_int_t Am, An, Bm, Bn, Cm, Cn;
        magma_trans_t transA = magma_trans_const(trans_A), transB = magma_trans_const(trans_B);
        magmaDoubleComplex_ptr d_A, d_B, d_C;
        real_Double_t t1, t2, t3;

        // Note: Am = number of physical rows of A matrix
        // Note: An = number of physical columns of A matrix
        Am = (transA == MagmaNoTrans) ? M : K;
        An = (transA == MagmaNoTrans) ? K : M;
        Bm = (transB == MagmaNoTrans) ? K : N;
        Bn = (transB == MagmaNoTrans) ? N : K;
        Cm = M;
        Cn = N;

        LDDA = ((Am+31)/32)*32;
        LDDB = ((Bm+31)/32)*32;
        LDDC = ((Cm+31)/32)*32;

        // Allocate memory for the matrices on GPU     
        t1 = magma_sync_wtime(nullptr);
        magma_zmalloc(&d_A, LDDA*An );
        magma_zmalloc(&d_B, LDDB*Bn );
        magma_zmalloc(&d_C, LDDC*Cn );

        // Copy data from host (CPU) to device (GPU)
        // Casting is required from std:complex<double> to  magmaDoubleComplex;
        magma_zsetmatrix( Am, An, _cast_Zptr(A), lda, d_A, LDDA );
        magma_zsetmatrix( Bm, Bn, _cast_Zptr(B), ldb, d_B, LDDB );
        magma_zsetmatrix( Cm, Cn, _cast_Zptr(C), ldc, d_C, LDDC );
        t2 = magma_sync_wtime(nullptr);
        tm_transfer_in = t2 - t1;
      
        magma_zgemm(transA, transB, M, N, K,
                    _cast_Z(alpha), d_A, LDDA,
                    d_B, LDDB,
                    _cast_Z(beta),  d_C, LDDC);

        t3 = magma_sync_wtime(nullptr);
        tm_blas = t3 - t2;
      
        // Copy solution from device (GPU) to host (CPU)
        magma_zgetmatrix( Cm, Cn, d_C, LDDC, _cast_Zptr(C), ldc );
	tm_transfer_out = magma_sync_wtime(nullptr) - t3; 
        
	// Free memory on GPU
        magma_free(d_A);
        magma_free(d_B);
        magma_free(d_C);
    }

    // Eigen values and vectors
    void heevd(char jobz, char uplo, int_t N, std::complex<double> *A,       
               int_t lda, double *W, int_ptr_t info) //virtual
    {
        magma_vec_t JOBZ = magma_vec_const(jobz);
        magma_uplo_t UPLO = magma_uplo_const(uplo);
        magmaDoubleComplex *h_work, aux_work[1];
        int_t lwork=-1, lrwork=-1, *iwork, aux_iwork[1], liwork=-1;
        double *rwork, aux_rwork[1];
	real_Double_t t1, t2;

        //magma_zheevd( JOBZ, UPLO, N, nullptr, lda, nullptr,
        //              _cast_Zptr(work),  lwork, rwork, lrwork, iwork, liwork, info );

	t1 = magma_sync_wtime(nullptr);
        magma_zheevd( JOBZ, UPLO, N, nullptr, lda, nullptr,
                      aux_work,  lwork, aux_rwork, lrwork, aux_iwork, liwork, info );
	
	tm_query = magma_sync_wtime(nullptr) - t1;

        lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
        lrwork = (magma_int_t) aux_rwork[0];
        liwork = aux_iwork[0];

        // allocate memory on CPU
        magma_dmalloc_cpu(&rwork, lrwork);
        magma_imalloc_cpu(&iwork, liwork);
        magma_zmalloc_pinned(&h_work, lwork);

#if 0
	using std::printf;
	using std::fflush;
	printf("magma_zheevd: N=%d, lda=%d, lwork=%d, lrwork=%d, liwork=%d\n",
	       N, lda, lwork, lrwork, liwork);
	fflush(stdout);
#endif
	t2 = magma_sync_wtime(nullptr);
        magma_zheevd( JOBZ, UPLO, N, _cast_Zptr(A), lda, W,
                      h_work, lwork, rwork, lrwork, iwork, liwork, info );
	tm_blas = magma_sync_wtime(nullptr) - t2;
#if 0
	printf("magma_zheevd: info=%d\n",
	       *info);
	fflush(stdout);
#endif	

        // free allocated memory
        magma_free_cpu(rwork);
        magma_free_cpu(iwork);
        magma_free_pinned(h_work);
    }

    // LU decomposition
    void getrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, int_ptr_t info) // virtual
    {
        magma_zgetrf(M, N, _cast_Zptr(A), lda, ipiv, info);
    }

    // Inverse matrix 
    void getri(int_t N, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, int_ptr_t info)  //virtual 
    {
        magmaDoubleComplex_ptr d_A , dwork;
        magma_int_t ldda, ldwork;
	real_Double_t t1, t2, t3;
        ldda = ((lda+31)/32)*32;
        ldwork = N * magma_get_zgetri_nb(N); // magma_get_zgetri_nb optimizes the blocksize

	t1 = magma_sync_wtime(nullptr);
        magma_zmalloc( &d_A, ldda*N );
        magma_zmalloc( &dwork, ldwork );

        // copy matrix from CPU to GPU
        magma_zsetmatrix( N, N, _cast_Zptr(A), lda, d_A, ldda );
	tm_transfer_in = magma_sync_wtime(nullptr) - t1;

        // calculate the inverse matrix with zgetri
	t2 = magma_sync_wtime(nullptr);
        magma_zgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, info );
	tm_blas = magma_sync_wtime(nullptr) - t2;

        // copy matrix from GPU to CPU
	t3 = magma_sync_wtime(nullptr);
        magma_zgetmatrix( N, N, d_A, ldda, _cast_Zptr(A), lda );
	tm_transfer_out = magma_sync_wtime(nullptr) - t3;

        magma_free( d_A );
        magma_free( dwork );
    }

    // Solve Linear Equation 
    void getrs(char trans, int_t N, int_t NRHS, std::complex<double> *A, int_t lda,
               int_ptr_t ipiv, std::complex<double> *B, int_t ldb, int_ptr_t info) //virtual
    {
        magma_trans_t Trans = magma_trans_const(trans);
        magmaDoubleComplex_ptr d_A, d_B;
        magma_int_t ldda, lddb;
	real_Double_t t1, t2, t3;
        ldda = ((lda+31)/32)*32;
        lddb = ((ldb+31)/32)*32;

        //allocate memory on GPU
	t1 = magma_sync_wtime(nullptr);
        magma_zmalloc( &d_A, ldda*N );
        magma_zmalloc( &d_B, lddb*NRHS );

        // copy matrix from CPU to GPU
        magma_zsetmatrix( N, N, _cast_Zptr(A), lda, d_A, ldda );
        magma_zsetmatrix( N, NRHS, _cast_Zptr(B), ldb, d_B, lddb );
	tm_transfer_in = magma_sync_wtime(nullptr) - t1;

	t2 = magma_sync_wtime(nullptr);
        magma_zgetrs_gpu( Trans, N, NRHS, d_A, ldda, ipiv, d_B, lddb, info );
	tm_blas = magma_sync_wtime(nullptr) - t2;

        // copy matrix from GPU to CPU
	t3 = magma_sync_wtime(nullptr);
        magma_zgetmatrix( N, NRHS, d_B, lddb, _cast_Zptr(B), ldb );
	tm_transfer_out = magma_sync_wtime(nullptr) - t3;

        // free memory
        magma_free( d_A );
        magma_free( d_B );
    }

    // QR decomposition
    void geqrf(int_t M, int_t N, std::complex<double> *A, int_t lda,
               std::complex<double> *tau, int_ptr_t info) // virtual
    {
        magmaDoubleComplex *h_work;
        magma_int_t lwork=-1, nb;

        nb = magma_get_zgeqrf_nb(M);
        lwork = std::max( N*nb, 2*nb*nb );

        magma_zmalloc_cpu(&h_work, lwork);

        // perform zgeqrf
        magma_zgeqrf(M, N, _cast_Zptr(A), lda, _cast_Zptr(tau), h_work, lwork, info);

        magma_free_cpu( h_work );
    }

    void ungqr(int_t M, int_t N, int_t K, std::complex<double> *A,
               int_t lda, std::complex<double> *tau, int_ptr_t info) //virtual
    {
        magma_zungqr2( M, N, K, _cast_Zptr(A), lda, _cast_Zptr(tau), info );
    }

};

} //end namespace matrix_hao_lib

#endif 

#endif
