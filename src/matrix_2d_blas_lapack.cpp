#include <cmath>
#include "matrix_all.h"
//#include "magma.h"
//#include "magma_lapack.h"

using std::complex;
using std::conj;
using std::cout;

namespace matrix_hao_lib
{


 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */
 /*************************************/

  void gmm(const Matrix<float,2>& A, const Matrix<float,2>& B, Matrix<float,2>& C, 
          char TRANSA, char TRANSB, float alpha, float beta)
 {
     BL_INT  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N') ? A.L1:A.L2;
     K=(TRANSA=='N') ? A.L2:A.L1;
     N=(TRANSB=='N') ? B.L2:B.L1;
     LDA=A.L1;
     LDB=B.L1;
     LDC=C.L1;
     FORTRAN_NAME(sgemm)(&TRANSA, &TRANSB, &M, &N, &K, 
                         (BL_FLOAT* )&alpha, (BL_FLOAT* )A.base_array, &LDA, 
                         (BL_FLOAT* )B.base_array, &LDB, 
                         (BL_FLOAT* )&beta,  (BL_FLOAT* )C.base_array, &LDC);
 }



 void gmm(const Matrix<double,2>& A, const Matrix<double,2>& B, Matrix<double,2>& C,
          char TRANSA, char TRANSB, double alpha, double beta)
 {
     BL_INT  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N') ? A.L1:A.L2;
     K=(TRANSA=='N') ? A.L2:A.L1;
     N=(TRANSB=='N') ? B.L2:B.L1;
     LDA=A.L1;
     LDB=B.L1;
     LDC=C.L1;
     FORTRAN_NAME(dgemm)(&TRANSA, &TRANSB, &M, &N, &K,
                         (BL_DOUBLE* )&alpha, (BL_DOUBLE* )A.base_array, &LDA,
                         (BL_DOUBLE* )B.base_array, &LDB,
                         (BL_DOUBLE* )&beta,  (BL_DOUBLE* )C.base_array, &LDC);
 }


 void gmm(const Matrix<complex<float>,2>& A, const Matrix<complex<float>,2>& B, Matrix<complex<float>,2>& C,
          char TRANSA, char TRANSB, complex<float> alpha,complex<float> beta)
 {
     BL_INT  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N') ? A.L1:A.L2;
     K=(TRANSA=='N') ? A.L2:A.L1;
     N=(TRANSB=='N') ? B.L2:B.L1;
     LDA=A.L1;
     LDB=B.L1;
     LDC=C.L1;
     FORTRAN_NAME(cgemm)(&TRANSA, &TRANSB, &M, &N, &K,
                         (BL_COMPLEX8* )&alpha, (BL_COMPLEX8* )A.base_array, &LDA,
                         (BL_COMPLEX8* )B.base_array, &LDB,
                         (BL_COMPLEX8* )&beta,  (BL_COMPLEX8* )C.base_array, &LDC);
 }

 void gmm(const Matrix<complex<double>,2>& A, const Matrix<complex<double>,2>& B, Matrix<complex<double>,2>& C,
          char TRANSA, char TRANSB, complex<double> alpha, complex<double> beta)
 {
     BL_INT  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N') ? A.L1:A.L2;
     K=(TRANSA=='N') ? A.L2:A.L1;
     N=(TRANSB=='N') ? B.L2:B.L1;
     LDA=A.L1;
     LDB=B.L1;
     LDC=C.L1;
     FORTRAN_NAME(zgemm)(&TRANSA, &TRANSB, &M, &N, &K,
                         (BL_COMPLEX16* )&alpha, (BL_COMPLEX16* )A.base_array, &LDA,
                         (BL_COMPLEX16* )B.base_array, &LDB,
                         (BL_COMPLEX16* )&beta,  (BL_COMPLEX16* )C.base_array, &LDC);
 }  


 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */   /* Using MAGMA library */
 /*************************************/


 void gmm_magma(const Matrix<float,2>& A, const Matrix<float,2>& B, Matrix<float,2>& C,
                char TRANSA, char TRANSB, float alpha, float beta)
 {
     magma_int_t M, N, K; 
     magma_int_t LDA, LDB, LDC, LDDA, LDDB, LDDC;
     magma_int_t Am, An, Bm, Bn;
     magma_trans_t transA = magma_trans_const(TRANSA), transB = magma_trans_const(TRANSB);
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


 void gmm_magma(const Matrix<double,2>& A, const Matrix<double,2>& B, Matrix<double,2>& C,
                 char TRANSA, char TRANSB, double alpha, double beta)
 {
     magma_int_t M, N, K; 
     magma_int_t LDA, LDB, LDC, LDDA, LDDB, LDDC;
     magma_int_t Am, An, Bm, Bn;
     magma_trans_t transA = magma_trans_const(TRANSA), transB = magma_trans_const(TRANSB);
     magmaDouble_ptr d_A, d_B, d_C;

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
     magma_dmalloc(&d_A, LDDA*An );
     magma_dmalloc(&d_B, LDDB*Bn );
     magma_dmalloc(&d_C, LDDC*N );

     // Copy data from host (CPU) to device (GPU)
     magma_dsetmatrix( Am, An, A.base_array, LDA, d_A, LDDA );
     magma_dsetmatrix( Bm, Bn, B.base_array, LDB, d_B, LDDB );
     magma_dsetmatrix( M, N, C.base_array, LDC, d_C, LDDC );
     
     magma_dgemm(transA, transB, M, N, K,
		 alpha, d_A, LDDA,
                        d_B, LDDB,
    		 beta,  d_C, LDDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_dgetmatrix( M, N, d_C, LDDC, C.base_array, LDC );

     // Free memory on GPU
     magma_free(d_A);
     magma_free(d_B);
     magma_free(d_C);
 }

 // Internal functions to aid safe typecasting to MAGMA-specific datatypes
 // Note: Since this can be a risky cast, I isolate these casts as private functions.

 static inline magmaFloatComplex _cast_C_magma(const std::complex<float> &Z)
 {
    using std::real;
    using std::imag;
    return MAGMA_C_MAKE(real(Z), imag(Z));
 }

 static inline magmaFloatComplex_ptr _cast_Cptr_magma(std::complex<float> *Z)
 {
    return reinterpret_cast<magmaFloatComplex_ptr>(Z);
 }

 static inline magmaDoubleComplex _cast_Z_magma(const std::complex<double> &Z)
 {
    using std::real;
    using std::imag;
    return MAGMA_Z_MAKE(real(Z), imag(Z));
 }

 static inline magmaDoubleComplex_ptr _cast_Zptr_magma(std::complex<double> *Z)
 {
    return reinterpret_cast<magmaDoubleComplex_ptr>(Z);
 }


void gmm_magma(const Matrix<complex<float>,2>& A, const Matrix<complex<float>,2>& B, Matrix<complex<float>,2>& C,
               char TRANSA, char TRANSB, complex<float> alpha, complex<float> beta)
{
     magma_int_t M, N, K; 
     magma_int_t LDA, LDB, LDC, LDDA, LDDB, LDDC;
     magma_int_t Am, An, Bm, Bn;
     magma_trans_t transA = magma_trans_const(TRANSA), transB = magma_trans_const(TRANSB);
     magmaFloatComplex_ptr d_A, d_B, d_C;

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
     magma_cmalloc(&d_A, LDDA*An );
     magma_cmalloc(&d_B, LDDB*Bn );
     magma_cmalloc(&d_C, LDDC*N );

     // Copy data from host (CPU) to device (GPU)
     // Casting is required from std:complex<float> to  magmaFloatComplex;
     magma_csetmatrix( Am, An, _cast_Cptr_magma(A.base_array), LDA, d_A, LDDA );
     magma_csetmatrix( Bm, Bn, _cast_Cptr_magma(B.base_array), LDB, d_B, LDDB );
     magma_csetmatrix( M, N, _cast_Cptr_magma(C.base_array), LDC, d_C, LDDC );
     
     // This casting gives an warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
     magma_cgemm(transA, transB, M, N, K,
		 _cast_C_magma(alpha), d_A, LDDA,
                                       d_B, LDDB,
		 _cast_C_magma(beta),  d_C, LDDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_cgetmatrix( M, N, d_C, LDDC, _cast_Cptr_magma(C.base_array), LDC );

     // Free memory on GPU
     magma_free(d_A);
     magma_free(d_B);
     magma_free(d_C);
}

void gmm_magma(const Matrix<complex<double>,2>& A, const Matrix<complex<double>,2>& B, Matrix<complex<double>,2>& C,
          char TRANSA, char TRANSB, complex<double> alpha, complex<double> beta)
{
     magma_int_t M, N, K; 
     magma_int_t LDA, LDB, LDC, LDDA, LDDB, LDDC;
     magma_int_t Am, An, Bm, Bn;
     magma_trans_t transA = magma_trans_const(TRANSA), transB = magma_trans_const(TRANSB);
     magmaDoubleComplex_ptr d_A, d_B, d_C;

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
     magma_zmalloc(&d_A, LDDA*An );
     magma_zmalloc(&d_B, LDDB*Bn );
     magma_zmalloc(&d_C, LDDC*N );

     // Copy data from host (CPU) to device (GPU)
     // Casting is required from std:complex<double> to  magmaDoubleComplex;
     magma_zsetmatrix( Am, An, _cast_Zptr_magma(A.base_array), LDA, d_A, LDDA );
     magma_zsetmatrix( Bm, Bn, _cast_Zptr_magma(B.base_array), LDB, d_B, LDDB );
     magma_zsetmatrix( M, N, _cast_Zptr_magma(C.base_array), LDC, d_C, LDDC );

     // This casting gives an warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]     
     magma_zgemm(transA, transB, M, N, K,
		 _cast_Z_magma(alpha), d_A, LDDA,
                                       d_B, LDDB,
		 _cast_Z_magma(beta),  d_C, LDDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_zgetmatrix( M, N, d_C, LDDC, _cast_Zptr_magma(C.base_array), LDC );

     // Free memory on GPU
     magma_free(d_A);
     magma_free(d_B);
     magma_free(d_C);
}

 /*************************************/
 /*Check Hermitian of the matrix*******/
 /*************************************/
 void check_Hermitian(const Matrix<complex<double>,2>& A)
 {
     if(A.L1!=A.L2) throw std::invalid_argument("Input for Hermitian is not square matrix!");
     double error=0; double norm=0;
     for(size_t i=0; i<A.L1; i++)
     {
         for(size_t j=i; j<A.L2; j++)
         {
             error+=abs(A(i,j)-conj(A(j,i)));
             norm+=abs(A(i,j));
         }
     }
     norm/=(A.L_f()*1.0);
     if(error/norm>1e-12) cout<<"Warning!!!!!Matrix is not Hermition!";
 }


 /*************************************/
 /*Diagonalize Hermitian Matrix********/
 /*************************************/
 void eigen(Matrix<complex<double>,2>& A, Matrix<double,1>& W, char JOBZ, char UPLO)
 {
     if(A.L1!=A.L2) throw std::invalid_argument("Input for eigen is not square matrix!");
     BL_INT N=A.L1; BL_INT info; 
     BL_INT lwork=-1; complex<double> work_test[1];
     BL_INT lrwork=(1<(3*N-2))?(3*N-2):1; double* rwork= new double[lrwork];

     FORTRAN_NAME(zheev)(&JOBZ,&UPLO,&N,(BL_COMPLEX16* )A.base_array,&N,(BL_DOUBLE* )W.base_array,
                         (BL_COMPLEX16* )work_test,&lwork,(BL_DOUBLE* )rwork,&info);

     //lwork=static_cast<BL_INT>(lround(work_test[0].real()));
     lwork=lround(work_test[0].real());
     complex<double>* work= new complex<double>[lwork];
     FORTRAN_NAME(zheev)(&JOBZ,&UPLO,&N,(BL_COMPLEX16* )A.base_array,&N,(BL_DOUBLE* )W.base_array,
                         (BL_COMPLEX16* )work,&lwork,(BL_DOUBLE* )rwork,&info);
     delete[] rwork;
     delete[] work;
}


 /*************************************/
 /*Diagonalize Hermitian Matrix********/     /* Using MAGMA library */
 /*************************************/
 void eigen_magma(Matrix<complex<double>,2>& A, Matrix<double,1>& W, char JOBZ, char UPLO)
 {
     if(A.L1!=A.L2) throw std::invalid_argument("Input for eigen is not square matrix!");
     magma_vec_t jobz = magma_vec_const(JOBZ);
     magma_uplo_t uplo = magma_uplo_const(UPLO);
     double *rwork, aux_rwork[1];
     magma_int_t lrwork;
     magma_int_t *iwork, aux_iwork[1];
     magma_int_t N=A.L1, info, lwork, liwork, lda;
     magmaDoubleComplex *h_work, aux_work[1];
     
     lda = N;

     // query for workspace sizes
     magma_zheevd( jobz, uplo,
		   N, NULL, lda, NULL,
		   aux_work,  -1,
		   aux_rwork, -1,
		   aux_iwork, -1,
		   &info );

     lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
     lrwork = (magma_int_t) aux_rwork[0];
     liwork = aux_iwork[0];

     // allocate memory on CPU
     magma_dmalloc_cpu(&rwork, lrwork);
     magma_imalloc_cpu(&iwork, liwork);
     magma_zmalloc_pinned(&h_work, lwork);

     // Perform operation using magma 
     magma_zheevd( jobz, uplo,
		   N, _cast_Zptr_magma(A.base_array), lda, W.base_array,
		   h_work, lwork,
		   rwork, lrwork,
		   iwork, liwork,
		   &info );
     
     // free allocated memory
     magma_free_cpu(rwork);
     magma_free_cpu(iwork);
     magma_free_pinned(h_work);
 }


 /******************************************/
 /*LU Decomposition a complex square Matrix*/
 /******************************************/
 template<> LUDecomp<complex<double>>::LUDecomp(const Matrix<complex<double>,2>& x)
 {
     if(x.L1!=x.L2) throw std::invalid_argument("Input for LU is not square matrix!");
     A=x; 
     ipiv=Matrix<BL_INT,1>(x.L1);
     BL_INT N=A.L1;
     //cout << "The originial value of ipiv: \n" <<  ipiv << std::endl;
     FORTRAN_NAME(zgetrf)(&N,&N,(BL_COMPLEX16* )A.base_array,&N,ipiv.base_array,&info);
     //cout << "The value of ipiv after: \n" <<  ipiv << std::endl;
     if(info<0) {cout<<"The "<<info<<"-th parameter is illegal!\n"; throw std::runtime_error(" ");} 
}
 

 /******************************************/
 /*LU Decomposition a complex square Matrix*/     /* Using MAGMA library */
 /******************************************/
  // In this routine, there are several ways of implementing LU decomp. 
  // All of them are equal basically
/* template<> LUDecomp_magma<complex<double>>::LUDecomp_magma(const Matrix<complex<double>,2>& x)
 {
     if(x.L1!=x.L2) throw std::invalid_argument("Input for LU is not square matrix!");
     A=x; 
     
     ipiv=Matrix<magma_int_t,1>(x.L1);
     magma_int_t N=A.L1, lda;
     //magma_int_t *ipv;
     lda = N;

     //cout << "The value of the pivot: " << ipiv << std::endl;

     //magma_imalloc_cpu(&ipv, N);
     //     cout << "The originial value of ipiv: \n" <<  ipiv << std::endl;
     //cout << "The value of ipiv after: \n" <<  ipv[0] << " "<< ipv[1] << " " << ipv[2] << std::endl;
     //cout << "The value of ipiv after: \n" <<  ipv[0] << std::endl;
     
     magma_zgetrf( N, N, _cast_Zptr_magma(A.base_array), lda, ipiv.base_array, &info);

     cout << "The value of the pivot: " << ipiv << std::endl;

     //magma_zgetrf( N, N, _cast_Zptr_magma(A.base_array), lda, ipv, &info);
     //cout << "The value of ipiv after: \n" <<  ipv[0] << " "<< ipv[1] << " " << ipv[2] << std::endl;
     cout << "info: " << info << "\n" << std::endl;

     //magma_free_cpu(ipv);
     if(info<0) {cout<<"The "<<info<<"-th parameter is illegal!\n"; throw std::runtime_error(" ");} 
     }*/


 /******************************************/
 /*LU Decomposition a complex square Matrix*/     /* Using MAGMA library */
 /******************************************/
template<> LUDecomp_magma<complex<double>>::LUDecomp_magma(const Matrix<complex<double>,2>& x)
 {
     if(x.L1!=x.L2) throw std::invalid_argument("Input for LU is not square matrix!");
     A=x; 
     ipiv=Matrix<magma_int_t,1>(x.L1);
     magma_int_t N=A.L1;
     //cout << "The originial value of ipiv: \n" <<  ipiv << std::endl;
     magma_zgetrf( N, N, _cast_Zptr_magma(A.base_array), N, ipiv.base_array, &info);
     //cout << "The value of ipiv after: \n" <<  ipiv << std::endl;
     if(info<0) {cout<<"The "<<info<<"-th parameter is illegal!\n"; throw std::runtime_error(" ");} 
}

 
 /************************/
 /*Determinant of  Matrix*/
 /************************/ 
complex<double> determinant(const LUDecomp<complex<double>>& x)
 {
     if(x.info>0) return 0;
    
     complex<double> det={1,0};
     BL_INT L=x.ipiv.L1;
     for(BL_INT i=0;i<L;i++)
     {
         if(x.ipiv(i)!=(i+1)) det*=(-x.A(i,i));
         else det*=x.A(i,i);
     }
     return det;
}

 /*****************************************/
 /*Get Log(|det|) and det/|det| of  Matrix*/
 /*****************************************/
void lognorm_phase_determinant(const LUDecomp<complex<double>>& x, complex<double>& lognorm, complex<double>& phase)
 {
     if(x.info>0)
     {
         cout<<"WARNING!!!! lognorm_phase_determinant function has zero determinant!\n";
         lognorm=complex<double>(-1e300,0.0);
         phase=complex<double>(1.0,0.0);
         return;
     }

     lognorm=complex<double>(0.0,0.0); phase=complex<double>(1.0,0.0);
     BL_INT L=x.ipiv.L1;
     for(BL_INT i=0;i<L;i++)
     {
         lognorm+=log(abs(x.A(i,i)));
         if(x.ipiv(i)!=(i+1)) phase*=(-x.A(i,i)/abs(x.A(i,i)));
         else phase*=(x.A(i,i)/abs(x.A(i,i)));
     }
     return;
 }


 /****************************/
 /*Log Determinant of  Matrix*/
 /****************************/
 complex<double> log_determinant(const LUDecomp<complex<double>>& x)
 {
     complex<double> log_det,phase; 
     lognorm_phase_determinant(x,log_det,phase);
     log_det+=log(phase);
     return log_det;
 }



 /************************/
 /*Determinant of  Matrix*/     /* Using MAGMA library */
 /************************/
 complex<double> determinant_magma(const LUDecomp_magma<complex<double>>& x)
 {
     if(x.info>0) return 0;
    
     complex<double> det={1,0};
     magma_int_t L=x.ipiv.L1;
     for(magma_int_t i=0;i<L;i++)
     {
         if(x.ipiv(i)!=(i+1)) det*=(-x.A(i,i));
         else det*=x.A(i,i);
     }
     //cout << "\n";
     //cout << det << std::endl;
     return det;
 }

 /*****************************************/
 /*Get Log(|det|) and det/|det| of  Matrix*/     /* Using MAGMA library */
 /*****************************************/
 void lognorm_phase_determinant_magma(const LUDecomp_magma<complex<double>>& x, complex<double>& lognorm, complex<double>& phase)
 {
     if(x.info>0)
     {
         cout<<"WARNING!!!! lognorm_phase_determinant function has zero determinant!\n";
         lognorm=complex<double>(-1e300,0.0);
         phase=complex<double>(1.0,0.0);
         return;
     }

     lognorm=complex<double>(0.0,0.0); phase=complex<double>(1.0,0.0);
     magma_int_t L=x.ipiv.L1;
     for(magma_int_t i=0;i<L;i++)
     {
         lognorm+=log(abs(x.A(i,i)));
         if(x.ipiv(i)!=(i+1)) phase*=(-x.A(i,i)/abs(x.A(i,i)));
         else phase*=(x.A(i,i)/abs(x.A(i,i)));
     }
     //cout << "\n";
     //cout << phase << std::endl;
     return;
 }


 /****************************/
 /*Log Determinant of  Matrix*/     /* Using MAGMA library */
 /****************************/
 complex<double> log_determinant_magma(const LUDecomp_magma<complex<double>>& x)
 {
     complex<double> log_det,phase; 
     lognorm_phase_determinant_magma(x,log_det,phase);
     log_det+=log(phase);
     return log_det;
 }


 /*********************************************************************************************************************/
 /*Inverse of  Matrix: If determinant of the matrix is out of machine precision, inverse should be fine, since it solve*
  *The linear equation, every small value is well defined                                                             */
 /*********************************************************************************************************************/ 
Matrix<complex<double>,2> inverse(const LUDecomp<complex<double>>& x)
 {
     Matrix<complex<double>,2> A=x.A; //We know x.A own the matrix
     BL_INT N=A.L1; BL_INT info;

     BL_INT lwork=-1; complex<double> work_test[1];
     FORTRAN_NAME(zgetri)(&N,(BL_COMPLEX16* )A.base_array,&N,x.ipiv.base_array,(BL_COMPLEX16* )work_test,&lwork,&info);

     lwork=lround(work_test[0].real());
     complex<double>* work= new complex<double>[lwork];
     FORTRAN_NAME(zgetri)(&N,(BL_COMPLEX16* )A.base_array,&N,x.ipiv.base_array,(BL_COMPLEX16* )work,&lwork,&info);
     delete[] work; 

     return A;
}

 /*********************************************************************************************************************/
 /*Inverse of  Matrix: If determinant of the matrix is out of machine precision, inverse should be fine, since it solve*
  *The linear equation, every small value is well defined                                                             */      /* Using MAGMA library */
 /*********************************************************************************************************************/
 Matrix<complex<double>,2> inverse_magma(const LUDecomp_magma<complex<double>>& x)
 {
     Matrix<complex<double>,2> A=x.A; //We know x.A own the matrix
     //magmaDoubleComplex *work, tmp;
     magmaDoubleComplex_ptr d_A , dwork;
     magma_int_t N=A.L1, lda, ldda, info, ldwork;
     lda = N;
     ldda = ((lda+31)/32)*32;
     ldwork = N * magma_get_zgetri_nb(N); // magma_get_zgetri_nb optimizes the blocksize

     //magma_int_t lwork=-1
     /* Calling lapackf77_zgetri requires magma_lapack.h which has a conflict with Hao's functions. 
        Query for a workspace size with magma_zgetri_gpu will give illegal info value (-6), although the final result is still correct for the current test case. */
     // query for workspace size
     //cout << info << std::endl;
     //lapackf77_zgetri( &N, NULL, &lda, NULL, &tmp, &lwork, &info );
     //magma_zgetri_gpu( N, NULL, lda, NULL, tmp, lwork, &info );

     // We can also use lapack zgetri function from MKL 
     //FORTRAN_NAME(zgetri)( &N, NULL, &lda, NULL, reinterpret_cast<BL_COMPLEX16*>( &tmp ), &lwork, &info );
     //cout << info << std::endl;

     //lwork = int( MAGMA_Z_REAL(tmp));
     
     //magma_zmalloc_cpu( &work, lwork );
     magma_zmalloc( &d_A, ldda*N );
     magma_zmalloc( &dwork, ldwork );

     // copy matrix from CPU to GPU
     magma_zsetmatrix( N, N, _cast_Zptr_magma(A.base_array), lda, d_A, ldda );
     
     // calculate the inverse matrix with zgetri 
     //cout << info << std::endl;
     magma_zgetri_gpu( N, d_A, ldda, x.ipiv.base_array, dwork, ldwork, &info );
     //cout << info << std::endl;
     // copy matrix from GPU to CPU
     magma_zgetmatrix( N, N, d_A, ldda, _cast_Zptr_magma(A.base_array), lda );

     magma_free( d_A );
     magma_free( dwork );
     //magma_free_cpu( work );

     return A;
 }


 /******************************************************/
 /*Solve Linear Equation of the matrix A*M=B: M=A^{-1}B*/
 /******************************************************/
Matrix<complex<double>,2> solve_lineq(const LUDecomp<complex<double>>& x, const Matrix<complex<double>,2>& B, char TRANS)
 {
     if(x.A.L1!=B.L1) throw std::invalid_argument("Input size for solving linear equation is not consistent!");
     Matrix<complex<double>,2> M; M=B;
     BL_INT N=B.L1; BL_INT NRHS=B.L2; BL_INT info;
     FORTRAN_NAME(zgetrs)(&TRANS,&N,&NRHS,(BL_COMPLEX16* )x.A.base_array,&N,x.ipiv.base_array,(BL_COMPLEX16* )M.base_array,&N,&info);
     if(info!=0) 
     {
         cout<<"Solve linear equation is not suceesful: "<<info<<"-th parameter is illegal! \n"; 
         throw std::runtime_error(" ");
     }
     return M;
}


 /******************************************************/
 /*Solve Linear Equation of the matrix A*M=B: M=A^{-1}B*/      /* Using MAGMA library */
 /******************************************************/
Matrix<complex<double>,2> solve_lineq_magma(const LUDecomp_magma<complex<double>>& x, const Matrix<complex<double>,2>& B, char TRANS)
 {
     if(x.A.L1!=B.L1) throw std::invalid_argument("Input size for solving linear equation is not consistent!");
     Matrix<complex<double>,2> M; M=B;
     magma_trans_t trans = magma_trans_const(TRANS);
     magmaDoubleComplex_ptr d_A, d_B;
     magma_int_t N=B.L1, nrhs=B.L2, lda, ldda, ldb, lddb, info;
     lda = N;
     ldda = ((lda+31)/32)*32;
     ldb = N;
     lddb = ((ldb+31)/32)*32;
     
     //allocate memory on GPU
     magma_zmalloc( &d_A, ldda*N );
     magma_zmalloc( &d_B, lddb*N );

     // copy matrix from CPU to GPU
     magma_zsetmatrix( N, N, _cast_Zptr_magma(x.A.base_array), lda, d_A, ldda );
     magma_zsetmatrix( N, N, _cast_Zptr_magma(M.base_array), ldb, d_B, lddb );

     magma_zgetrs_gpu( trans, N, nrhs, d_A, ldda, x.ipiv.base_array, d_B, lddb, &info );
     //cout << info << std::endl;
     // copy matrix from GPU to CPU
     magma_zgetmatrix( N, N, d_B, lddb, _cast_Zptr_magma(M.base_array), ldb );

     // free memory
     magma_free( d_A );
     magma_free( d_B );

     if(info!=0) 
     {
         cout<<"Solve linear equation is not suceesful: "<<info<<"-th parameter is illegal! \n"; 
         throw std::runtime_error(" ");
     }
     return M;
}


 /******************************/
 /*QR decompostion of matrix ph*/
 /******************************/
double QRMatrix(Matrix<complex<double>,2>& ph)
 {
     BL_INT L=ph.L1; BL_INT N=ph.L2; BL_INT info;
     BL_INT lwork=-1; complex<double> work_test[1];
     complex<double>* tau= new complex<double>[N];

     FORTRAN_NAME(zgeqrf) (&L,&N,(BL_COMPLEX16* )ph.base_array,&L,(BL_COMPLEX16* )tau,(BL_COMPLEX16* )work_test,&lwork,&info);

     lwork=lround(work_test[0].real());
     complex<double>* work= new complex<double>[lwork];
     FORTRAN_NAME(zgeqrf) (&L,&N,(BL_COMPLEX16* )ph.base_array,&L,(BL_COMPLEX16* )tau,(BL_COMPLEX16* )work,&lwork,&info);
     if(info!=0) {cout<<"QR run is not suceesful: "<<info<<"-th parameter is illegal! \n"; throw std::runtime_error(" ");}

     complex<double> det={1.0,0.0}; for (size_t i=0; i<ph.L2; i++)  det*=ph(i,i); 

     FORTRAN_NAME(zungqr) (&L,&N,&N,(BL_COMPLEX16* )ph.base_array,&L,(BL_COMPLEX16* )tau,(BL_COMPLEX16* )work,&lwork,&info);

     if(det.real()<0) {det=-det; for(size_t i=0; i<ph.L1; i++) ph(i,0)=-ph(i,0);}

     delete[] tau;delete[] work;

     return det.real();
 }

/******************************/
 /*QR decompostion of matrix ph*/     /* Using MAGMA library */
 /******************************/
double QRMatrix_magma(Matrix<complex<double>,2>& ph)
 {
     magmaDoubleComplex *tau, *h_work;
     //magmaDoubleComplex tmp[1];
     magma_int_t L=ph.L1, N=ph.L2, lda, lwork=-1, nb, info;

     lda = L;
     nb = magma_get_zgeqrf_nb(L);
     //cout << "nb value and L value : " << nb << " and " << L << std::endl;
     

     //FORTRAN_NAME(zgeqrf) ( &L, &N, NULL, &L, NULL, reinterpret_cast<BL_COMPLEX16*>(tmp), &lwork, &info );
     //lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
     //cout << "lwork value : " << lwork << std::endl;

     // The value of lwork calculated by max( N*nb, 2*nb*nb ) should be larger than method above (query space with zgeqrf). If there is an error, try to activate zgeqrf.
     lwork = std::max( lwork, std::max( N*nb, 2*nb*nb ));
     // cout << "lwork value : " << lwork << std::endl;

     // Allocate memory 
     magma_zmalloc_cpu(&tau, N);  // should be min(L,N) but N is always smaller 
     magma_zmalloc_cpu(&h_work, lwork);

     // perform zgeqrf
     magma_zgeqrf(L, N, _cast_Zptr_magma(ph.base_array), lda, tau, h_work, lwork, &info);

     if(info!=0) {cout<<"QR run is not suceesful: "<<info<<"-th parameter is illegal! \n"; throw std::runtime_error(" ");}

     complex<double> det={1.0,0.0}; for (size_t i=0; i<ph.L2; i++)  det*=ph(i,i); 
     
     magma_zungqr( L, N, N, _cast_Zptr_magma(ph.base_array), lda, tau, h_work, lwork, &info );

     // This method gives the same result except that it recomputes T matrices
     //magma_zungqr2( L, N, N, _cast_Zptr_magma(ph.base_array), lda, tau, &info );

     if(det.real()<0) {det=-det; for(size_t i=0; i<ph.L1; i++) ph(i,0)=-ph(i,0);}

     // free memory 
     magma_free_cpu( tau );
     magma_free_cpu( h_work );
     
     return det.real();
 }


 /*******************************/
 /*Diagonal array multipy matrix*/
 /*******************************/
 Matrix<complex<double>,2> D_Multi_Matrix(const Matrix<complex<double>,1>& D,const Matrix<complex<double>,2>& ph)
 {
     if(D.L1!=ph.L1) {cout<<"D_Multi_Matrix input error: D.L1!=ph.L1! \n"; throw std::runtime_error(" ");} 
     Matrix<complex<double>,2> ph_new(ph.L1,ph.L2);
     for(size_t i=0; i<ph.L1; i++)
     {
         for(size_t j=0; j<ph.L2; j++) ph_new(i,j)=D(i)*ph(i,j);
     }
     return ph_new; 
 }
 
} //end namespace matrix_hao_lib
