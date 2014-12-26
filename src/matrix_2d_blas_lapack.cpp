#include <cmath>
#include "matrix_all.h"

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


 /******************************************/
 /*LU Decomposition a complex square Matrix*/
 /******************************************/
 template<> LUDecomp<complex<double>>::LUDecomp(const Matrix<complex<double>,2>& x)
 {
  if(x.L1!=x.L2) throw std::invalid_argument("Input for LU is not square matrix!");
  A=x; 
  ipiv=Matrix<BL_INT,1>(x.L1);
  BL_INT N=A.L1;
  FORTRAN_NAME(zgetrf)(&N,&N,(BL_COMPLEX16* )A.base_array,&N,ipiv.base_array,&info);
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


 /*********************************************************************************************************************/
 /*Inverse of  Matrix: If determinant of the matrix is outof machine precision, inverse should be fine, since it solve*
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

 /******************************************************/
 /*Solve Linear Equation of the matrix A*M=B: M=A^{-1}B*/
 /******************************************************/
 Matrix<complex<double>,2> solve_lineq(const LUDecomp<complex<double>>& x, const Matrix<complex<double>,2>& B, char TRANS)
 {
  if(x.A.L1!=B.L1) throw std::invalid_argument("Input size for solving linear equation is not consistent!");
  Matrix<complex<double>,2> M; M=B;
  BL_INT N=B.L1; BL_INT NRHS=B.L2; BL_INT info;
  FORTRAN_NAME(zgetrs)(&TRANS,&N,&NRHS,(BL_COMPLEX16* )x.A.base_array,&N,x.ipiv.base_array,(BL_COMPLEX16* )M.base_array,&N,&info);
  if(info!=0) {cout<<"Solve linear equation is not suceesful: "<<info<<"-th parameter is illegal! \n"; throw std::runtime_error(" ");}
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
