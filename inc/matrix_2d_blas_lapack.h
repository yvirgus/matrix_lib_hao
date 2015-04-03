#ifndef MATRIX_HAO_BLAS_LAPACK
#define MATRIX_HAO_BLAS_LAPACK

#include "magma.h"
#include "matrix_define.h"

namespace matrix_hao_lib
{

 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */
 /*************************************/

  /* void gmm(const Matrix<float,2>& A, const Matrix<float,2>& B, Matrix<float,2>& C, 
          char TRANSA='N', char TRANSB='N', float alpha=1, float beta=0);

 void gmm(const Matrix<double,2>& A, const Matrix<double,2>& B, Matrix<double,2>& C,
          char TRANSA='N', char TRANSB='N', double alpha=1, double beta=0);

 void gmm(const Matrix<std::complex<float>,2>& A, const Matrix<std::complex<float>,2>& B, Matrix<std::complex<float>,2>& C,
          char TRANSA='N', char TRANSB='N', std::complex<float> alpha=1,std::complex<float> beta=0);

 void gmm(const Matrix<std::complex<double>,2>& A, const Matrix<std::complex<double>,2>& B, Matrix<std::complex<double>,2>& C,
 char TRANSA='N', char TRANSB='N', std::complex<double> alpha=1, std::complex<double> beta=0);  */

 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */ /* Using MAGMA library */
 /*************************************/

 void gmm_magma(const Matrix<float,2>& A, const Matrix<float,2>& B, Matrix<float,2>& C, 
          char TRANSA='N', char TRANSB='N', float alpha=1, float beta=0);

 void gmm_magma(const Matrix<double,2>& A, const Matrix<double,2>& B, Matrix<double,2>& C,
          char TRANSA='N', char TRANSB='N', double alpha=1, double beta=0);

 void gmm_magma(const Matrix<std::complex<float>,2>& A, const Matrix<std::complex<float>,2>& B, Matrix<std::complex<float>,2>& C,
          char TRANSA='N', char TRANSB='N', std::complex<float> alpha=1,std::complex<float> beta=0);

 void gmm_magma(const Matrix<std::complex<double>,2>& A, const Matrix<std::complex<double>,2>& B, Matrix<std::complex<double>,2>& C,
          char TRANSA='N', char TRANSB='N', std::complex<double> alpha=1, std::complex<double> beta=0);


 /*************************************/
 /*Diagonalize Hermitian Matrix********/
 /*************************************/
  void check_Hermitian(const Matrix<std::complex<double>,2>& A);
  /*   void eigen(Matrix<std::complex<double>,2>& A, Matrix<double,1>& W, char JOBZ='V', char UPLO='U'); */

 /*************************************/
 /*Diagonalize Hermitian Matrix********/    /* Using MAGMA library */
 /*************************************/
 void eigen_magma(Matrix<std::complex<double>,2>& A, Matrix<double,1>& W, char JOBZ='V', char UPLO='U');


 /*******************************************/
 /*LU Decomposition of Complex double Matrix*/
 /*******************************************/
 /*template <class T> class LUDecomp
 {
     public:
     Matrix<T,2> A;
     Matrix<BL_INT,1> ipiv;
     BL_INT info;
   
     LUDecomp() {}
     LUDecomp(const Matrix<T,2>& x);
     LUDecomp(const LUDecomp<T>& x) {A=x.A;ipiv=x.ipiv;info=x.info;}
     LUDecomp(LUDecomp<T>&& x) {A=std::move(x.A);ipiv=std::move(x.ipiv);info=x.info;}
     ~LUDecomp() {}
     LUDecomp<T>& operator = (const LUDecomp<T>& x) {A=x.A;ipiv=x.ipiv;info=x.info;return *this;}
     LUDecomp<T>& operator = (LUDecomp<T>&& x) {A=std::move(x.A);ipiv=std::move(x.ipiv);info=x.info;return *this;}
     };*/

 /*******************************************/
 /*LU Decomposition of Complex double Matrix*/    /* Using MAGMA library */
 /*******************************************/
 template <class T> class LUDecomp_magma
 {
     public:
     Matrix<T,2> A;
     Matrix<magma_int_t,1> ipiv;
     magma_int_t info;
   
     LUDecomp_magma() {}
     LUDecomp_magma(const Matrix<T,2>& x);
     LUDecomp_magma(const LUDecomp_magma<T>& x) {A=x.A;ipiv=x.ipiv;info=x.info;}
     LUDecomp_magma(LUDecomp_magma<T>&& x) {A=std::move(x.A);ipiv=std::move(x.ipiv);info=x.info;}
     ~LUDecomp_magma() {}
     LUDecomp_magma<T>& operator = (const LUDecomp_magma<T>& x) {A=x.A;ipiv=x.ipiv;info=x.info;return *this;}
     LUDecomp_magma<T>& operator = (LUDecomp_magma<T>&& x) {A=std::move(x.A);ipiv=std::move(x.ipiv);info=x.info;return *this;}
 };

 /************************/
 /*Determinant of  Matrix*/
 /************************/
 /* std::complex<double> determinant(const LUDecomp<std::complex<double>>& x);
 void lognorm_phase_determinant(const LUDecomp<std::complex<double>>& x, std::complex<double>& lognorm, std::complex<double>& phase);
 std::complex<double> log_determinant(const LUDecomp<std::complex<double>>& x); */

 /************************/
 /*Determinant of  Matrix*/    /* Using MAGMA library */
 /************************/
 std::complex<double> determinant_magma(const LUDecomp_magma<std::complex<double>>& x);
 void lognorm_phase_determinant_magma(const LUDecomp_magma<std::complex<double>>& x, std::complex<double>& lognorm, std::complex<double>& phase);
 std::complex<double> log_determinant_magma(const LUDecomp_magma<std::complex<double>>& x);


 /********************/
 /*Inverse of  Matrix*/
 /********************/
 /* Matrix<std::complex<double>,2> inverse(const LUDecomp<std::complex<double>>& x);*/


 /******************************************************/
 /*Solve Linear Equation of the matrix A*M=B: M=A^{-1}B*/
 /******************************************************/
 /* Matrix<std::complex<double>,2> solve_lineq(const LUDecomp<std::complex<double>>& x, const Matrix<std::complex<double>,2>& B
    ,char TRANS='N'); */


 /***********************************************************/
 /*QR decompostion of matrix ph, return the determinant of R*/
 /***********************************************************/
 /* double QRMatrix(Matrix<std::complex<double>,2>& ph);*/


 /*******************************/
 /*Diagonal array multipy matrix*/
 /*******************************/
  Matrix<std::complex<double>,2> D_Multi_Matrix(const Matrix<std::complex<double>,1>& D,const Matrix<std::complex<double>,2>& ph); 

 
}//end namespace matrix_hao_lib

#endif
