#ifndef MATRIX_HAO_LINALG
#define MATRIX_HAO_LINALG

/*#define "lib_hao/matrix_2d.h"
  #define "lib_hao/blas_lapack_trait.h"*/

#include <cmath>
#include "matrix_2d.h"
#include "lib_hao/blas_lapack_traits.h"

using std::cout;
//using std::conj;

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
        //int_t N=A.L1, LDA, lwork=-1, lrwork=-1, iwork[1], liwork=-1, info=-1;
        //std::complex<double> work[1];
        //double rwork[1];
        int_t N=A.L1, LDA, info=-1;
        LDA = N;

        /*        _impl->heevd(JOBZ, UPLO, N, A.base_array, 
                     LDA, W.base_array, work, 
                     lwork, rwork, lrwork, iwork,
                     liwork, info);
        */
        _impl->heevd(JOBZ, UPLO, N, A.base_array, 
                     LDA, W.base_array, &info);
    }     

};

/*******************************************/
/*LU Decomposition of Complex double Matrix*/
/*******************************************/
 template <typename T, typename _Int_t> class LU_decomp
{
public:
    // This encapsulates the real linear algebra library implementation
    typedef blas_lapack_traits<_Int_t> linalg_t;
    typedef typename blas_lapack_traits<_Int_t>::int_t int_t;

protected:
    linalg_t *_impl;

public:
    Matrix<T,2> A;
    Matrix<int_t,1> ipiv;
    int_t info;


    LU_decomp()
        : _impl(nullptr) {}
    LU_decomp(blas_lapack_traits<_Int_t> *linalg_impl)
        : _impl(linalg_impl)
    {}
    LU_decomp(const Matrix<T,2>& x, blas_lapack_traits<_Int_t> *linalg_impl)
        : _impl(linalg_impl)
    {
        assign(x);
    }

    void assign(const Matrix<T,2>& x)
    {
        if (x.L1 != x.L2) 
            throw std::invalid_argument("Input for LU is not square matrix!");
        A = x;
        ipiv = Matrix<int_t,1>(x.L1);
        int_t N = A.L1;
        _impl->getrf(N, N, A.base_array, N, ipiv.base_array, &info);

        if (info < 0) 
        {
            cout<<"The "<<info<<"-th parameter is illegal!\n"; 
            throw std::runtime_error(" "); 
        }
    }

    LU_decomp(const LU_decomp<T,_Int_t>& x) 
    {
        A = x.A;
        ipiv = x.ipiv;
        info = x.info;
        _impl = _impl;
    }
    LU_decomp(LU_decomp<T,_Int_t>&& x) 
    {
        A = std::move(x.A);
        ipiv = std::move(x.ipiv);
        info = x.info;
        _impl = _impl;
    }
    ~LU_decomp() {}
    LU_decomp<T,_Int_t>& operator=(const LU_decomp<T,_Int_t>& x) {A=x.A;ipiv=x.ipiv;info=x.info;return *this;}
    LU_decomp<T,_Int_t>& operator = (LU_decomp<T,_Int_t>&& x) {A=std::move(x.A);ipiv=std::move(x.ipiv);info=x.info;return *this;}


/************************/
/*Determinant of  Matrix*/
/************************/
    T determinant_in()
    {
        if (info > 0) return 0;

        T det={1,0};
        int_t L = A.L1;
        
        for (int_t i = 0; i < L ; i++)
            {
                if (ipiv(i) != (i+1)) det *= (-A(i,i));
                else det *= A(i,i);
            }
        return det;
    }

 /*****************************************/
 /*Get Log(|det|) and det/|det| of  Matrix*/
 /*****************************************/
void lognorm_phase_determinant_in(T &lognorm, T &phase)
 {
     if(info>0)
     {
         cout<<"WARNING!!!! lognorm_phase_determinant function has zero determinant!\n";
         lognorm= T(-1e300,0.0);
         phase= T(1.0,0.0);
         return;
     }

     lognorm= T(0.0,0.0); phase= T(1.0,0.0);
     int_t L = ipiv.L1;
     for(int_t i=0;i<L;i++)
     {
         lognorm+=log(abs(A(i,i)));
         if(ipiv(i)!=(i+1)) phase*=(-A(i,i)/abs(A(i,i)));
         else phase*=(A(i,i)/abs(A(i,i)));
     }
     return;
 }


 /****************************/
 /*Log Determinant of  Matrix*/
 /****************************/
 T log_determinant_in()
 {
     T log_det,phase; 
     lognorm_phase_determinant_in(log_det,phase);
     log_det+=log(phase);
     return log_det;
 }

 /**************************************/
 /************Inverse Matrix************/
 /**************************************/
 Matrix<T,2> inverse_in()
 {
     int_t N = A.L1, LDA;
     LDA = N;

     _impl->getri(N, A.base_array, LDA, ipiv.base_array, &info);

     return A;
 }

 /******************************************************/
 /*Solve Linear Equation of the matrix A*M=B: M=A^{-1}B*/
 /******************************************************/
 Matrix<T,2> solve_lineq_in(const Matrix<T,2> &B, char TRANS = 'N')
 {
     if(A.L1!=B.L1) throw std::invalid_argument("Input size for solving linear equation is not consistent!");
     Matrix<T,2> M; M=B;
     int_t N=B.L1, NRHS=B.L2, LDA, LDB;
     LDA = N;
     LDB = N;

     _impl->getrs(TRANS, N, NRHS, A.base_array, LDA, ipiv.base_array, M.base_array, LDB, &info );

     if(info!=0)
         {   
             cout<<"Solve linear equation is not suceesful: "<<info<<"-th parameter is illegal! \n";
             throw std::runtime_error(" ");
         }

     return M;

 }

};

}//end namespace matrix_hao_lib

#endif
