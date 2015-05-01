/*
// This is the test of zheevd with our own code, testing for N = 210
(0.120625, 0) (0.0623417, -0.490279) (0.306079, -0.816414) (0.99718, -0.424599)
(0.0623417, 0.490279) (0.995178, 0) (0.191079, -0.583414) (0.761785, -0.0741798)
(0.306079, 0.816414) (0.191079, 0.583414) (0.267303, 0) (0.218683, -0.620994)
(0.99718, 0.424599) (0.761785, 0.0741798) (0.218683, 0.620994) (0.578585, 0)
(0.120625,0) (0.0623417,-0.490279) (0.306079,-0.816414) (0.99718,-0.424599)
(0.0623417,0.490279) (0.995178,0) (0.191079,-0.583414) (0.761785,-0.0741798)
(0.306079,0.816414) (0.191079,0.583414) (0.267303,0) (0.218683,-0.620994)
(0.99718,0.424599) (0.761785,0.0741798) (0.218683,0.620994) (0.578585,0)
aux_work, aux_rwork, aux_iwork : 44520 89251 1053
lwork, lrwork, liwork : 44520 89251 1053

CPU time for zheevd : 12.6611 with size N : 210


// This is the test of zheevd with MAGMA testing for N = 210
MAGMA 1.6.1  compiled for CUDA capability >= 2.0
CUDA runtime 5050, driver 6000. OpenMP threads 1. ACML 5.1.0.0
ndevices 2
device 0: Tesla M2075, 1147.0 MHz clock, 5375.4 MB memory, capability 2.0
device 1: Tesla M2075, 1147.0 MHz clock, 5375.4 MB memory, capability 2.0
Usage: ./testing_zheevd [options] [-h|--help]

using: jobz = Vectors needed, uplo = Lower
    N   CPU Time (sec)   GPU Time (sec)
=======================================
 aux_work, aux_rwork, aux_iwork : 44520 89251 1053
 lwork, lrwork, liwork : 44520 89251 1053
(0.120625, 0) (0.0623417, -0.490279) (0.306079, -0.816414) (0.99718, -0.424599)
(0.0623417, 0.490279) (0.995178, 0) (0.191079, -0.583414) (0.761785, -0.0741798)
(0.306079, 0.816414) (0.191079, 0.583414) (0.267303, 0) (0.218683, -0.620994)
(0.99718, 0.424599) (0.761785, 0.0741798) (0.218683, 0.620994) (0.578585, 0)
  210      0.07             0.05
    | S_magma - S_lapack | / |S| = 3.04e-18   ok

*/ 

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdio>
#include <fstream>
#include <cassert>
#include <vector>
#include <complex>
#include <iostream>
                         //#include <ctime>
#include <sys/time.h>

typedef struct
{
  double real, imag;
} doubleccomplex;

#define FORTRAN_NAME(x) x##_
typedef  int              BL_INT;
typedef  float            BL_FLOAT;
typedef  double           BL_DOUBLE;
typedef  doubleccomplex   BL_COMPLEX16;

extern "C" void zheevd(char jobz, char uplo, int n, doubleccomplex *a, int lda, double *w, int *info);
extern "C" void zheevd_(char *jobz, char *uplo, int *n, doubleccomplex *a, int *lda, double *w, doubleccomplex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info, int jobz_len=1, int uplo_len=1);
extern "C" void zlarnv(int idist, int *iseed, int n, doubleccomplex *x);
extern "C" void zlarnv_(int *idist, int *iseed, int *n, doubleccomplex *x);

using namespace std;

namespace my_tests
{

#define Aij(i,j)  A[i + j*lda]
    // --------------------
    // Make a matrix symmetric/Hermitian.
    // Makes diagonal real.
    // Sets Aji = conj( Aij ) for j < i, that is, copy lower triangle to upper triangle.
    //extern "C"
    void magma_zmake_hermitian( BL_INT N, complex<double>* A, BL_INT lda )
    {
        BL_INT i, j;
        for( i=0; i<N; ++i ) {
	  Aij(i,i) = std::complex<double>( Aij(i,i).real() , 0. );
            for( j=0; j<i; ++j ) {
	      Aij(j,i) = std::complex<double>( Aij(i,j).real(), Aij(i,j).imag() );
            }
        }
        
    }

#undef Aij 
   

    double magma_wtime( void )
    {
        struct timeval t;
        gettimeofday( &t, NULL );
        return t.tv_sec + t.tv_usec*1e-6;
    }

    void fill_random(complex<double>* A, BL_INT N)
    {
        BL_INT lapack_ran_ISEED[4] = { 0, 0, 0, 1 };
        BL_INT ione = 1;
        BL_INT mtx_size = N*N;
        FORTRAN_NAME(zlarnv)(&ione, lapack_ran_ISEED, &mtx_size, reinterpret_cast<doubleccomplex*>(A) );
    }

    void zheevd_size_test()
    {
        double cpu_time;
        //std::clock_t cpu_time;
        BL_INT N=210, lda=N, lwork=-1, lrwork=-1, aux_iwork[1], liwork=-1, info;
        char jobz='V', uplo='L';
        complex<double> *A = new complex<double>[N*lda];
        complex<double> *w = new complex<double>[N];
        complex<double> aux_work[1];
        double aux_rwork[1];

        // fill matrix A with random complex number 
        fill_random(A, N);

                
        // Make matrix A into Hermitian
        magma_zmake_hermitian( N, A, lda);

        for (int i=0; i<=3; i++ ){
            for (int j=0; j<=3; j++ ){
                cout << A[i + j*lda] << " ";
            }
            cout << endl;
        }

        FORTRAN_NAME(zheevd)(&jobz, &uplo, &N, reinterpret_cast<BL_COMPLEX16*>(A), &lda, 
                     reinterpret_cast<BL_DOUBLE*>(w), reinterpret_cast<BL_COMPLEX16*>(aux_work),
                     &lwork, aux_rwork, &lrwork, aux_iwork, &liwork, &info);

        cout << " aux_work, aux_rwork, aux_iwork : " << 
            aux_work[0].real() << " " << aux_rwork[0] << " " << aux_iwork[0] << endl; 

        lwork = lround( aux_work[0].real());
        complex<double> *work = new complex<double>[lwork];

        lrwork = lround(aux_rwork[0]);
        double *rwork = new double[lrwork];

        liwork = aux_iwork[0];
        BL_INT *iwork = new BL_INT[liwork];

        cout << " lwork, lrwork, liwork : " << 
            lwork << " " << lrwork << " " << liwork << endl; 

        cpu_time = magma_wtime();
        //cpu_time = std::clock();
        FORTRAN_NAME(zheevd)(&jobz, &uplo, &N, reinterpret_cast<BL_COMPLEX16*>(A), &lda, 
                     reinterpret_cast<BL_DOUBLE*>(w), reinterpret_cast<BL_COMPLEX16*>(work),
                     &lwork, rwork, &lrwork, iwork, &liwork, &info);
        cpu_time = magma_wtime() - cpu_time;
        //cpu_time = std::clock() - cpu_time;
        cout << "\nCPU time for zheevd : " << cpu_time << " with size N : " << N << "\n" <<  endl;
  
        delete[] A;
        delete[] w;
        delete[] work;
        delete[] rwork;
        delete[] iwork;


    }

 void zheevd_test()
 {
#if defined(_OPENMP)
    int omp_threads = 0;
    #pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
    }
    printf( "OpenMP threads %d.\n", omp_threads );
#else
    printf( "Code not compiled with OpenMP.\n" );
#endif

     zheevd_size_test();
 }

} //end my_tests namespace


int main(int argc, char** argv)
{
    using my_tests::zheevd_test;

    zheevd_test();
    return 0;
}
