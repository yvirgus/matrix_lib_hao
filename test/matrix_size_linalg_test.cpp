#include "matrix_all.h"
#include <cstdio>
#include <fstream>
#include <cassert>
#include <vector>

#include "lib_hao/matrix_linalg.h"
#include "lib_hao/f77lapack_traits.h"
#include "lib_hao/magma_traits.h"
#include "lib_hao/flops.h"

using namespace std;

namespace matrix_hao_lib
{

    namespace io_detail
    {
        template<typename T> struct matrix_fmt_info
        {
        };

        template<> struct matrix_fmt_info<float>
        {
            static const char dtype_code ='S';
        };
        template<> struct matrix_fmt_info<double>
        {
            static const char dtype_code ='D';
        };
        template<> struct matrix_fmt_info<complex<float>>
        {
            static const char dtype_code ='C';
        };
        template<> struct matrix_fmt_info<complex<double>>
        {
            static const char dtype_code ='Z';
        };
    } // end of namespace io_detail

    template<typename T> Matrix<T,2> read_matrix(ifstream &file) 
    {
        int M, N, i, j;
        char dtype;
        string info;

        if (!file) {
            throw std::invalid_argument("File cannot be opened.\n");
        }

        getline(file, info);

        //cout << info << endl;

        file >> dtype >> M >> N;

        //cout << dtype << " " << M << " " << N << endl;
        Matrix<T,2> A(M,N);

        assert( dtype == io_detail::matrix_fmt_info<T>::dtype_code );

        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                file >> A(i,j);
            }
        }

        return A;
        
    }
    
    template<typename T> Matrix<T,2> read_matrix(const char *filename)
    {
        ifstream file;
        // prepare file to throw if failbit gets set
        std::ios_base::iostate exceptionMask = file.exceptions() | std::ios::failbit;
        file.exceptions(exceptionMask);

        try {
            file.open(filename);
        }
        catch (std::ios_base::failure &e) {
            std::cerr << "Error opening file " << filename << ": " << e.what() << '\n';
            throw;
        }
        Matrix<T,2> A = read_matrix<T>(file);

        return A;
    }


    BL_INT lapack_ran_ISEED[4] = { 0, 127, 0, 127 };

    void fill_random(Matrix<float,2> &A)
    {
        BL_INT itwo = 2;
        BL_INT size_A = A.L1 * A.L2;
        FORTRAN_NAME(slarnv)(&itwo, lapack_ran_ISEED, &size_A, A.base_array);
    }

    void fill_random(Matrix<double,2> &A)
    {
        BL_INT itwo = 2;
        BL_INT size_A = A.L1 * A.L2;
        FORTRAN_NAME(dlarnv)(&itwo, lapack_ran_ISEED, &size_A, A.base_array);
    }

    void fill_random(Matrix<complex<float>,2> &A)
    {
        BL_INT itwo = 2;
        BL_INT size_A = A.L1 * A.L2;
        FORTRAN_NAME(clarnv)(&itwo, lapack_ran_ISEED, &size_A, reinterpret_cast<ccomplex*>(A.base_array));
    }
 
    void fill_random(Matrix<complex<double>,2> &A)
    {
        BL_INT itwo = 2;
        BL_INT size_A = A.L1 * A.L2;
        FORTRAN_NAME(zlarnv)(&itwo, lapack_ran_ISEED, &size_A, reinterpret_cast<doubleccomplex*>(A.base_array));
    }

void size_dgemm_magma_double_test()
 {
     //cout << "new_dgemm_magma_test" << endl;
     real_Double_t cpu_time, gpu_time;
     bool has_exact;
#if 0
     Matrix<double,2> a=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-567392-A_Matrix-double-1000x20000.txt");
     Matrix<double,2> b=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-567392-B_Matrix-double-20000x1000.txt");
     Matrix<double,2> c(1000,1000);
     Matrix<double,2> c_exact=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-567392-C_Matrix-double-1000x1000.txt");
     has_exact = true;
#endif
#if 0
     Matrix<double,2> a=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-215900-A_Matrix-double-3136x3136.txt");
     Matrix<double,2> b=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-215900-B_Matrix-double-3136x3136.txt");
     Matrix<double,2> c(3136, 3136);
     Matrix<double,2> c_exact=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-215900-C_Matrix-double-3136x3136.txt");
     has_exact = true;
#endif
#if 0
     Matrix<double,2> a(3136, 3136); fill_random(a);
     Matrix<double,2> b(3136, 3136); fill_random(b);
     Matrix<double,2> c(3136, 3136);
     Matrix<double,2> c_exact(3136, 3136);
     has_exact = false;
#endif
#if 1
     Matrix<double,2> a(10304, 10304); fill_random(a);
     Matrix<double,2> b(10304, 10304); fill_random(b);
     Matrix<double,2> c(10304, 10304);
     Matrix<double,2> c_exact(10304, 10304);
     has_exact = false;
#endif
#if 0
     Matrix<double,2> a=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-379403-A_Matrix-double-5184x5184.txt");
     Matrix<double,2> b=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-379403-B_Matrix-double-5184x5184.txt");
     Matrix<double,2> c(5184, 5184);
     Matrix<double,2> c_exact=read_matrix<double>("/particle/disk2/yvirgus/python-scratch/matrix-mul/mul-379403-C_Matrix-double-5184x5184.txt");
     has_exact = true;
#endif

     f77lapack_traits<BL_INT> xlapack_f77;
     linalg<BL_INT> LA_f77(&xlapack_f77);

     cout << "starting computation..." << endl;
     cout.flush();

     cpu_time = magma_wtime();
     LA_f77.gmm(a, b, c, 'N', 'N');
     cpu_time = magma_wtime() - cpu_time;
     cout << "cpu time: " << cpu_time << endl;
     cout.flush();
     if (!has_exact) c_exact = c;

     magma_traits<magma_int_t> xlapack;
     linalg<magma_int_t> LA(&xlapack);

     gpu_time = magma_wtime();
     LA.gmm(a, b, c, 'N', 'N');
     gpu_time = magma_wtime() - gpu_time;
     cout << "gpu time: " << gpu_time << endl;
     cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     cout.flush();

     // SECOND TEST
     cpu_time = magma_wtime();
     LA_f77.gmm(a, b, c, 'N', 'N');
     cpu_time = magma_wtime() - cpu_time;
     cout << "cpu time: " << cpu_time << endl;
     cout.flush();

     gpu_time = magma_wtime();
     LA.gmm(a, b, c, 'N', 'N');
     gpu_time = magma_wtime() - gpu_time;
     cout << "gpu time: " << gpu_time << endl;
     cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     cout.flush();
     // END SECOND TEST

     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if (flag==0) 
         cout<<"New gmm_magma passed double test! \n";
     else
         cout << "WARNING!!!!!!!!! New gmm_magma failed double test! " << flag << " values mismatched" << endl;
 }

struct gemm_sizes {
    int M, N, K;
};

void sgemm_float_matrix(int M, int N, int K)
 {
     real_Double_t gflops, magma_perf, cpu_perf, cpu_time, magma_time;
     bool has_exact;     

     gflops = FLOPS_SGEMM( M, N, K ) / 1e9;

     Matrix<float,2> a(M, K); fill_random(a);
     Matrix<float,2> b(K, N); fill_random(b);
     Matrix<float,2> c(M, N);
     Matrix<float,2> c_exact(M, N);
     has_exact = false;


     f77lapack_traits<BL_INT> xlapack_f77;
     linalg<BL_INT> LA_f77(&xlapack_f77);

     //cout << "starting computation..." << endl;
     cout.flush();

     //------ Performing dgemm with lapack 
     cpu_time = magma_wtime();
     LA_f77.gmm(a, b, c, 'N', 'N');
     cpu_time = magma_wtime() - cpu_time;
     cpu_perf = gflops / cpu_time;
     //cout << "cpu time: " << cpu_time << endl;
     //cout.flush();
     if (!has_exact) c_exact = c;

     magma_traits<magma_int_t> xlapack;
     linalg<magma_int_t> LA(&xlapack);

     //------ Performing dgemm with magma
  
     LA.gmm(a, b, c, 'N', 'N');
     magma_time = xlapack.tm_blas;
     magma_perf = gflops / magma_time;
     //cout << "gpu time: " << gpu_time << endl;
     //cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     //cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     //cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     //cout.flush();

     //cout << "    M     N     K    MAGMA Gflop/s (ms)  in (ms)  out (ms)     CPU Gflop/s (ms) " << endl;
     //cout << "=======================================================================================\n";
     //cout << "  " << M << "  " << N << "  " << K << "   " << magma_perf << " (" << magma_time*1000 << ")    " << xlapack.tm_transfer_in*1000 << "   " <<  xlapack.tm_transfer_out*1000 << "   " << cpu_perf << " (" << cpu_time*1000 << ")   ";

     
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-4) flag++;}
     }
     //if (flag==0) 
       //cout<<"New gmm_magma passed double complex test! \n";
       //cout << "ok" << endl;
       //else
       //cout << "WARNING!!!!!!!!! New gmm_magma failed double complex test! " << flag << " values mismatched" << endl;
       //cout << "failed" << endl; 

     printf("%5d %5d %5d   %7.2f (%7.2f) %7.2f %7.2f  %7.2f (%7.2f)    %s \n",  
            M, N, K, magma_perf, magma_time*1000, xlapack.tm_transfer_in*1000, 
            xlapack.tm_transfer_out*1000, cpu_perf, cpu_time*1000, (flag == 0 ? "ok" : "failed")); 
  }

void sgemm_float_matrix_size_test()
  {
    //int M, N, K;
    //gemm_sizes mtx_sizes{10,5,8};
    //mtx_sizes  = {1, 3, 7};
    //cout << A.M << A.N << A.K << endl;
    
    //vector<int> sizes;
    vector<gemm_sizes> mtx_sizes;


    // With struct
    for (int i = 8; i <= 1087; i *= 2){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }
    //    for (int i = 256; i <= 2560; i += 256){
    //for (int i = 1088; i <= 10304; i += 1024){
    for (int i = 1088; i <= 5184; i += 1024){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }
    cout << "\nTesting sgemm (Matrix Multiplication) :\n";
    cout << "    M     N     K   MAGMA Gflop/s (ms)    in     out     CPU Gflop/s (ms)  result" << endl;
    cout << "                                         (ms)    (ms)" << endl;
    cout << "=================================================================================\n";

    //    for (vector<int>::iterator it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
    for (auto it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
      //gemm_sizes &A = *it; 
      sgemm_float_matrix(it->M, it->N, it->K);
    }
  }

void dgemm_double_matrix(int M, int N, int K)
 {
     real_Double_t gflops, magma_perf, cpu_perf, cpu_time, magma_time;
     bool has_exact;
     
     gflops = FLOPS_DGEMM( M, N, K ) / 1e9;
     //gflops = 2.0*M*N*K / 1e9;

     Matrix<double,2> a(M, K); fill_random(a);
     Matrix<double,2> b(K, N); fill_random(b);
     Matrix<double,2> c(M, N);
     Matrix<double,2> c_exact(M, N);
     has_exact = false;


     f77lapack_traits<BL_INT> xlapack_f77;
     linalg<BL_INT> LA_f77(&xlapack_f77);

     //cout << "starting computation..." << endl;
     cout.flush();

     //------ Performing dgemm with lapack 
     cpu_time = magma_wtime();
     LA_f77.gmm(a, b, c, 'N', 'N');
     cpu_time = magma_wtime() - cpu_time;
     cpu_perf = gflops / cpu_time;
     //cout << "cpu time: " << cpu_time << endl;
     //cout.flush();
     if (!has_exact) c_exact = c;

     magma_traits<magma_int_t> xlapack;
     linalg<magma_int_t> LA(&xlapack);

     //------ Performing dgemm with magma
  
     LA.gmm(a, b, c, 'N', 'N');
     magma_time = xlapack.tm_blas;
     magma_perf = gflops / magma_time;
     //cout << "gpu time: " << gpu_time << endl;
     //cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     //cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     //cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     //cout.flush();

     //cout << "    M     N     K    MAGMA Gflop/s (ms)  in (ms)  out (ms)     CPU Gflop/s (ms) " << endl;
     //cout << "=======================================================================================\n";
     //cout << "  " << M << "  " << N << "  " << K << "   " << magma_perf << " (" << magma_time*1000 << ")    " << xlapack.tm_transfer_in*1000 << "   " <<  xlapack.tm_transfer_out*1000 << "   " << cpu_perf << " (" << cpu_time*1000 << ")   ";

     
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-11) flag++;}
     }
     //if (flag==0) 
       //cout<<"New gmm_magma passed double test! \n";
       //cout << "ok" << endl;
       //else
       //cout << "WARNING!!!!!!!!! New gmm_magma failed double test! " << flag << " values mismatched" << endl;
       //cout << "failed" << endl; 

     printf("%5d %5d %5d   %7.2f (%7.2f) %7.2f %7.2f  %7.2f (%7.2f)    %s \n",  
            M, N, K, magma_perf, magma_time*1000, xlapack.tm_transfer_in*1000, 
            xlapack.tm_transfer_out*1000, cpu_perf, cpu_time*1000, (flag == 0 ? "ok" : "failed")); 
  }

void dgemm_double_matrix_size_test()
  {
    //int M, N, K;
    //gemm_sizes mtx_sizes{10,5,8};
    //mtx_sizes  = {1, 3, 7};
    //cout << A.M << A.N << A.K << endl;
    
    vector<int> sizes;
    vector<gemm_sizes> mtx_sizes;

    // First approach without struct 
    /*
    for (int i = 1088; i <= 10304; i += 1024){
      sizes.push_back(i);
    }
    cout << "    M     N     K    MAGMA Gflop/s (ms)  in (ms)  out (ms)  CPU Gflop/s (ms)  result" << endl;
    cout << "=======================================================================================\n";

    //    for (vector<int>::iterator it = sizes.begin() ; it != sizes.end(); ++it){
    for (auto it = sizes.begin() ; it != sizes.end(); ++it){
      M = *it;
      N = *it;
      K = *it;
      dgemm_double_matrix(M, N, K);
    }
    */

    // With struct
    for (int i = 8; i <= 1087; i *= 2){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }
    //    for (int i = 256; i <= 2560; i += 256){
    //for (int i = 1088; i <= 10304; i += 1024){
    for (int i = 1088; i <= 5184; i += 1024){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }

    cout << "\nTesting dgemm (Matrix Multiplication) :\n";
    cout << "    M     N     K   MAGMA Gflop/s (ms)    in     out     CPU Gflop/s (ms)  result" << endl;
    cout << "                                         (ms)    (ms)" << endl;
    cout << "=================================================================================\n";

    //    for (vector<int>::iterator it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
    for (auto it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
      //gemm_sizes &A = *it; 
      dgemm_double_matrix(it->M, it->N, it->K);
    }
  }

void cgemm_float_complex_matrix(int M, int N, int K)
 {
     real_Double_t gflops, magma_perf, cpu_perf, cpu_time, magma_time;
     bool has_exact;     

     gflops = FLOPS_CGEMM( M, N, K ) / 1e9;

     Matrix<complex<float>,2> a(M, K); fill_random(a);
     Matrix<complex<float>,2> b(K, N); fill_random(b);
     Matrix<complex<float>,2> c(M, N);
     Matrix<complex<float>,2> c_exact(M, N);
     has_exact = false;


     f77lapack_traits<BL_INT> xlapack_f77;
     linalg<BL_INT> LA_f77(&xlapack_f77);

     //cout << "starting computation..." << endl;
     cout.flush();

     //------ Performing dgemm with lapack 
     cpu_time = magma_wtime();
     LA_f77.gmm(a, b, c, 'N', 'N');
     cpu_time = magma_wtime() - cpu_time;
     cpu_perf = gflops / cpu_time;
     //cout << "cpu time: " << cpu_time << endl;
     //cout.flush();
     if (!has_exact) c_exact = c;

     magma_traits<magma_int_t> xlapack;
     linalg<magma_int_t> LA(&xlapack);

     //------ Performing dgemm with magma
  
     LA.gmm(a, b, c, 'N', 'N');
     magma_time = xlapack.tm_blas;
     magma_perf = gflops / magma_time;
     //cout << "gpu time: " << gpu_time << endl;
     //cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     //cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     //cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     //cout.flush();

     //cout << "    M     N     K    MAGMA Gflop/s (ms)  in (ms)  out (ms)     CPU Gflop/s (ms) " << endl;
     //cout << "=======================================================================================\n";
     //cout << "  " << M << "  " << N << "  " << K << "   " << magma_perf << " (" << magma_time*1000 << ")    " << xlapack.tm_transfer_in*1000 << "   " <<  xlapack.tm_transfer_out*1000 << "   " << cpu_perf << " (" << cpu_time*1000 << ")   ";

     
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-4) flag++;}
     }
     //if (flag==0) 
       //cout<<"New gmm_magma passed double complex test! \n";
       //cout << "ok" << endl;
       //else
       //cout << "WARNING!!!!!!!!! New gmm_magma failed double complex test! " << flag << " values mismatched" << endl;
       //cout << "failed" << endl; 

     printf("%5d %5d %5d   %7.2f (%7.2f) %7.2f %7.2f  %7.2f (%7.2f)    %s \n",  
            M, N, K, magma_perf, magma_time*1000, xlapack.tm_transfer_in*1000, 
            xlapack.tm_transfer_out*1000, cpu_perf, cpu_time*1000, (flag == 0 ? "ok" : "failed")); 
  }

void cgemm_float_complex_matrix_size_test()
  {
    //int M, N, K;
    //gemm_sizes mtx_sizes{10,5,8};
    //mtx_sizes  = {1, 3, 7};
    //cout << A.M << A.N << A.K << endl;
    
    //vector<int> sizes;
    vector<gemm_sizes> mtx_sizes;


    // With struct
    for (int i = 8; i <= 1087; i *= 2){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }
    //    for (int i = 256; i <= 2560; i += 256){
    //for (int i = 1088; i <= 10304; i += 1024){
    for (int i = 1088; i <= 5184; i += 1024){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }
    cout << "\nTesting cgemm (Matrix Multiplication) :\n";
    cout << "    M     N     K   MAGMA Gflop/s (ms)    in     out     CPU Gflop/s (ms)  result" << endl;
    cout << "                                         (ms)    (ms)" << endl;
    cout << "=================================================================================\n";

    //    for (vector<int>::iterator it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
    for (auto it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
      //gemm_sizes &A = *it; 
      cgemm_float_complex_matrix(it->M, it->N, it->K);
    }
  }

void zgemm_double_complex_matrix(int M, int N, int K)
 {
     real_Double_t gflops, magma_perf, cpu_perf, cpu_time, magma_time;
     bool has_exact;     

     gflops = FLOPS_ZGEMM( M, N, K ) / 1e9;

     Matrix<complex<double>,2> a(M, K); fill_random(a);
     Matrix<complex<double>,2> b(K, N); fill_random(b);
     Matrix<complex<double>,2> c(M, N);
     Matrix<complex<double>,2> c_exact(M, N);
     has_exact = false;


     f77lapack_traits<BL_INT> xlapack_f77;
     linalg<BL_INT> LA_f77(&xlapack_f77);

     //cout << "starting computation..." << endl;
     cout.flush();

     //------ Performing dgemm with lapack 
     cpu_time = magma_wtime();
     LA_f77.gmm(a, b, c, 'N', 'N');
     cpu_time = magma_wtime() - cpu_time;
     cpu_perf = gflops / cpu_time;
     //cout << "cpu time: " << cpu_time << endl;
     //cout.flush();
     if (!has_exact) c_exact = c;

     magma_traits<magma_int_t> xlapack;
     linalg<magma_int_t> LA(&xlapack);

     //------ Performing dgemm with magma
  
     LA.gmm(a, b, c, 'N', 'N');
     magma_time = xlapack.tm_blas;
     magma_perf = gflops / magma_time;
     //cout << "gpu time: " << gpu_time << endl;
     //cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     //cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     //cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     //cout.flush();

     //cout << "    M     N     K    MAGMA Gflop/s (ms)  in (ms)  out (ms)     CPU Gflop/s (ms) " << endl;
     //cout << "=======================================================================================\n";
     //cout << "  " << M << "  " << N << "  " << K << "   " << magma_perf << " (" << magma_time*1000 << ")    " << xlapack.tm_transfer_in*1000 << "   " <<  xlapack.tm_transfer_out*1000 << "   " << cpu_perf << " (" << cpu_time*1000 << ")   ";

     
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-11) flag++;}
     }
     //if (flag==0) 
       //cout<<"New gmm_magma passed double complex test! \n";
       //cout << "ok" << endl;
       //else
       //cout << "WARNING!!!!!!!!! New gmm_magma failed double complex test! " << flag << " values mismatched" << endl;
       //cout << "failed" << endl; 

     printf("%5d %5d %5d   %7.2f (%7.2f) %7.2f %7.2f  %7.2f (%7.2f)    %s \n",  M, N, K, magma_perf, magma_time*1000, xlapack.tm_transfer_in*1000, xlapack.tm_transfer_out*1000, cpu_perf, cpu_time*1000, (flag == 0 ? "ok" : "failed")); 
  }

void zgemm_double_complex_matrix_size_test()
  {
    //int M, N, K;
    //gemm_sizes mtx_sizes{10,5,8};
    //mtx_sizes  = {1, 3, 7};
    //cout << A.M << A.N << A.K << endl;
    
    //vector<int> sizes;
    vector<gemm_sizes> mtx_sizes;


    // With struct
    for (int i = 8; i <= 1087; i *= 2){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }
    //    for (int i = 256; i <= 2560; i += 256){
    //for (int i = 1088; i <= 10304; i += 1024){
    for (int i = 1088; i <= 5184; i += 1024){
      gemm_sizes A = {i,i,i};
      mtx_sizes.push_back(A);
    }

    cout << "\nTesting zgemm (Matrix Multiplication) :\n";
    cout << "    M     N     K   MAGMA Gflop/s (ms)    in     out     CPU Gflop/s (ms)  result" << endl;
    cout << "                                         (ms)    (ms)" << endl;
    cout << "=================================================================================\n";

    //    for (vector<int>::iterator it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
    for (auto it = mtx_sizes.begin() ; it != mtx_sizes.end(); ++it){
      //gemm_sizes &A = *it; 
      zgemm_double_complex_matrix(it->M, it->N, it->K);
    }
  }


    void eigen_double_complex(int M=1088, char jobz='V', char uplo='L')
    {
      
      real_Double_t cpu_time, magma_time;
      bool has_exact;

      complex<double> *a_alt_ptr;
      Matrix<complex<double>,2> a(M, M); fill_random(a);
      Matrix<complex<double>,2> I(M, M);
      
      for (int i = 0; i < M; i++) {
	for (int j = 0; j < M; j++) {
	  if (i==j){
	    I(i,j) = {1.000000000000000, 0.000000000000000};
	  }
	  else{
	    I(i,j) = {0.000000000000000, 0.000000000000000};
	  }
	}
      }

      Matrix<complex<double>,2> c(M, M), a_exact(M,M);
      
      Matrix<double,1> w(M), w_lapack(M), w_exact(M);
      has_exact = false;

      f77lapack_traits<BL_INT> xlapack_f77;
      linalg<BL_INT> LA_f77(&xlapack_f77);

      // Hermitize the matrix:
      LA_f77.gmm(a, I, c, 'C', 'N');
      a = a + c;

      Matrix<complex<double>,2> a_lapack = a;
      check_Hermitian(a);
      //cout << "a = \n" << a << endl;
      //cout << "a_lapack = \n" << a << endl;

      // HACK: reallocate with pinned memory
      magma_int_t alloc_status = magma_zmalloc_pinned(reinterpret_cast<magmaDoubleComplex**>(&a_alt_ptr), M*M);
      assert(alloc_status == MAGMA_SUCCESS);
      copy(a.base_array, a.base_array+a.L_f(), a_alt_ptr);
      a.point(M*M, a_alt_ptr);
      //cout << "Reallocated A array using zmalloc_pinned" << endl;
      // END HACK

      //cout << "M = " << M << endl;
      //cout.flush();

      //------ Performing eigen with lapack 
      cpu_time = magma_wtime();
      //LA_f77.eigen(a_lapack, w_lapack, 'V', 'L');     
      LA_f77.eigen(a_lapack, w_lapack, jobz, uplo);     
      //eigen(a_lapack, w_lapack, 'V', 'L');
      cpu_time = magma_wtime() - cpu_time;
      //  cout << "CPU time (sec) " << cpu_time << endl;
      // cout.flush();

      if (!has_exact)
	{
	  a_exact = a_lapack;
	  w_exact = w_lapack;
	}

      magma_traits<magma_int_t> xlapack;
      linalg<magma_int_t> LA(&xlapack);

      //cout << "M = " << M << endl;
      //cout.flush();
      //------ Performing eigen with magma
      magma_time = magma_wtime();
      //LA.eigen(a, w, 'V', 'L'); 
      LA.eigen(a, w, jobz, uplo);   
      //eigen_magma(a,w, 'V', 'L');
      magma_time = magma_wtime() - magma_time;          
      //      cout << "GPU time (sec) " << magma_time << endl;
      //     cout.flush();

      size_t flag=0;
      for(size_t i=0; i<a.L1; i++)
	{
	  for(size_t j=0; j<a.L2; j++) {if(abs(abs(a(i,j))-abs(a_exact(i,j)))>1e-12) flag++;}
	}

      for(size_t i=0; i<w.L1; i++) {if(abs(w(i)-w_exact(i))>1e-12) flag++;}

      //printf("%5d     %7.3f       %7.3f       %7.3f       %7.3f        %s \n",  
      //          M, magma_time, xlapack.tm_query, xlapack.tm_blas,  cpu_time, (flag == 0 ? "ok" : "failed"));
      printf("%5d     %7.3f          %7.3f        %s \n",  M, magma_time, cpu_time, (flag == 0 ? "ok" : "failed"));
      
      //      cout << "flag = " << flag << endl;
      //      if(flag==0) cout<<"New Eigen passed Hermition test! \n";
      //      else cout<<"WARNING!!!!!!!!! New Eigen failed Hermintion test! \n";
     //cout<<setprecision(16);
     //cout<<w<<endl;
     //cout<<a<<endl;
      magma_free_pinned(a_alt_ptr);
    }

void eigen_double_complex_size_test()
  {
    int M;
    char jobz='V', uplo='L'; 
    vector<int> sizes;

    for (int i = 210; i <= 1000; i += 200){
      sizes.push_back(i);
    }

    for (int i = 1088; i <= 10304; i += 1024){
        sizes.push_back(i);
    }
    //cout << "\nTesting zheevd (Eigen) using: jobz = %s, uplo = %s\n", (jobz == 'V' ? "Vectors needed" : "No vectors"), (uplo == 'L' ? "Lower" : "Upper") ;
    //cout << "    M    GPU time (s) query time (s) zheevd time (s) CPU time (s)  result" << endl;
    //cout << "===========================================================================\n";

    printf("\nTesting zheevd (Eigen) using: jobz = %s, uplo = %s\n", (jobz == 'V' ? "Vectors needed" : "No vectors"), (uplo == 'L' ? "Lower" : "Upper"));
    cout << "    M    GPU time (s)    CPU time (s)   result" << endl;
    cout << "==============================================\n";

    //    for (vector<int>::iterator it = sizes.begin() ; it != sizes.end(); ++it){
    for (auto it = sizes.begin() ; it != sizes.end(); ++it){
      M = *it;
      eigen_double_complex(M, jobz, uplo);
      //break;
    }

  }


  void LU_decompose(int M)
  {
      real_Double_t gflops, cpu_time, magma_time, cpu_perf, gpu_perf;
      bool has_exact;

      gflops = FLOPS_ZGETRF( M, M ) / 1e9;

      Matrix<complex<double>,2> A(M, M); fill_random(A);
      Matrix<complex<double>,2> A_lapack = A;
      has_exact = false;
      Matrix<complex<double>,2> A_exact(M, M);

      cpu_time = magma_wtime();
      f77lapack_traits<BL_INT> xlapack_f77;                         
      LU_decomp<complex<double>,BL_INT> LU_lapack(A_lapack, &xlapack_f77);
      cpu_time = magma_wtime() - cpu_time;
      cpu_perf = gflops / cpu_time;
      //cout << "CPU time (sec) " << cpu_time << endl;
      //cout.flush();

      //cout << A_lapack << endl;
      if (!has_exact)
	{
	  A_exact = LU_lapack.A;
	}
      //cout << A_exact << endl;

      //cout << A << endl;
      magma_time = magma_wtime();
      magma_traits<magma_int_t> xlapack;                         
      LU_decomp<complex<double>,magma_int_t> LU(A, &xlapack);               
      magma_time = magma_wtime() - magma_time;
      gpu_perf = gflops / magma_time;
      //cout << "GPU time (sec) " << magma_time << endl;
      //cout.flush();
                                        
      //cout << "LU.A = " << LU.A << endl;
      size_t flag=0;
      for(size_t i=0; i<A_exact.L1; i++)
	{
	  for(size_t j=0; j<A_exact.L2; j++) {if(abs(LU.A(i,j)-A_exact(i,j))>1e-10) flag++;}
	}
      printf("%5d  %7.2f (%7.2f)    %7.2f (%7.2f)    %s \n", (int) M, gpu_perf, magma_time, cpu_perf, cpu_time, (flag == 0 ? "ok" : "failed"));
      //cout << "flag : " << flag << endl;
      //if(flag==0) cout<<"New LU_decomp passed complex double test! \n";
      //else cout<<"WARNING!!!!!!!!! New LU_decomp failed complex double test! \n";
 }

  void LU_decomp_size_test()
  {
    int M;
    vector<int> sizes;

    for (int i = 16; i <= 1087; i += 128){
      sizes.push_back(i);
    }

    for (int i = 1088; i <= 10304; i += 1024){
    //for (auto i = 1088; i <= 2500; i += 1024){
      sizes.push_back(i);
    }
    cout << "\nTesting zgetrf (LU decomposition) :\n";
    cout << "    M     MAGMA Gflop/s (s)   CPU Gflop/s (s)  result" << endl;
    cout << "========================================================\n";

    //    for (vector<int>::iterator it = sizes.begin() ; it != sizes.end(); ++it){
    for (auto it = sizes.begin() ; it != sizes.end(); ++it){
      M = *it;
      LU_decompose(M);
    }
  }


  void inverse_mtx(int M=3000)
  {
     real_Double_t gflops, magma_perf, cpu_perf, cpu_time, magma_time;
     bool has_exact;     

     gflops = FLOPS_ZGETRI( M ) / 1e9;

     Matrix<complex<double>,2> A(M, M); fill_random(A);
     Matrix<complex<double>,2> A_lapack = A;
     Matrix<complex<double>,2> A_exact(M,M);
     has_exact = false;
     
     f77lapack_traits<BL_INT> xlapack_f77;
     LU_decomp<complex<double>,BL_INT>  LU_lapack( A_lapack, &xlapack_f77 );

     cpu_time = magma_wtime();
     A_lapack = LU_lapack.inverse_in();
     cpu_time = magma_wtime() - cpu_time;
     cpu_perf = gflops / cpu_time;
     //cout << "CPU time : " << cpu_time << endl;
     //cout.flush();

     if (!has_exact){
       A_exact = A_lapack;
     }

     magma_traits<magma_int_t> xlapack;
     LU_decomp<complex<double>,magma_int_t>  LU( A, &xlapack );

     magma_time = magma_wtime();
     A=LU.inverse_in();
     magma_time = magma_wtime() - magma_time;
     magma_perf = gflops / xlapack.tm_blas;
     //cout << "gpu time: " << magma_time << endl;
     //cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     //cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     //cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     //cout.flush();
     
     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(A(i,j)-A_exact(i,j))>1e-12) flag++;}
     }
      printf("%5d  %7.2f (%7.3f)    %7.2f (%7.3f)    %s \n", (int) M, magma_perf, magma_time, cpu_perf, cpu_time, (flag == 0 ? "ok" : "failed"));
      //if(flag==0) cout<<"New Inverse passed complex double test! \n";
      //else cout<<"WARNING!!!!!!!!! New Inverse failed complex double test! \n";
  }

  void inverse_mtx_size_test()
  {
    int M;
    vector<int> sizes;


    for (int i = 8; i <= 1087; i *= 2){
      sizes.push_back(i);
    }
    /*    
    for (int i = 1088; i <= 10304; i += 1024){
        //for (auto i = 1088; i <= 2500; i += 1024){
      sizes.push_back(i);
    }
    */
    cout << "\nTesting zgetri (Inverse Matrix) :\n";
    cout << "   M    MAGMA Gflop/s (s)     CPU Gflop/s (s)  result" << endl;
    cout << "========================================================\n";

    //    for (vector<int>::iterator it = sizes.begin() ; it != sizes.end(); ++it){
    for (auto it = sizes.begin() ; it != sizes.end(); ++it){
      M = *it;
      inverse_mtx(M);
    }
  }


  void linear_eq(int M=3000)
  {
     real_Double_t gflops, magma_perf, cpu_perf, cpu_time, magma_time;
     bool has_exact;     

     gflops = FLOPS_ZGETRS( M, M ) / 1e9;

     Matrix<complex<double>,2> A(M, M); fill_random(A);
     Matrix<complex<double>,2> B(M, M); fill_random(B);
     Matrix<complex<double>,2> A_lapack = A;
     Matrix<complex<double>,2> B_lapack = B;
     Matrix<complex<double>,2> X_exact(M,M);
     has_exact = false;
     
     f77lapack_traits<BL_INT> xlapack_f77;
     LU_decomp<complex<double>,BL_INT>  LU_lapack( A_lapack, &xlapack_f77 );

     cpu_time = magma_wtime();
     Matrix<complex<double>,2> X_lapack=LU_lapack.solve_lineq_in(B_lapack);
     cpu_time = magma_wtime() - cpu_time;
     cpu_perf = gflops / cpu_time;
     //cout << "CPU time : " << cpu_time << endl;
     //cout.flush();

     if (!has_exact){
       X_exact = X_lapack;
     }

     magma_traits<magma_int_t> xlapack;
     LU_decomp<complex<double>,magma_int_t>  LU( A, &xlapack );

     magma_time = magma_wtime();
     Matrix<complex<double>,2> X=LU.solve_lineq_in(B);
     magma_time = magma_wtime() - magma_time;
     magma_perf = gflops / xlapack.tm_blas;
     //cout << "gpu time: " << magma_time << endl;
     //cout << "- inbound data transfer:  " << xlapack.tm_transfer_in << endl;
     //cout << "- outbound data transfer: " << xlapack.tm_transfer_out << endl;
     //cout << "- computation (BLAS):     " << xlapack.tm_blas << endl;
     //cout.flush();
     
     size_t flag=0;
     for(size_t i=0; i<X_exact.L1; i++)
       {
	 for(size_t j=0; j<X_exact.L2; j++) {if(abs(X(i,j)-X_exact(i,j))>1e-10) flag++;}
       }
     //if(flag==0) cout<<"New Solve_lineq passed complex double test! \n";
     //else cout<<"WARNING!!!!!!!!! New Solve_lineq failed complex double test! \n";

     printf("%5d  %5d  %7.2f (%7.3f)    %7.2f (%7.3f)    %s \n", (int) M, M, magma_perf, magma_time, cpu_perf, cpu_time, (flag == 0 ? "ok" : "failed"));

  }

  void linear_eq_size_test()
  {
    int M;
    vector<int> sizes;
    
    for (int i = 8; i <= 1087; i *= 2 ){
      sizes.push_back(i);
    }
    /*
    for (int i = 1088; i <= 10304; i += 1024){
    //for (auto i = 1088; i <= 2500; i += 1024){
      sizes.push_back(i);
    }
    */
    cout << "\nTesting zgetrs (Solving Linear Equation) :\n";
    cout << "   M   NRHS   MAGMA Gflop/s (s)     CPU Gflop/s (s)  result" << endl;
    cout << "========================================================\n";

    //    for (vector<int>::iterator it = sizes.begin() ; it != sizes.end(); ++it){
    for (auto it = sizes.begin() ; it != sizes.end(); ++it){
      M = *it;
      linear_eq(M);
    }
  }


  void QR_decompose(int M)
  {
      real_Double_t cpu_time, magma_time;
      bool has_exact;

      Matrix<complex<double>,2> A(M, M); fill_random(A);
      Matrix<complex<double>,2> A_lapack = A;
      has_exact = false;
      Matrix<complex<double>,2> A_exact(M, M);
      double det_exact;

      cpu_time = magma_wtime();
      f77lapack_traits<BL_INT> xlapack_f77;                         
      linalg<BL_INT> LA_lapack(&xlapack_f77);

      double det_lapack=LA_lapack.QRMatrix(A_lapack);
      cpu_time = magma_wtime() - cpu_time;

      if (!has_exact){
          A_exact = A_lapack;
          det_exact = det_lapack;
      }

      magma_time = magma_wtime();
      magma_traits<magma_int_t> xlapack;                         
      linalg<magma_int_t> LA(&xlapack);
      
      double det = LA.QRMatrix(A);
      magma_time = magma_wtime() - magma_time;

     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(abs(A(i,j))-abs(A_exact(i,j)))>1e-12) flag++;}
     }
     if(abs(det-det_exact)>1e-12) flag++;
     //if(flag==0) cout<<"New QRMatrix magma passed complex double test! \n";
     //else cout<<"WARNING!!!!!!!!! New QRMatrix magma failed complex double test! \n";

      printf("%5d %5d %5d  %7.5f    %7.5f    %s \n", (int) M, M, M, magma_time, cpu_time, (flag == 0 ? "ok" : "failed"));

 }

  void QR_decomp_size_test()
  {
    int M;
    vector<int> sizes;

    for (int i = 8; i <= 1087; i *= 2){
      sizes.push_back(i);
    }

    //for (int i = 1088; i <= 10304; i += 1024){
    for (auto i = 1088; i <= 5184; i += 1024){
      sizes.push_back(i);
    }
    cout << "\nTesting zgeqrf and zungqr (QR decomposition) :\n";
    cout << "    M    N    K      MAGMA (s)      CPU (s)    result" << endl;
    cout << "=====================================================\n";

    //    for (vector<int>::iterator it = sizes.begin() ; it != sizes.end(); ++it){
    for (auto it = sizes.begin() ; it != sizes.end(); ++it){
      M = *it;
      QR_decompose(M);
    }
  }

 void size_inverse_test()
 {
     real_Double_t cpu_time, gpu_time;
     Matrix<complex<double>, 2> A = read_matrix<complex<double>>("/particle/disk2/yvirgus/python-scratch/Matrix-double-complex-5000x5000.txt");
     //cout << A << endl;
     //Matrix<complex<double>, 2> A_exact = read_matrix<complex<double>>("/particle/disk2/yvirgus/python-scratch/Matrix-double-complex-1000x1000-inverse.txt");  
     //cout << A_exact << endl;
     
     cpu_time = magma_wtime();
     f77lapack_traits<BL_INT> xlapack_f77;
     LU_decomp<complex<double>,BL_INT>  LU_f77( A, &xlapack_f77 );
     cpu_time = magma_wtime() - cpu_time;
     cout << "cpu time: " << cpu_time << endl;
     
     cpu_time = magma_wtime();
     A=LU_f77.inverse_in();
     cpu_time = magma_wtime() - cpu_time;
     cout << "cpu time: " << cpu_time << endl;
     
     //cout << A[0][0].real() << endl;
     
     Matrix<complex<double>, 2> B = read_matrix<complex<double>>("/particle/disk2/yvirgus/python-scratch/Matrix-double-complex-5000x5000.txt");

     magma_traits<magma_int_t> xlapack_magma;
     gpu_time = magma_wtime();
     LU_decomp<complex<double>,magma_int_t>  LU_magma( B, &xlapack_magma );
     gpu_time = magma_wtime() - gpu_time;
     cout <<"gpu time: " << gpu_time << endl;

     gpu_time = magma_wtime();
     B=LU_magma.inverse_in();
     gpu_time = magma_wtime() - gpu_time;
     cout <<"gpu time: " << gpu_time << endl;
     
     /*
     //cout << A << endl;
     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }
     cout << flag << endl;
     if(flag==0) cout<<"Size Inverse passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Size Inverse failed complex double test! \n";
     */
 }

 void matrix_size_linalg_test()
 {
     //size_dgemm_magma_double_test();
     //size_inverse_test();

     //sgemm_float_matrix_size_test();
     //dgemm_double_matrix_size_test();
     //cgemm_float_complex_matrix_size_test();
     // zgemm_double_complex_matrix_size_test();
     //eigen_double_complex_size_test();    
   //LU_decomp_size_test();
   //inverse_mtx_size_test();
   //linear_eq_size_test();
   //QR_decomp_size_test();
   
 }

} //end namespace matrix_hao_lib
