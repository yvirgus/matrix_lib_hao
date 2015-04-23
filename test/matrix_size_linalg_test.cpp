#include "matrix_all.h"
#include <cstdio>
#include <fstream>
#include <cassert>

#include "lib_hao/matrix_linalg.h"
#include "lib_hao/f77lapack_traits.h"
#include "lib_hao/magma_traits.h"

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

    void fill_random(Matrix<double,2> &A)
    {
        BL_INT itwo = 2;
        BL_INT size_A = A.L1 * A.L2;
        FORTRAN_NAME(dlarnv)(&itwo, lapack_ran_ISEED, &size_A, A.base_array);
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
     size_dgemm_magma_double_test();
     //size_inverse_test();
 }

} //end namespace matrix_hao_lib
