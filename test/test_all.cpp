#include "matrix_all.h"

#ifdef USE_MAGMA
#include "magma.h"
//#include "magma_lapack.h"
#endif

namespace matrix_hao_lib
{
 void matrix_class_test();
// void matrix_2d_blas_lapack_test();
 void matrix_linalg_test();

 #ifdef USE_MAGMA
 void matrix_size_linalg_test();
    //void zheevd_test();
 #endif

}

using namespace std;
using namespace matrix_hao_lib;

int main(int argc, char** argv)
{
    int rank=0;
    
#ifdef USE_MAGMA
    magma_init();
#endif 

#ifdef MPI_HAO
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        cout<<"\n\n\n=======Testing======="<<endl;
        matrix_class_test(); 
        //matrix_2d_blas_lapack_test();
        matrix_linalg_test();

#ifdef USE_MAGMA
        matrix_size_linalg_test();
        //zheevd_test();
#endif

    }

#ifdef USE_MAGMA
    magma_finalize();
#endif 

#ifdef MPI_HAO
    matrix_mpi_test();
#endif


#ifdef MPI_HAO
    MPI_Finalize();
#endif

    return 0;
}
