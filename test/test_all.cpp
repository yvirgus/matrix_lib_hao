#include "matrix_all.h"
#include "magma.h"

namespace matrix_hao_lib
{
 void matrix_class_test();
 void matrix_2d_blas_lapack_test();
}

using namespace std;
using namespace matrix_hao_lib;

int main(int argc, char** argv)
{
    int rank=0;

    magma_init();

#ifdef MPI_HAO
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        cout<<"\n\n\n=======Testing======="<<endl;
        matrix_class_test(); 
        matrix_2d_blas_lapack_test();
    }

#ifdef MPI_HAO
    matrix_mpi_test();
#endif


#ifdef MPI_HAO
    MPI_Finalize();
#endif

    return 0;
}
