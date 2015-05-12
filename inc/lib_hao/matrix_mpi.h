#ifndef MATRIX_HAO_MPI
#define MATRIX_HAO_MPI

#ifdef MPI_HAO


#include <mpi.h>
#include <complex>
namespace matrix_hao_lib
{

 void matrix_mpi_test();

 void MPIBcast(Matrix<double,1>               & buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
 void MPIBcast(Matrix<double,2>               & buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
 void MPIBcast(Matrix<double,3>               & buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
 void MPIBcast(Matrix<std::complex<double>,1> & buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
 void MPIBcast(Matrix<std::complex<double>,2> & buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);
 void MPIBcast(Matrix<std::complex<double>,3> & buffer, int root=0,  const MPI_Comm& comm=MPI_COMM_WORLD);

}




#endif

#endif
