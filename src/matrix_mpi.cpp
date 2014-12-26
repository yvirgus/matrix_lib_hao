#ifdef MPI_HAO
#include <cmath>
#include "matrix_all.h"
using std::complex;
using std::conj;
using std::cout;
using std::endl;

namespace matrix_hao_lib
{
  //buffer.L,buffer.own, buffer.type, buffer.Li do not need to be changed

 void MPIBcast(Matrix<double,1>& buffer, int root,  const MPI_Comm& comm)
 {
  MPI_Bcast(buffer.base_array, buffer.L_f(), MPI_DOUBLE, root, comm);
 }
 
 void MPIBcast(Matrix<double,2>& buffer, int root,  const MPI_Comm& comm)
 {
  MPI_Bcast(buffer.base_array, buffer.L_f(), MPI_DOUBLE, root, comm);
 }

 void MPIBcast(Matrix<double,3>& buffer, int root,  const MPI_Comm& comm)
 {
  MPI_Bcast(buffer.base_array, buffer.L_f(), MPI_DOUBLE, root, comm);
 }

 void MPIBcast(Matrix<complex<double>,1>& buffer, int root,  const MPI_Comm& comm)
 {
  MPI_Bcast(buffer.base_array, buffer.L_f(), MPI_DOUBLE_COMPLEX, root, comm);
 }

 void MPIBcast(Matrix<complex<double>,2>& buffer, int root,  const MPI_Comm& comm)
 {
  MPI_Bcast(buffer.base_array, buffer.L_f(), MPI_DOUBLE_COMPLEX, root, comm);
 }

 void MPIBcast(Matrix<complex<double>,3>& buffer, int root,  const MPI_Comm& comm)
 {
  MPI_Bcast(buffer.base_array, buffer.L_f(), MPI_DOUBLE_COMPLEX, root, comm);
 }
}

#endif
