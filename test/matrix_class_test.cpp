#include "matrix_all.h"

using namespace std;

namespace matrix_hao_lib
{
 void matrix_1d_c_slicing()
 {
  Matrix<double,1> A={4,{14.861,12.630129,20.984,23.753129}};

  size_t flag=0;
  for(size_t i=0; i<A.L1; i++) 
  {
   if(abs(A[i]-A(i))>1e-12) flag++;
  }

  A[1]=2.0;
  if(abs(A(1)-2.0)>1e-12) flag++;

  if(flag==0) cout<<"Matrix_1d slicing passed double test! \n";
  else cout<<"WARNING!!!!!!!!! Matrix_1d slicing failed double test! \n";
 }

 void matrix_2d_c_slicing()
 {
  Matrix<double,2> A={2,3,{14.861,12.630129,20.984,23.753129,1.0,2.0}};
  Matrix<double,1> B=A[1];
  Matrix<double,1> C={2,{3,1}};
  size_t flag=0;
  //Reminder: Asigment will not work for creating a link matrix! Only construction works!
  //1. If we use Matrix<double,1> B;B=A[1]; B will be a new matrix instead of link matrix!!!
  //2. If we use Matrix<double,1> B(2);B=A[1]; B will be a new matrix instead of link matrix!!!
  //Only way to get the link matrix is use Matrix<double,1> B=A[1] !!!!!


  if(B.owns) cout<<"WARNING!!!!!!!!! Matrix_2d slicing failed double test! B.owns:"<<B.owns<<"\n";
  for(size_t i=0; i<B.L1; i++)
  {
   if(abs(B[i]-A(i,1))>1e-12) flag++;
  }

  A[1]=C;

  for(size_t i=0; i<B.L1; i++)
  {
   if(abs(B[i]-C(i))>1e-12) flag++;
  }
  if(flag==0) cout<<"Matrix_2d slicing passed double test! \n";
  else cout<<"WARNING!!!!!!!!! Matrix_2d slicing failed double test! \n";

  //cout<<A;
 }


 void matrix_3d_c_slicing()
 {
  Matrix<complex<double>,3> A={2,3,2,{
                             complex<double>(1.0,2.3),complex<double>(3.0,4.0),complex<double>(2.123,3.11),
                             complex<double>(2.0,3.3),complex<double>(4.0,5.0),complex<double>(3.123,4.11),
                             complex<double>(3.0,2.3),complex<double>(3.0,4.0),complex<double>(2.123,3.11),
                             complex<double>(2.0,3.3),complex<double>(4.0,5.0),complex<double>(3.123,6.11)
                                 }};
  Matrix<complex<double>,2> B=A[1];
  Matrix<complex<double>,2> C={2,3,{complex<double>(0.0,0.0),complex<double>(0.0,0.0),complex<double>(0.123,0.11),
                           complex<double>(0.0,0.0),complex<double>(0.0,0.0),complex<double>(0.123,0.11)
                          }};

  size_t flag=0;
  if(B.owns) cout<<"WARNING!!!!!!!!! Matrix_3d slicing failed complex double test! B.owns:"<<B.owns<<"\n";
  for(size_t i=0; i<B.L1; i++)
  {
   for(size_t j=0; j<B.L2; j++)
   {
    if(abs(B(i,j)-A(i,j,1))>1e-12) flag++;
   }
  }

  A[1]=C;

  for(size_t i=0; i<B.L1; i++)
  {
   for(size_t j=0; j<B.L2; j++)
   {
    if(abs(B(i,j)-C(i,j))>1e-12) flag++;
   }
  }
  if(flag==0) cout<<"Matrix_3d slicing passed complex double test! \n";
  else cout<<"WARNING!!!!!!!!! Matrix_3d slicing failed complex double test! \n";

  //cout<<B;
 }

 void matrix_conj_test()
 {
  Matrix_base<complex<double>> A={{1.0,2.3},{3.0,4.0},{2.123,3.11}};
  Matrix_base<complex<double>> B=A.conj();
  Matrix<complex<double>,2>    C={2,3,{ {0.0,1.0},{5.0,2.0},{0.123,0.31},
                                        {0.0,2.0},{3.0,4.0},{0.123,0.21} }};
  Matrix<complex<double>,2>    D=conj(C);

  size_t flag=0;
  for(size_t i=0; i<A.L_f(); i++)
  {
   if(abs(A.base_array[i]-conj(B.base_array[i]))>1e-12) flag++;
  }
  for(size_t i=0; i<C.L1; i++)
  {
   for(size_t j=0; j<C.L2; j++)
   {
    if(abs(C(i,j)-conj(D(i,j)))>1e-12) flag++;
   }
  }

  if(flag==0) cout<<"Matrix conj passed complex double test! \n";
  else cout<<"WARNING!!!!!!!!! Matrix conj failed complex double test! \n";

  //Matrix<complex<double>,3>    TMP={2,2,2,{ {0.0,1.0},{5.0,2.0},{0.123,0.31},
  //                                          {0.0,2.0},{3.0,4.0},{0.123,0.21},{5.0,2.0},{0.123,0.31} }};
  //cout<<TMP<<endl;
  //cout<<conj(TMP)<<endl;
  //cout<<flag<<endl;
 }

 void matrix_2d_trans_conjtrans_test()
 {
  Matrix<complex<double>,2>    A={2,3,{ {0.0,1.0},{5.0,2.0},{0.123,0.31},
                                        {0.0,2.0},{3.0,4.0},{0.123,0.21} }};
  Matrix<complex<double>,2>    B=trans(A);
  Matrix<complex<double>,2>    C=conjtrans(A);
  size_t flag=0;

  for(size_t i=0; i<A.L1; i++)
  {
   for(size_t j=0; j<A.L2; j++)
   {
    if(abs(A(i,j)-B(j,i))>1e-12) flag++;
    if(abs(A(i,j)-conj(C(j,i)))>1e-12) flag++;
   }
  }

  //cout<<A<<endl;
  //cout<<B<<endl;
  //cout<<C<<endl;

  if(flag==0) cout<<"Matrix 2d conj trans passed complex double test! \n";
  else cout<<"WARNING!!!!!!!!! Matrix 2d conj trans failed complex double test! \n";
 }

 void matrix_exp_test()
 {
  Matrix<complex<double>,1>    C={6,{ {0.0,1.0},{5.0,2.0},{0.123,0.31},
                                        {0.0,2.0},{3.0,4.0},{0.123,0.21} }};
  Matrix<complex<double>,1>    D=exp(C);
  size_t flag=0;
  for(size_t i=0; i<C.L1; i++)  {if(abs(D(i)-exp(C(i)))>1e-12) flag++;}
  if(flag==0) cout<<"Matrix exp passed complex double test! \n";
  else cout<<"WARNING!!!!!!!!! Matrix exp failed complex double test! \n";
  //cout<<flag<<endl;
 }

 void matrix_class_test()
 {
  matrix_1d_c_slicing();
  matrix_2d_c_slicing();
  matrix_3d_c_slicing();
  matrix_conj_test();
  matrix_2d_trans_conjtrans_test();
  matrix_exp_test();
  return;
 }
}
