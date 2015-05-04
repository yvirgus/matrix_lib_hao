#include "matrix_all.h"
#include <cstdio>

using namespace std;

namespace matrix_hao_lib
{

 void gmm_float_test()
 {
     Matrix<float,2> a={2,3,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<float,2> b={3,2,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<float,2> c(2,2);
     Matrix<float,2> c_exact={2,2,{14.861,12.630129,20.984,23.753129}};
     gmm(a,b,c);
     //cout<<setprecision(10);
     //cout<<c<<endl;
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if(flag==0) cout<<"Gmm passed float test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm failed float test! \n";
 }

 void gmm_magma_float_test()
 {
     Matrix<float,2> a={2,3,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<float,2> b={3,2,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<float,2> c(2,2);
     Matrix<float,2> c_exact={2,2,{14.861,12.630129,20.984,23.753129}};
     //cout<<a<<endl;
     //cout<<b<<endl;
     //cout<<c<<endl;
     gmm_magma(a,b,c);
     //cout<<setprecision(10);
     //cout<<c<<endl;
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if(flag==0) cout<<"Gmm_magma passed float test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm_magma failed float test! \n";
 }

 void gmm_double_test()
 {
     Matrix<double,2> a={2,3,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> b={3,2,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> c(2,2);
     Matrix<double,2> c_exact={2,2,{14.861,12.630129,20.984,23.753129}};
     //cout<<a<<endl;
     //cout<<b<<endl;
     //cout<<c<<endl;
     gmm(a,b,c);
     //cout<<setprecision(10);
     //cout<<c<<endl;
     size_t flag=0;
     //cout<<c<<endl;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Gmm passed double test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm failed double test! \n";
 }

 void gmm_magma_double_test()
 {
     Matrix<double,2> a={2,3,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> b={3,2,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> c(2,2);
     Matrix<double,2> c_exact={2,2,{14.861,12.630129,20.984,23.753129}};
     //cout<<a<<endl;
     //cout<<b<<endl;
     //cout<<c<<endl;
     gmm_magma(a,b,c);
     //cout<<setprecision(10);
     //cout<<c<<endl;
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Gmm_magma passed double test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm_magma failed double test! \n";
 }

 void gmm_complexfloat_test()
 {
     Matrix<complex<float>,2> a={2,3,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                       {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<float>,2> b={3,2,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                       {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<float>,2> c(2,2);
     Matrix<complex<float>,2> c_exact={2,2,
                                       { {-13.769,40.877}, {-16.551971,38.73806},
                                         {-17.756,56.71},  {-22.838971, 66.77106} }
                                       };
     gmm(a,b,c);
     //cout<<setprecision(10);
     //cout<<c<<endl;
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if(flag==0) cout<<"Gmm passed complex float test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm failed complex float test! \n";
 }


 void gmm_magma_complexfloat_test()
 {
     Matrix<complex<float>,2> a={2,3,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                       {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<float>,2> b={3,2,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                       {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<float>,2> c(2,2);
     Matrix<complex<float>,2> c_exact={2,2,
                                       { {-13.769,40.877}, {-16.551971,38.73806},
                                         {-17.756,56.71},  {-22.838971, 66.77106} }
                                       };
     //cout<<a<<endl;
     //cout<<b<<endl;
     //cout<<c<<endl;
     gmm_magma(a,b,c);
     //cout<<setprecision(10);
     //cout<<c<<endl;
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if(flag==0) cout<<"Gmm_magma passed complex float test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm_magma failed complex float test! \n";
 }


 void gmm_complexdouble_test()
 {
     Matrix<complex<double>,2> a={2,3,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> b={3,2,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> c(2,2);
     Matrix<complex<double>,2> c_exact={2,2,
                                       { {-13.769,40.877}, {-16.551971,38.73806}, 
                                         {-17.756,56.71},  {-22.838971, 66.77106} }
                                       }; 
     gmm(a,b,c);

     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Gmm passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm failed complex double test! \n";
 }

 void gmm_magma_complexdouble_test()
 {
     Matrix<complex<double>,2> a={2,3,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> b={3,2,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> c(2,2);
     Matrix<complex<double>,2> c_exact={2,2,
                                       { {-13.769,40.877}, {-16.551971,38.73806}, 
                                         {-17.756,56.71},  {-22.838971, 66.77106} }
                                       }; 
     //cout<<a<<endl;
     //cout<<b<<endl;
     //cout<<c<<endl;
     gmm_magma(a,b,c);
     //cout<<c<<endl;
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Gmm_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Gmm_magma failed complex double test! \n";
 }
 

 void eigen_test()
 {
     Matrix<complex<double>,2> a={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-4.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-3.11},{5.123,-3.11},{3,0.0}     } };
     Matrix<double,1> w(3);
     check_Hermitian(a);
     //cout << a << endl;
     eigen(a,w);
     //cout << a << endl;
     Matrix<complex<double>,2> a_exact={3,3,{ {-0.4053433965286621, -0.3217472918461721},
                                              {-0.3733963692733272,  0.6060804552476304},    
                                              {0.47478104875888194,  0},
                                              {0.13035873463974057,  0.6902772720595061},   
                                              {-0.26751344366934643,-0.20279279787239068},    
                                              {0.6275631654012745,   0},
                                              {-0.179307184764388,   0.4544757777410628},
                                              {-0.5593786354476359,  0.26009385608337265},
                                              {-0.6170473475925071,  0}     } };
     Matrix<double,1> w_exact={3,{-4.7040348985237666,-1.1586196209127053,11.862654519436473}};

     size_t flag=0;
     for(size_t i=0; i<a.L1; i++)
     {
         for(size_t j=0; j<a.L2; j++) {if(abs(abs(a(i,j))-abs(a_exact(i,j)))>1e-13) flag++;}
     }

     for(size_t i=0; i<w.L1; i++) {if(abs(w(i)-w_exact(i))>1e-13) flag++;}

     if(flag==0) cout<<"Eigen passed Hermition test! \n";
     else cout<<"WARNING!!!!!!!!! Eigen failed Hermintion test! \n";
     //cout<<setprecision(16);
     //cout<<w<<endl;
     //cout<<a<<endl;
 }

 void eigen_magma_test()
 {
     Matrix<complex<double>,2> a={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-4.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-3.11},{5.123,-3.11},{3,0.0}     } };
     Matrix<double,1> w(3);
     check_Hermitian(a);
     //cout << a << endl;
     //cout << w << endl;
     eigen_magma(a,w);
     //cout << a << endl;
     //cout << w << endl;
     Matrix<complex<double>,2> a_exact={3,3,{ {-0.4053433965286621, -0.3217472918461721},
                                              {-0.3733963692733272,  0.6060804552476304},    
                                              {0.47478104875888194,  0},
                                              {0.13035873463974057,  0.6902772720595061},   
                                              {-0.26751344366934643,-0.20279279787239068},    
                                              {0.6275631654012745,   0},
                                              {-0.179307184764388,   0.4544757777410628},
                                              {-0.5593786354476359,  0.26009385608337265},
                                              {-0.6170473475925071,  0}     } };
     Matrix<double,1> w_exact={3,{-4.7040348985237666,-1.1586196209127053,11.862654519436473}};

     size_t flag=0;
     for(size_t i=0; i<a.L1; i++)
     {
         for(size_t j=0; j<a.L2; j++) {if(abs(abs(a(i,j))-abs(a_exact(i,j)))>1e-13) flag++;}
     }

     for(size_t i=0; i<w.L1; i++) {if(abs(w(i)-w_exact(i))>1e-13) flag++;}

     if(flag==0) cout<<"Eigen_magma passed Hermition test! \n";
     else cout<<"WARNING!!!!!!!!! Eigen_magma failed Hermintion test! \n";
     //cout<<setprecision(16);
     //cout<<w<<endl;
     //cout<<a<<endl;
 }

 void LUDecomp_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11}, 
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11}, 
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     LUDecomp<complex<double>> LU(A);
     //cout << LU.A << endl;
     //cout << LU.ipiv << endl;

     Matrix<complex<double>,2> A_exact={3,3,{ {3,4} ,   {0.75236,0.03351999999999994}, {0.12,-0.16},
                                        {2,0},   {3.6182800000000004,3.04296},    {0.21807341113346007,-0.647707935025115},
                                        {5.123,-6.11},{-1.05914748,4.42519664},{-0.14942307391746978,-5.208155953378981} } };

     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LU.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LUDecomp<complex<double>> LUC(LU);
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LUDecomp<complex<double>> LUR(std::move(LU));
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUR.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LUDecomp<complex<double>> LUEC;LUEC=LUC;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUEC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     LUDecomp<complex<double>> LUER;LUER=std::move(LUR);
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUER.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     if(flag==0) cout<<"LUDecomp passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! LUDecomp failed complex double test! \n";
 }


  /********************************************************************************************************************************************************/
  // This test function was written to diagnose the issue with LU decomposition function in magma (magma_zgetrf).
  // The issus: magma_zgetrf will give correct value of the matrix after the LU decomposition. Howerver, the pivot value (ipiv) is 'incorrect'. 
  // It turns out that Hao's code asks for 64-bit while compiling the code (see the Makefile or CMakeLists.txt).
  // The MAGMA library was built in 32-bit environment. When the code is run, the size of magma_int_t became 8 instead of 4.
  // This results in 'incorrect' value for ipiv. Wirawan found this issue by converting the value in hex. 
  // In the mean time, we will remove the 64-bit flag from MKL and deactivate all Hao's FORTRAN_NAME functions which require MKL library. Now the environment is 32-bit.

  /*  void LUDecomp_magma_test_no_class()
  {
    magmaDoubleComplex *A;
    magma_int_t *ipiv;
    magma_int_t N=3, info;

    // allocate memory
    magma_imalloc_cpu(&ipiv, N+2);
    magma_zmalloc_pinned(&A, N*N);

    // fill in the matrix A 
    MAGMA_Z_REAL(A[0])=1.0;
    MAGMA_Z_IMAG(A[0])=0.0;
    MAGMA_Z_REAL(A[1])=3.0;
    MAGMA_Z_IMAG(A[1])=4.0;
    MAGMA_Z_REAL(A[2])=2.123;
    MAGMA_Z_IMAG(A[2])=3.11;
    MAGMA_Z_REAL(A[3])=3.0;
    MAGMA_Z_IMAG(A[3])=-2.0;
    MAGMA_Z_REAL(A[4])=2.0;
    MAGMA_Z_IMAG(A[4])=0.0;
    MAGMA_Z_REAL(A[5])=5.123;
    MAGMA_Z_IMAG(A[5])=3.11;
    MAGMA_Z_REAL(A[6])=2.123;
    MAGMA_Z_IMAG(A[6])=-5.11;
    MAGMA_Z_REAL(A[7])=5.123;
    MAGMA_Z_IMAG(A[7])=-6.11;
    MAGMA_Z_REAL(A[8])=3.0;
    MAGMA_Z_IMAG(A[8])=0.0;
    cout << "N = " << N << std::endl;

    // check the sizeof magma_int_t (4 or 8 for32-bit or 64-bit environment, respectively)
    cout << "Sizeof magma_int_t = " << sizeof(magma_int_t) << std::endl;

    for (int i = 0; i < N+2; ++i) ipiv[i] = -1;
    cout << "The value of ipiv before: \n" <<  ipiv[0] << " "<< ipiv[1] << " "<< ipiv[2] <<" "<< ipiv[3] <<" " << ipiv[4] << std::endl;
    cout << "A original value" << std::endl;
    for(int i=0;i<N*N;i++)
      { 
        //printf("%f + i*%f ", MAGMA_Z_REAL(A[i]), MAGMA_Z_IMAG(A[i]));
        cout << MAGMA_Z_REAL(A[i]) << " + i*" << MAGMA_Z_IMAG(A[i]) << "   ";
        if ((i+1)%(N) == 0) cout << "\n";
       }
    cout << "\n";
    magma_zgetrf( N, N, A, N, ipiv+1, &info);
    cout << "info = " << info << std::endl;
    cout << "The value of ipiv after: \n" <<  ipiv[0] << " "<< ipiv[1] << " "<< ipiv[2] <<" "<< ipiv[3] <<" " << ipiv[4] << std::endl;
    using std::printf;

    // check the value in hexadecimal representation
    printf("In hex: %016lx %016lx %016lx %016lx %016lx\n",
           ipiv[0], ipiv[1], ipiv[2], ipiv[3], ipiv[4]);
    for(int i=0;i<N*N;i++)
      { 
        //printf("%f + i*%f ", MAGMA_Z_REAL(A[i]), MAGMA_Z_IMAG(A[i]));
        cout << MAGMA_Z_REAL(A[i]) << " + i*" << MAGMA_Z_IMAG(A[i]) << "   ";
        if ((i+1)%(N) == 0) cout << "\n";
       }
    cout << "\n";
    magma_free_cpu(ipiv);   
    magma_free_pinned(A);
    if(info<0) {cout<<"The "<<info<<"-th parameter is illegal!\n"; throw std::runtime_error(" ");} 
    }*/

  /********************************************************************************************************************************************************/

 void LUDecomp_magma_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11}, 
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11}, 
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     
     //cout << A << endl;
     LUDecomp_magma<complex<double>> LU(A);
     //cout << LU.A << endl;
     //cout << "The value of the pivot: " << LU.ipiv << endl;
     Matrix<complex<double>,2> A_exact={3,3,{ {3,4} ,   {0.75236,0.03351999999999994}, {0.12,-0.16},
                                        {2,0},   {3.6182800000000004,3.04296},    {0.21807341113346007,-0.647707935025115},
                                        {5.123,-6.11},{-1.05914748,4.42519664},{-0.14942307391746978,-5.208155953378981} } };

     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LU.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LUDecomp_magma<complex<double>> LUC(LU);
     //cout << LUC.A << endl;
     //cout << "The value of the pivot: " << LUC.ipiv << endl;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LUDecomp_magma<complex<double>> LUR(std::move(LU));
     //cout << LUR.A << endl;
     //cout << "The value of the pivot: " << LUR.ipiv << endl;

     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUR.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LUDecomp_magma<complex<double>> LUEC;LUEC=LUC;
     //cout << LUEC.A << endl;
     //cout << "The value of the pivot: " << LUEC.ipiv << endl;

     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUEC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     LUDecomp_magma<complex<double>> LUER;LUER=std::move(LUR);
     //cout << LUER.A << endl;
     //cout << "The value of the pivot: " << LUER.ipiv << endl;     
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUER.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     if(flag==0) cout<<"LUDecomp_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! LUDecomp_magma failed complex double test! \n";
 }

 void determinant_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     complex<double> det=determinant(LUDecomp<complex<double>>(A));
     complex<double> det_exact={123.11968700000003,3.3324580000000115};
     if(abs(det-det_exact)<1e-13) cout<<"Determinant passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Determinant failed complex double test! \n";
     //cout<<setprecision(16);
     //cout<<det<<"\n";
 }

 void log_determinant_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     A*=1.e103;
     complex<double> logdet=log_determinant(LUDecomp<complex<double>>(A));
     complex<double> logdet_exact={716.3123168546207,0.027060209772387683};
     if(abs(logdet-logdet_exact)<1e-12) cout<<"Log_determinant passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Log_determinant failed complex double test! \n";
     //cout<<abs(logdet-logdet_exact)<<"\n";
     //cout<<setprecision(16);
     //cout<<logdet<<"\n";
 }

 void determinant_magma_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     complex<double> det=determinant_magma(LUDecomp_magma<complex<double>>(A));
     complex<double> det_exact={123.11968700000003,3.3324580000000115};
     if(abs(det-det_exact)<1e-13) cout<<"Determinant_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Determinant_magma failed complex double test! \n";
     //cout<<setprecision(16);
     //cout<<det<<"\n";
 }

 void log_determinant_magma_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     A*=1.e103;
     complex<double> logdet=log_determinant_magma(LUDecomp_magma<complex<double>>(A));
     complex<double> logdet_exact={716.3123168546207,0.027060209772387683};
     if(abs(logdet-logdet_exact)<1e-12) cout<<"Log_determinant_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Log_determinant_magma failed complex double test! \n";
     //cout<<abs(logdet-logdet_exact)<<"\n";
     //cout<<setprecision(16);
     //cout<<logdet<<"\n";
 }


 void inverse_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     Matrix<complex<double>,2> A_exact={3,3,{ {-0.31516333912326305,0.13336022037456957} , 
                                              {0.16746685439563874,-0.0779491606298965}, 
                                              {-0.005504176768078849,0.1918486231848867},
                                              {0.1412286826747599,-0.11408929794801193},   
                                              {-0.1402834127458906,0.038283792754219295},    
                                              {0.061029436341995695,0.01438130659499342},
                                              {-0.01293596267860185,-0.1487405620815458},
                                              {0.17584867623524927,-0.010672609392757534},
                                              {-0.12306156095719788,-0.04540218264765162} } };
     A=inverse(LUDecomp<complex<double>>(A));
     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Inverse passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Inverse failed complex double test! \n";
 }

 void solve_lineq_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     Matrix<complex<double>,2> B={3,2,{ {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
                                        {3.0,-6.0},   {2.0,1.0},    {6.123,3.11},} };
     Matrix<complex<double>,2> X_exact={3,2,{ {0.785989996146147, 0.12584834096778363} ,   
                                              {0.3050317378766687,-0.22890518276854455},    
                                              {-0.1429470443202702,0.20747587687923086},
                                              {0.6345942167676883, 1.253141477086266},   
                                              {0.825768240961444,-0.8208234397212029},   
                                              {0.6299516251873555,0.037643960766659545},} };
     Matrix<complex<double>,2> X=solve_lineq(LUDecomp<complex<double>>(A),B);

     size_t flag=0;
     for(size_t i=0; i<X_exact.L1; i++)
     {
         for(size_t j=0; j<X_exact.L2; j++) {if(abs(X(i,j)-X_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Solve_lineq passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Solve_lineq failed complex double test! \n";

 } 

 void QRMatrix_test()
 {
     Matrix<complex<double>,2> A={3,2,{ {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
                                        {3.0,-6.0},   {2.0,1.0},    {6.123,3.11},} };
     double det=QRMatrix(A);
     Matrix<complex<double>,2> A_exact={3,2,{ {-0.26392384387316437, 0} ,   
                                              {-0.3958857658097466 , 0.6598096096829109},    
                                              {-0.41211708220794624, 0.41040157722277065},
                                              {0.20568020122880237 , 0.7338652779407804},   
                                              {-0.41851770493832796, 0.22064009932009565},    
                                              {0.3071492824057953  ,-0.3177382636670606},} };
     double det_exact=51.76794728400964;
     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(abs(A(i,j))-abs(A_exact(i,j)))>1e-12) flag++;}
     }
     if(abs(det-det_exact)>1e-12) flag++;
     if(flag==0) cout<<"QRMatrix passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! QRMatrix failed complex double test! \n";
     //cout<<setprecision(16);
     //cout<<A<<endl;
     //cout<<det<<endl;
 }

 void D_Multi_Matrix_test()
 {
     Matrix<complex<double>,2> A={3,2,{ {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
                                        {3.0,-6.0},   {2.0,1.0},    {6.123,3.11},} };
     Matrix<complex<double>,1> D={3, { {1.2,0.0},{2.0,0.0},{3.0,0.0} } };
     Matrix<complex<double>,2> B=D_Multi_Matrix(D,A);
     Matrix<complex<double>,2> B_exact={3,2,{ {2.4,0.0} ,   {6.0,10.0},    {9.369,9.33},
                                              {3.6,-7.2},   {4.0,2.0 },    {18.369,9.33},} };
     size_t flag=0;
     for(size_t i=0; i<B_exact.L1; i++)
     {
         for(size_t j=0; j<B_exact.L2; j++) {if(abs(abs(B(i,j))-abs(B_exact(i,j)))>1e-12) flag++;}
     }
     if(flag==0) cout<<"D_Multi_Matrix passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! D_Multi_Matrix failed complex double test! \n"; 
 }

 void inverse_magma_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     Matrix<complex<double>,2> A_exact={3,3,{ {-0.31516333912326305,0.13336022037456957} , 
                                              {0.16746685439563874,-0.0779491606298965}, 
                                              {-0.005504176768078849,0.1918486231848867},
                                              {0.1412286826747599,-0.11408929794801193},   
                                              {-0.1402834127458906,0.038283792754219295},    
                                              {0.061029436341995695,0.01438130659499342},
                                              {-0.01293596267860185,-0.1487405620815458},
                                              {0.17584867623524927,-0.010672609392757534},
                                              {-0.12306156095719788,-0.04540218264765162} } };
     A=inverse_magma(LUDecomp_magma<complex<double>>(A));
     //cout << A << std::endl;
     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Inverse_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Inverse_magma failed complex double test! \n";
 }

 void solve_lineq_magma_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     Matrix<complex<double>,2> B={3,2,{ {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
                                        {3.0,-6.0},   {2.0,1.0},    {6.123,3.11},} };
     Matrix<complex<double>,2> X_exact={3,2,{ {0.785989996146147, 0.12584834096778363} ,   
                                              {0.3050317378766687,-0.22890518276854455},    
                                              {-0.1429470443202702,0.20747587687923086},
                                              {0.6345942167676883, 1.253141477086266},   
                                              {0.825768240961444,-0.8208234397212029},   
                                              {0.6299516251873555,0.037643960766659545},} };
     //cout << A << std::endl;
     //cout << B << std::endl;     
     Matrix<complex<double>,2> X=solve_lineq_magma(LUDecomp_magma<complex<double>>(A),B);
     //cout << X << std::endl;

     size_t flag=0;
     for(size_t i=0; i<X_exact.L1; i++)
     {
         for(size_t j=0; j<X_exact.L2; j++) {if(abs(X(i,j)-X_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"Solve_lineq_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! Solve_lineq_magma failed complex double test! \n";
 } 

 void QRMatrix_magma_test()
 {
     Matrix<complex<double>,2> A={3,2,{ {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
                                        {3.0,-6.0},   {2.0,1.0},    {6.123,3.11},} };
     //cout<<A<<endl;
     double det=QRMatrix_magma(A);
     Matrix<complex<double>,2> A_exact={3,2,{ {-0.26392384387316437, 0} ,   
                                              {-0.3958857658097466 , 0.6598096096829109},    
                                              {-0.41211708220794624, 0.41040157722277065},
                                              {0.20568020122880237 , 0.7338652779407804},   
                                              {-0.41851770493832796, 0.22064009932009565},    
                                              {0.3071492824057953  ,-0.3177382636670606},} };
     double det_exact=51.76794728400964;
     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(abs(A(i,j))-abs(A_exact(i,j)))>1e-12) flag++;}
     }
     if(abs(det-det_exact)>1e-12) flag++;
     if(flag==0) cout<<"QRMatrix_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! QRMatrix_magma failed complex double test! \n";
     //cout<<setprecision(16);
     //cout<<A<<endl;
     //cout<<det<<endl;
 }

 void matrix_2d_blas_lapack_test()
 {
     
     gmm_float_test();
     gmm_magma_float_test();
     /*
     gmm_double_test();
     gmm_magma_double_test();
     gmm_complexfloat_test();
     gmm_magma_complexfloat_test();
     gmm_complexdouble_test();
     gmm_magma_complexdouble_test();
     eigen_test();
     eigen_magma_test();
     LUDecomp_test();
     LUDecomp_magma_test();
     //LUDecomp_magma_test_no_class();
     determinant_test();
     determinant_magma_test();
     log_determinant_test();
     log_determinant_magma_test();
     inverse_test();
     inverse_magma_test();
     solve_lineq_test();
     solve_lineq_magma_test();
     QRMatrix_test();
     QRMatrix_magma_test();
     D_Multi_Matrix_test();
     */
 }

} //end namespace matrix_hao_lib
