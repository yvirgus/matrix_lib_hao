#include "matrix_all.h"
#include <cstdio>

#include "lib_hao/matrix_linalg.h"
#include "lib_hao/f77lapack_traits.h"
#include "lib_hao/magma_traits.h"

using namespace std;

namespace matrix_hao_lib
{

 void new_dgemm_f77_double_test()
 {
     cout << "new_dgemm_f77_test" << endl;

     Matrix<double,2> a={2,3,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> b={3,2,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> c(2,2);
     Matrix<double,2> c_exact={2,2,{14.861,12.630129,20.984,23.753129}};

     typedef f77lapack_traits<BL_INT> xlapack_t;
     xlapack_t xlapack;
     linalg<BL_INT> LA(&xlapack);
     LA.gmm(a, b, c, 'N', 'N');
     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if (flag==0) 
         cout<<"New gmm passed double test! \n";
     else
         cout << "WARNING!!!!!!!!! New gmm failed double test! " << flag << " values mismatched" << endl;
 }

 void new_dgemm_magma_double_test()
 {
     cout << "new_dgemm_magma_test" << endl;

     Matrix<double,2> a={2,3,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> b={3,2,{0.0,3.0,2.123,
                             2.0,4.0,3.123 }};
     Matrix<double,2> c(2,2);
     Matrix<double,2> c_exact={2,2,{14.861,12.630129,20.984,23.753129}};

     typedef magma_traits<magma_int_t> xlapack_t;
     xlapack_t xlapack;
     linalg<magma_int_t> LA(&xlapack);
     LA.gmm(a, b, c, 'N', 'N');
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

 void new_dgemm_f77_complexDouble_test()
 {
     cout << "new_dgemm_f77_test" << endl;

     Matrix<complex<double>,2> a={2,3,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> b={3,2,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> c(2,2);
     Matrix<complex<double>,2> c_exact={2,2,
                                        { {-13.769,40.877}, {-16.551971,38.73806}, 
                                          {-17.756,56.71},  {-22.838971, 66.77106} }
     }; 


     typedef f77lapack_traits<BL_INT> xlapack_t;
     xlapack_t xlapack;
     linalg<BL_INT> LA(&xlapack);
     LA.gmm(a, b, c, 'N', 'N');

     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if (flag==0) 
         cout<<"New gmm passed complex double test! \n";
     else
         cout << "WARNING!!!!!!!!! New gmm failed complex double test! " << flag << " values mismatched" << endl;
 }
   
 void new_dgemm_magma_complexDouble_test()
 {
     cout << "new_dgemm_magma_test" << endl;

     Matrix<complex<double>,2> a={2,3,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> b={3,2,{ {0.0,0.8},{3.0,4.0},{2.123,3.11},
                                        {2.0,3.3},{4.0,5.0},{3.123,4.11} } };
     Matrix<complex<double>,2> c(2,2);
     Matrix<complex<double>,2> c_exact={2,2,
                                        { {-13.769,40.877}, {-16.551971,38.73806}, 
                                          {-17.756,56.71},  {-22.838971, 66.77106} }
     }; 

     typedef magma_traits<magma_int_t> xlapack_t;
     xlapack_t xlapack;
     linalg<magma_int_t> LA(&xlapack);
     LA.gmm(a, b, c, 'N', 'N');

     size_t flag=0;
     for(size_t i=0; i<c.L1; i++)
     {
         for(size_t j=0; j<c.L2; j++) {if(abs(c(i,j)-c_exact(i,j))>1e-5) flag++;}
     }
     if (flag==0) 
         cout<<"New gmm_magma passed complex double test! \n";
     else
         cout << "WARNING!!!!!!!!! New gmm_magma failed complex double test! " << flag << " values mismatched" << endl;
 }
    
    void new_eigen_test()
    {
     Matrix<complex<double>,2> a={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-4.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-3.11},{5.123,-3.11},{3,0.0}     } };
     Matrix<double,1> w(3);
     check_Hermitian(a);
     //cout << a << endl;
     typedef f77lapack_traits<BL_INT> xlapack_t;
     xlapack_t xlapack;
     linalg<BL_INT> LA(&xlapack);
     LA.eigen(a, w, 'V', 'U');     
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

     if(flag==0) cout<<"New Eigen passed Hermition test! \n";
     else cout<<"WARNING!!!!!!!!! New Eigen failed Hermintion test! \n";
     //cout<<setprecision(16);
     //cout<<w<<endl;
     //cout<<a<<endl;

    }

    void new_eigen_magma_test()
    {
     Matrix<complex<double>,2> a={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-4.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-3.11},{5.123,-3.11},{3,0.0}     } };
     Matrix<double,1> w(3);
     check_Hermitian(a);
     //cout << a << endl;
     typedef magma_traits<magma_int_t> xlapack_t;
     xlapack_t xlapack;
     linalg<magma_int_t> LA(&xlapack);
     LA.eigen(a, w, 'V', 'U');     
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

     if(flag==0) cout<<"New Eigen_magma passed Hermition test! \n";
     else cout<<"WARNING!!!!!!!!! New Eigen_magma failed Hermintion test! \n";
     //cout<<setprecision(16);
     //cout<<w<<endl;
     //cout<<a<<endl;

    }

 void matrix_linalg_test()
 {
     new_dgemm_f77_double_test();
     new_dgemm_magma_double_test();
     new_dgemm_f77_complexDouble_test();
     new_dgemm_magma_complexDouble_test();
     new_eigen_test();
     new_eigen_magma_test();
 }

} //end namespace matrix_hao_lib
