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

void new_LU_decomp_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11}, 
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11}, 
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };

     f77lapack_traits<BL_INT> xlapack;                         
     LU_decomp<complex<double>,BL_INT> LU(A, &xlapack);               

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

     LU_decomp<complex<double>,BL_INT> LUC(LU);
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LU_decomp<complex<double>,BL_INT> LUR(std::move(LU));
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUR.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LU_decomp<complex<double>,BL_INT> LUEC; LUEC=LUC;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUEC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     LU_decomp<complex<double>,BL_INT> LUER; LUER=std::move(LUR);
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUER.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     if(flag==0) cout<<"New LU_decomp passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! New LU_decomp failed complex double test! \n";
 }


void new_LU_decomp_magma_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11}, 
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11}, 
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };

     magma_traits<magma_int_t> xlapack;                         
     LU_decomp<complex<double>,magma_int_t> LU(A, &xlapack);               

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

     LU_decomp<complex<double>,magma_int_t> LUC(LU);
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LU_decomp<complex<double>,magma_int_t> LUR(std::move(LU));
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUR.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }

     LU_decomp<complex<double>,magma_int_t> LUEC; LUEC=LUC;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUEC.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     LU_decomp<complex<double>,magma_int_t> LUER; LUER=std::move(LUR);
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(LUER.A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }


     if(flag==0) cout<<"New LU_decomp_magma passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! New LU_decomp_magma failed complex double test! \n";
 }
    
 void new_determinant_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     // Need to be fixed to make it general
     //magma_traits<magma_int_t> xlapack;
     //LU_decomp<complex<double>,magma_int_t>  LU( A, &xlapack );

     f77lapack_traits<BL_INT> xlapack;
     LU_decomp<complex<double>,BL_INT>  LU( A, &xlapack );

     complex<double> det = LU.determinant_in();

     complex<double> det_exact={123.11968700000003,3.3324580000000115};
     if(abs(det-det_exact)<1e-13) cout<<"New Determinant passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! New Determinant failed complex double test! \n";
     //cout<<setprecision(16);
     //cout<<det<<"\n";
 }
    
 void new_log_determinant_test()
 {
     Matrix<complex<double>,2> A={3,3,{ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
                                        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
                                        {2.123,-5.11},{5.123,-6.11},{3,0.0} } };
     A*=1.e103;

     f77lapack_traits<BL_INT> xlapack;
     LU_decomp<complex<double>,BL_INT>  LU( A, &xlapack );

     complex<double> logdet=LU.log_determinant_in();

     complex<double> logdet_exact={716.3123168546207,0.027060209772387683};
     if(abs(logdet-logdet_exact)<1e-12) cout<<"New Log_determinant passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! New Log_determinant failed complex double test! \n";
     //cout<<abs(logdet-logdet_exact)<<"\n";
     //cout<<setprecision(16);
     //cout<<logdet<<"\n";
 }


 void new_inverse_test()
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
     //f77lapack_traits<BL_INT> xlapack;
     //LU_decomp<complex<double>,BL_INT>  LU( A, &xlapack );

     magma_traits<magma_int_t> xlapack;
     LU_decomp<complex<double>,magma_int_t>  LU( A, &xlapack );

     A=LU.inverse_in();

     size_t flag=0;
     for(size_t i=0; i<A_exact.L1; i++)
     {
         for(size_t j=0; j<A_exact.L2; j++) {if(abs(A(i,j)-A_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"New Inverse passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! New Inverse failed complex double test! \n";
 }
    
 void new_solve_lineq_test()
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

     //magma_traits<magma_int_t> xlapack;
     //LU_decomp<complex<double>,magma_int_t>  LU( A, &xlapack );

     f77lapack_traits<BL_INT> xlapack;
     LU_decomp<complex<double>,BL_INT>  LU( A, &xlapack );

     Matrix<complex<double>,2> X=LU.solve_lineq_in(B);

     size_t flag=0;
     for(size_t i=0; i<X_exact.L1; i++)
     {
         for(size_t j=0; j<X_exact.L2; j++) {if(abs(X(i,j)-X_exact(i,j))>1e-13) flag++;}
     }
     if(flag==0) cout<<"New Solve_lineq passed complex double test! \n";
     else cout<<"WARNING!!!!!!!!! New Solve_lineq failed complex double test! \n";

 } 

 void matrix_linalg_test()
 {
     new_dgemm_f77_double_test();
     new_dgemm_magma_double_test();
     new_dgemm_f77_complexDouble_test();
     new_dgemm_magma_complexDouble_test();
     new_eigen_test();
     new_eigen_magma_test();
     new_LU_decomp_test();
     new_LU_decomp_magma_test();
     new_determinant_test();
     new_log_determinant_test();
     new_inverse_test();
     new_solve_lineq_test();
 }

} //end namespace matrix_hao_lib
