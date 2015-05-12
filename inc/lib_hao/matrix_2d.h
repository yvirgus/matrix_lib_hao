#ifndef MATRIX_HAO_2D
#define MATRIX_HAO_2D

#include "lib_hao/matrix_base.h"

namespace matrix_hao_lib
{

 template<class T> class Matrix<T,2>:public Matrix_base<T>
 {
  public:
     size_t L1,L2;
 
     /*********************/
     /*PART I: CONSTRUCTOR*/
     /*********************/
 
     //Void constructor:
     Matrix(): Matrix_base<T>(),L1(0),L2(0){}
     //N constructor: vector is not initialized
     Matrix(size_t n1,size_t n2):Matrix_base<T>(n1*n2),L1(n1),L2(n2){}
     //List constructor:
     Matrix(size_t n1,size_t n2,const std::initializer_list <T> &args):Matrix_base<T>(args),L1(n1),L2(n2){lenth_check();}
     //Copy constructor:
     Matrix(const Matrix<T,2>& x):Matrix_base<T>(x),L1(x.L1),L2(x.L2){}
     //Move constructor:
     Matrix(Matrix<T,2>&& x):Matrix_base<T>(std::move(x)),L1(x.L1),L2(x.L2){} 
 
     /*********************/
     /*PART II: DESTRUCTOR*/
     /*********************/
     ~Matrix() {} // destruction in Matrix_base will be call automatically
 
     /***************************/
     /*PART III: MEMBER FUNCTION*/
     /***************************/
     void range_check(size_t i,size_t j) const  {if((i>=L1)||(j>=L2) ) throw std::invalid_argument("exceed the range of 2D array");}
     void lenth_check() const {if((L1*L2)!=this->L) throw std::invalid_argument("size not consistent in 2D array");}
     // subscripting, the matrix is column arranged:
     T operator ()(size_t i,size_t j) const  {/*range_check(i,j);*/ return this->base_array[i+j*L1];}
     T& operator()(size_t i,size_t j)        {/*range_check(i,j);*/ return this->base_array[i+j*L1];}
     Matrix<T,1> operator[](size_t i)
     {
         if(i>=L2) throw std::invalid_argument("i exceed the column range of 2D array");
         Matrix<T,1> A; A.L_f()=L1; A.owns=false; A.L1=L1;
         A.base_array=this->base_array; A.base_array+=(L1*i);
         return A; 
     }
 
     /**************************/
     /*PART IV: INSIDE OVERLOAD*/
     /**************************/
    //copy-assigment array to Matrix_base 
     Matrix<T,2>& operator  = (const Matrix_base<T>& B) {this->copy(B);lenth_check();return *this;}
     //move-assigment array to Matrix_base
     Matrix<T,2>& operator  = (Matrix_base<T>&& B) 
     {if(B.owns && this->owns) this->swap(B); else this->copy(B); lenth_check(); return *this;}
 
     //copy-assigment array to Matrix<T,2>
     Matrix<T,2>& operator  = (const Matrix<T,2>& B)    {L1=B.L1;L2=B.L2;this->copy(B);return *this;}
     //move-assigment array to Matrix<T,2>
     Matrix<T,2>& operator  = (Matrix<T,2>&& B)         
     {L1=B.L1;L2=B.L2; if(B.owns && this->owns) this->swap(B); else this->copy(B); return *this;}
 
 
     // += array A=A+B
     Matrix<T,2>& operator += (const Matrix<T,2>& B) {this->add_equal(B);return *this;}
     // -= array A=A-B
     Matrix<T,2>& operator -= (const Matrix<T,2>& B) {this->min_equal(B);return *this;}
     // *= array A=A*B
     Matrix<T,2>& operator *= (const Matrix<T,2>& B) {this->tim_equal(B);return *this;}
     // /= array A=A/B
     Matrix<T,2>& operator /= (const Matrix<T,2>& B) {this->div_equal(B);return *this;}
 
 
     // copy-assigment scalar 
     Matrix<T,2>& operator  = (const T & B) {this->copy(B);return *this;}
     // += scalar A=A+B
     Matrix<T,2>& operator += (const T & B) {this->add_equal(B);return *this;}
     // -= scalar A=A-B
     Matrix<T,2>& operator -= (const T & B) {this->min_equal(B);return *this;}
     // *= scalar A=A*B
     Matrix<T,2>& operator *= (const T & B) {this->tim_equal(B);return *this;}
     // /= scalar A=A/B
     Matrix<T,2>& operator /= (const T & B) {this->div_equal(B);return *this;}

 }; //end class matrix<T,2>


 /***************************/
 /*PART IV: OUTSIDE OVERLOAD*/
 /***************************/
 //for minus sign:  -(array) 
 template <class T>
 Matrix<T,2> operator -(const Matrix<T,2>& B) {Matrix<T,2> C(B.L1,B.L2);C=B.min_sign();return C;}

 //for conj
 template <class T>
 Matrix<std::complex<T>,2> conj(const Matrix<std::complex<T>,2>& A) {Matrix<std::complex<T>,2> B(A.L1,A.L2);B=A.conj();return B;}

 //for trans
 template <class T>
 Matrix<T,2> trans(const Matrix<T,2>& A) 
 {
     Matrix<T,2> B(A.L2,A.L1);
     for(size_t i=0; i<A.L1; i++)
     {
         for(size_t j=0; j<A.L2; j++) B(j,i)=A(i,j);
     }
     return B;
 }

 //for conjtrans
 template <class T>
 Matrix<std::complex<T>,2> conjtrans(const Matrix<std::complex<T>,2>& A) 
 {
     Matrix<std::complex<T>,2> B(A.L2,A.L1);
     for(size_t i=0; i<A.L1; i++)
     {
         for(size_t j=0; j<A.L2; j++) B(j,i)=std::conj(A(i,j));
     }
     return B;
 }


 //for cout
 template <class T>
 std::ostream& operator<< (std::ostream &out, const Matrix<T,2> &object)
 {
     out<<std::boolalpha;
     out<<std::scientific;
     out<<"Own: "<<object.owns<<"\n";
     out<<"Size: "<<object.L1<<" "<<object.L2<<"\n";
     for (size_t i = 0; i<object.L1; ++i) 
     {
         for (size_t j = 0; j<object.L2; ++j) out<<" "<<object(i,j);
         out<<"\n";
     }
     return out;
 }


} //end namespace matrix_hao_lib

#endif
