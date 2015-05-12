#ifndef MATRIX_HAO_ELEMENT_WISE
#define MATRIX_HAO_ELEMENT_WISE

#include <cmath>

namespace matrix_hao_lib
{

 //for add: (array+array)
 template <class T, int D>
 Matrix<T,D> operator + (const Matrix<T,D>& A,const Matrix<T,D>& B) {Matrix<T,D> C;C=A; C+=B; return C;} 
 template <class T, int D>
 Matrix<T,D> operator + (const Matrix<T,D>& A, Matrix<T,D>&& B)     {Matrix<T,D> C;C=std::move(B); C+=A; return C;} 
 template <class T, int D>
 Matrix<T,D> operator + (Matrix<T,D>&& A,const Matrix<T,D>& B)      {Matrix<T,D> C;C=std::move(A); C+=B; return C;} 
 template <class T, int D>
 Matrix<T,D> operator + (Matrix<T,D>&& A,Matrix<T,D>&& B)           {Matrix<T,D> C;C=std::move(A); C+=B; return C;} 


 //for minus:(array-array)
 template <class T, int D>
 Matrix<T,D> operator - (const Matrix<T,D>& A,const Matrix<T,D>& B) {Matrix<T,D> C;C=A; C-=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator - (const Matrix<T,D>& A, Matrix<T,D>&& B){Matrix<T,D> C;C=std::move(B); C.min_add_equal(A); return C;} 
 template <class T, int D>
 Matrix<T,D> operator - (Matrix<T,D>&& A,const Matrix<T,D>& B)      {Matrix<T,D> C;C=std::move(A); C-=B; return C;} 
 template <class T, int D>
 Matrix<T,D> operator - (Matrix<T,D>&& A,Matrix<T,D>&& B)           {Matrix<T,D> C;C=std::move(A); C-=B; return C;} 


 //for time: (array*array)
 template <class T, int D>
 Matrix<T,D> operator * (const Matrix<T,D>& A,const Matrix<T,D>& B) {Matrix<T,D> C;C=A; C*=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator * (const Matrix<T,D>& A, Matrix<T,D>&& B)     {Matrix<T,D> C;C=std::move(B); C*=A; return C;}
 template <class T, int D>
 Matrix<T,D> operator * (Matrix<T,D>&& A,const Matrix<T,D>& B)      {Matrix<T,D> C;C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator * (Matrix<T,D>&& A,Matrix<T,D>&& B)           {Matrix<T,D> C;C=std::move(A); C*=B; return C;}



 //for div:(array/array)
 template <class T, int D>
 Matrix<T,D> operator / (const Matrix<T,D>& A,const Matrix<T,D>& B) {Matrix<T,D> C;C=A; C/=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator / (const Matrix<T,D>& A, Matrix<T,D>&& B){Matrix<T,D> C;C=std::move(B); C.inv_div_equal(A); return C;}
 template <class T, int D>
 Matrix<T,D> operator / (Matrix<T,D>&& A,const Matrix<T,D>& B)      {Matrix<T,D> C;C=std::move(A); C/=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator / (Matrix<T,D>&& A,Matrix<T,D>&& B)           {Matrix<T,D> C;C=std::move(A); C/=B; return C;}


 //for add: (array+scalar)
 template <class T, int D>
 Matrix<T,D> operator + (const Matrix<T,D>& A,const T & B) {Matrix<T,D> C;C=A; C+=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator + (Matrix<T,D>&& A, const T & B)     {Matrix<T,D> C;C=std::move(A); C+=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator + (const T & B,const Matrix<T,D>& A) {Matrix<T,D> C;C=A; C+=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator + (const T & B,Matrix<T,D>&& A)      {Matrix<T,D> C;C=std::move(A); C+=B; return C;}


 //for minus: (array-scalar)
 template <class T, int D>
 Matrix<T,D> operator - (const Matrix<T,D>& A,const T & B) {Matrix<T,D> C;C=A; C-=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator - (Matrix<T,D>&& A, const T & B)     {Matrix<T,D> C;C=std::move(A); C-=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator - (const T & B,const Matrix<T,D>& A) {Matrix<T,D> C;C=A; C.min_add_equal(B);return C;}
 template <class T, int D>
 Matrix<T,D> operator - (const T & B,Matrix<T,D>&& A)      {Matrix<T,D> C;C=std::move(A);C.min_add_equal(B);return C;}



 //for time: (array*scalar)
 template <class T, int D>
 Matrix<T,D> operator * (const Matrix<T,D>& A,const T & B) {Matrix<T,D> C;C=A; C*=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator * (Matrix<T,D>&& A, const T & B)     {Matrix<T,D> C;C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator * (const T & B,const Matrix<T,D>& A) {Matrix<T,D> C;C=A; C*=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator * (const T & B,Matrix<T,D>&& A)      {Matrix<T,D> C;C=std::move(A); C*=B; return C;}


 //for div: (array/scalar)
 template <class T, int D>
 Matrix<T,D> operator / (const Matrix<T,D>& A,const T & B) {Matrix<T,D> C;C=A; C/=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator / (Matrix<T,D>&& A, const T & B)     {Matrix<T,D> C;C=std::move(A); C/=B; return C;}
 template <class T, int D>
 Matrix<T,D> operator / (const T & B,const Matrix<T,D>& A) {Matrix<T,D> C;C=A; C.inv_div_equal(B);return C;}
 template <class T, int D>
 Matrix<T,D> operator / (const T & B,Matrix<T,D>&& A)      {Matrix<T,D> C;C=std::move(A);C.inv_div_equal(B);return C;}

} //end namespace matrix_hao_lib

#endif
