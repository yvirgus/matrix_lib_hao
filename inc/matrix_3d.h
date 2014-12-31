#ifndef MATRIX_HAO_3D
#define MATRIX_HAO_3D

namespace matrix_hao_lib
{
 template<class T> class Matrix<T,3>:public Matrix_base<T>
 {
  public:
    size_t L1,L2,L3;

    /*********************/
    /*PART I: CONSTRUCTOR*/
    /*********************/

    //Void constructor:
    Matrix(): Matrix_base<T>(),L1(0),L2(0),L3(0){}
    //N constructor: vector is not initialized
    Matrix(size_t n1,size_t n2,size_t n3):Matrix_base<T>(n1*n2*n3),L1(n1),L2(n2),L3(n3){}
    //List constructor:
    Matrix(size_t n1,size_t n2,size_t n3,const std::initializer_list <T> &args):Matrix_base<T>(args),L1(n1),L2(n2),L3(n3)
    {lenth_check();}
    //Copy constructor:
    Matrix(const Matrix<T,3>& x):Matrix_base<T>(x),L1(x.L1),L2(x.L2),L3(x.L3){}
    //Move constructor:
    Matrix(Matrix<T,3>&& x):Matrix_base<T>(std::move(x)),L1(x.L1),L2(x.L2),L3(x.L3){} 

    /*********************/
    /*PART II: DESTRUCTOR*/
    /*********************/
    ~Matrix() {} // destruction in Matrix_base will be call automatically

    /***************************/
    /*PART III: MEMBER FUNCTION*/
    /***************************/
    void range_check(size_t i,size_t j,size_t k) const  
    {if( (i>=L1)||(j>=L2)||(k>=L3) ) throw std::invalid_argument("exceed the range of 3D array");}
    void lenth_check() const {if((L1*L2*L3)!=this->L) throw std::invalid_argument("size not consistent in 3D array");}
    // subscripting, the matrix is column arranged:
    T operator ()(size_t i,size_t j,size_t k) const  {/*range_check(i,j,k);*/ return this->base_array[i+j*L1+k*L1*L2];}
    T& operator()(size_t i,size_t j,size_t k)        {/*range_check(i,j,k);*/ return this->base_array[i+j*L1+k*L1*L2];}
    Matrix<T,2> operator[](size_t i)
    {
     if(i>=L3) throw std::invalid_argument("i exceed the L3 range of 3D array");
     Matrix<T,2> A; A.L_f()=L1*L2; A.owns=false; A.L1=L1; A.L2=L2;
     A.base_array=this->base_array; A.base_array+=(L1*L2*i);
     return A;
    }

    /**************************/
    /*PART IV: INSIDE OVERLOAD*/
    /**************************/
   //copy-assigment array to Matrix_base 
    Matrix<T,3>& operator  = (const Matrix_base<T>& B) {this->copy(B);lenth_check();return *this;}
    //move-assigment array to Matrix_base
    Matrix<T,3>& operator  = (Matrix_base<T>&& B)      
    {if(B.owns && this->owns) this->swap(B); else this->copy(B); lenth_check(); return *this;}

    //copy-assigment array to Matrix<T,3>
    Matrix<T,3>& operator  = (const Matrix<T,3>& B)    {L1=B.L1;L2=B.L2;L3=B.L3;this->copy(B);return *this;}
    //move-assigment array to Matrix<T,3>
    Matrix<T,3>& operator  = (Matrix<T,3>&& B)         
    {L1=B.L1;L2=B.L2;L3=B.L3; if(B.owns && this->owns) this->swap(B); else this->copy(B); return *this;}

    // += array A=A+B
    Matrix<T,3>& operator += (const Matrix<T,3>& B) {this->add_equal(B);return *this;}
    // -= array A=A-B
    Matrix<T,3>& operator -= (const Matrix<T,3>& B) {this->min_equal(B);return *this;}
    // *= array A=A*B
    Matrix<T,3>& operator *= (const Matrix<T,3>& B) {this->tim_equal(B);return *this;}
    // /= array A=A/B
    Matrix<T,3>& operator /= (const Matrix<T,3>& B) {this->div_equal(B);return *this;}


    // copy-assigment scaler 
    Matrix<T,3>& operator  = (const T & B) {this->copy(B);return *this;}
    // += scaler A=A+B
    Matrix<T,3>& operator += (const T & B) {this->add_equal(B);return *this;}
    // -= scaler A=A-B
    Matrix<T,3>& operator -= (const T & B) {this->min_equal(B);return *this;}
    // *= scaler A=A*B
    Matrix<T,3>& operator *= (const T & B) {this->tim_equal(B);return *this;}
    // /= scaler A=A/B
    Matrix<T,3>& operator /= (const T & B) {this->div_equal(B);return *this;}

 }; //end class matrix<T,3>


 /***************************/
 /*PART IV: OUTSIDE OVERLOAD*/
 /***************************/
 //for minus sign:  -(array) 
 template <class T>
 Matrix<T,3> operator -(const Matrix<T,3>& B) {Matrix<T,3> C(B.L1,B.L2,B.L3);C=B.min_sign();return C;}

 //for conj
 template <class T>
 Matrix<std::complex<T>,3> conj(const Matrix<std::complex<T>,3>& A) {Matrix<std::complex<T>,3> B(A.L1,A.L2,A.L3);B=A.conj();return B;}

 //for cout
 template <class T>
 std::ostream& operator<< (std::ostream &out, const Matrix<T,3> &object)
 {
  out<<std::boolalpha;
  out<<std::scientific;
  out<<"Own: "<<object.owns<<"\n";
  out<<"Size: "<<object.L1<<" "<<object.L2<<" "<<object.L3<<"\n";
  for (size_t i = 0; i<object.L3; ++i) 
  {
   for (size_t j = 0; j<object.L1; ++j)
   {
    for (size_t k = 0; k<object.L2; ++k)  out<<" "<<object(j,k,i);
    out<<"\n";   
   }
   out<<"\n";
  }
  return out;
 }

} //end namespace matrix_hao_lib

#endif
