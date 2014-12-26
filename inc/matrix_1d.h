#ifndef MATRIX_HAO_1D
#define MATRIX_HAO_1D

namespace matrix_hao_lib
{
 template<class T> class Matrix<T,1>:public Matrix_base<T>
 {
  public:
    size_t L1;

    /*********************/
    /*PART I: CONSTRUCTOR*/
    /*********************/

    //Void constructor:
    Matrix(): Matrix_base<T>(),L1(0){}
    //N constructor: vector is not initialized
    Matrix(size_t n):Matrix_base<T>(n),L1(n){}
    //List constructor:
    Matrix(size_t n,const std::initializer_list <T> &args):Matrix_base<T>(args),L1(n){lenth_check();}
    //Copy constructor:
    Matrix(const Matrix<T,1>& x):Matrix_base<T>(x),L1(x.L1){}
    //Move constructor:
    Matrix(Matrix<T,1>&& x):Matrix_base<T>(std::move(x)),L1(x.L1){} 

    /*********************/
    /*PART II: DESTRUCTOR*/
    /*********************/
    ~Matrix() {} // destruction in Matrix_base will be call automatically

    /***************************/
    /*PART III: MEMBER FUNCTION*/
    /***************************/
    void range_check(size_t i) const  {if (i>=L1) throw std::invalid_argument("exceed the range of 1D array");}
    void lenth_check() const {if(L1!=this->L) throw std::invalid_argument("size not consistent in 1D array");}
    // subscripting:
    T operator ()(size_t i) const  {range_check(i); return this->base_array[i];}
    T& operator()(size_t i)        {range_check(i); return this->base_array[i];}
    T operator [](size_t i) const  {range_check(i); return this->base_array[i];}
    T& operator[](size_t i)        {range_check(i); return this->base_array[i];}


    /**************************/
    /*PART IV: INSIDE OVERLOAD*/
    /**************************/
    //copy-assigment array to Matrix_base 
    Matrix<T,1>& operator  = (const Matrix_base<T>& B) {this->copy(B);lenth_check();return *this;}
    //move-assigment array to Matrix_base
    Matrix<T,1>& operator  = (Matrix_base<T>&& B)   
    {if(B.owns && this->owns) this->swap(B); else this->copy(B); lenth_check(); return *this;}

    //copy-assigment array to Matrix<T,1>
    Matrix<T,1>& operator  = (const Matrix<T,1>& B)    {L1=B.L1;this->copy(B);return *this;}
    //move-assigment array to Matrix<T,1>
    Matrix<T,1>& operator  = (Matrix<T,1>&& B) 
    {L1=B.L1; if(B.owns && this->owns) this->swap(B); else this->copy(B); return *this;}


    // += array A=A+B
    Matrix<T,1>& operator += (const Matrix<T,1>& B) {this->add_equal(B);return *this;}
    // -= array A=A-B
    Matrix<T,1>& operator -= (const Matrix<T,1>& B) {this->min_equal(B);return *this;}
    // *= array A=A*B
    Matrix<T,1>& operator *= (const Matrix<T,1>& B) {this->tim_equal(B);return *this;}
    // /= array A=A/B
    Matrix<T,1>& operator /= (const Matrix<T,1>& B) {this->div_equal(B);return *this;}


    // copy-assigment scaler 
    Matrix<T,1>& operator  = (const T & B) {this->copy(B);return *this;}
    // += scaler A=A+B
    Matrix<T,1>& operator += (const T & B) {this->add_equal(B);return *this;}
    // -= scaler A=A-B
    Matrix<T,1>& operator -= (const T & B) {this->min_equal(B);return *this;}
    // *= scaler A=A*B
    Matrix<T,1>& operator *= (const T & B) {this->tim_equal(B);return *this;}
    // /= scaler A=A/B
    Matrix<T,1>& operator /= (const T & B) {this->div_equal(B);return *this;}

 }; //end class matrix<T,1>


 /***************************/
 /*PART IV: OUTSIDE OVERLOAD*/
 /***************************/
 //for minus sign:  -(array) 
 template <class T>
 Matrix<T,1> operator -(const Matrix<T,1>& B) {Matrix<T,1> C(B.L1);C=B.min_sign();return C;}

 //for conj
 template <class T>
 Matrix<std::complex<T>,1> conj(const Matrix<std::complex<T>,1>& A) {Matrix<std::complex<T>,1> B(A.L1);B=A.conj();return B;}

 //for exp
 template <class T>
 Matrix<T,1> exp(const Matrix<T,1>& A) {Matrix<T,1> B(A.L1);B=A.exp();return B;}

 //for cout
 template <class T>
 std::ostream& operator<< (std::ostream &out, const Matrix<T,1> &object)
 {
  out<<std::boolalpha;
  out<<std::scientific;
  out<<"Own: "<<object.owns<<"\n";
  out<<"Size: "<<object.L1<<"\n";
  for (size_t i = 0; i<object.L_f(); ++i) out<<" "<<object.base_array[i];
  out<<"\n";
  return out;
 }

} //end namespace matrix_hao_lib

#endif
