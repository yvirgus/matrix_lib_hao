#ifndef MATRIX_HAO_BASE
#define MATRIX_HAO_BASE

#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <typeinfo>
#include <stdexcept>


namespace matrix_hao_lib
{

 //define a general class template
 template<class T = double, int D = 1> class Matrix
 {
  private:
     Matrix(); //this will not be complied
 };


 //one dimension array for the array for the matrix
 template<class T> class Matrix_base
 {
  protected:
     size_t L;
  public:
     T* base_array;
     bool owns;       
     char type;
 
     /*********************/
     /*PART I: CONSTRUCTOR*/
     /*********************/
 
     //Void constructor:
     Matrix_base(void): L(0),base_array(nullptr),owns(true) {this->settype();}
     //N constructor: T[n], if we want to initial to zero: use T[n]()
     Matrix_base(size_t n):L(n),base_array(new T[n]),owns(true) {this->settype();}
     //List constructor:
     Matrix_base(const std::initializer_list <T> &args)
     {
         L=args.size();
         base_array=new T[args.size()];
         owns=true;
         this->settype();
         std::copy(args.begin(),args.begin()+args.size(),base_array);
     }
     //Copy constructor:
     Matrix_base(const Matrix_base<T>& x):L(x.L_f()),owns(x.owns),type(x.type)
     {
         if(x.owns) {base_array=new T[x.L_f()];std::copy(x.base_array,x.base_array+x.L_f(),base_array);}
         else       {base_array=x.base_array;}
     }
     //Move constructor:
     Matrix_base(Matrix_base<T>&& x): L(x.L_f()),base_array(x.base_array),owns(x.owns),type(x.type)
     {
         if(x.owns) x.base_array=nullptr;
     } 
    
     
     /*********************/
     /*PART II: DESTRUCTOR*/
     /*********************/
     //Destructor
     ~Matrix_base() {if(base_array&&owns) delete[] base_array;}
 
 
 
     /***************************/
     /*PART III: MEMBER FUNCTION*/
     /***************************/
     size_t L_f() const  {return L;}
     size_t & L_f()      {return L;}
     void point(const size_t PL, T* P) 
     {if(base_array&&owns) delete[] base_array; L=PL;         base_array=P;         owns=false;}
     void point(std::vector<T>& vec) 
     {if(base_array&&owns) delete[] base_array; L=vec.size(); base_array=vec.data();owns=false;}
 
     //copy this=B array
     inline void copy(const Matrix_base<T> & B)
     {
         if(&B!=this)
         {
             if(L!=B.L_f())
             {
                 if(base_array&&owns) delete[] base_array;
                 base_array=new T[B.L_f()];
                 L=B.L_f();
                 owns=true;
                 //type will always be the same due to the template
             }
             std::copy(B.base_array,B.base_array+B.L_f(),base_array);
         }
     }
 
     //swap address of this and B array
     inline void swap(Matrix_base<T> & B)
     {
         T* base_array_tmp=base_array;base_array=B.base_array;B.base_array=base_array_tmp;
         if(L!=B.L_f()) {size_t L_tmp=L;L=B.L_f();B.L_f()=L_tmp;}
         if(owns!=B.owns) {owns=!owns;B.owns=!B.owns;}
         //type will always be the same due to the template
     }
 
 
     //conj
     Matrix_base<T> conj() const
     {Matrix_base<T> A(L);for(size_t i=0; i<L; ++i){A.base_array[i]=std::conj(base_array[i]);}; return A;}
 
     //exp
     Matrix_base<T> exp() const
     {Matrix_base<T> A(L);for(size_t i=0; i<L; ++i){A.base_array[i]=std::exp(base_array[i]);}; return A;}
 
     //minus sign -SELF
     Matrix_base<T> min_sign() const
     {Matrix_base<T> A(L);for(size_t i=0; i<L; ++i){A.base_array[i]=-base_array[i];}; return A;}
 
 
     //add_equal array SELF=SELF+B
     inline void add_equal(const Matrix_base<T> & B)
     {
         if(L!=B.L_f()) {throw std::invalid_argument("add equal do not have the same L in Matrix_base::add_equal!");}
         for(size_t i=0; i<B.L_f(); ++i){ base_array[i] += B.base_array[i];}
     }
 
     //min_equal array SELF=SELF-B
     inline void min_equal(const Matrix_base<T> & B)
     {
         if(L!=B.L_f()) {throw std::invalid_argument("min equal do not have the same L in Matrix_base::min_equal!");}
         for(size_t i=0; i<B.L_f(); ++i){ base_array[i] -= B.base_array[i];}
     }
 
     //min_add_equal array: SELF=-SELF+B
     inline void min_add_equal(const Matrix_base<T> & B)
     {
         if(L!=B.L_f()) {throw std::invalid_argument("min_add equal do not have the same L in Matrix_base::min_add_equal!");}
         for(size_t i=0; i<B.L_f(); ++i){ base_array[i]=B.base_array[i]-base_array[i];}
     }
 
 
     //tim_equal array SELF=SELF*B
     inline void tim_equal(const Matrix_base<T> & B)
     {
         if(L!=B.L_f()) {throw std::invalid_argument("tim equal do not have the same L in Matrix_base::tim_equal!");}
         for(size_t i=0; i<B.L_f(); ++i){ base_array[i] *= B.base_array[i];}
     }
 
     //div_equal array SELF=SELF/B
     inline void div_equal(const Matrix_base<T> & B)
     {
         if(L!=B.L_f()) {throw std::invalid_argument("div equal do not have the same L in Matrix_base::div_equal!");}
         for(size_t i=0; i<B.L_f(); ++i){ base_array[i] /= B.base_array[i];}
     }
 
     //inv_div_equal array SELF=(1/SELF)*B
     inline void inv_div_equal(const Matrix_base<T> & B)
     {
         if(L!=B.L_f()) {throw std::invalid_argument("inv div equal do not have the same L in Matrix_base::inv_div_equal!");}
         for(size_t i=0; i<B.L_f(); ++i){ base_array[i] = B.base_array[i]/base_array[i];}
     } 
 
 
     //copy this=B scaler
     inline void copy(const T & B)          {for(size_t i=0; i<L; ++i) base_array[i] =B;}
     //add_equal scaler SELF=SELF+B
     inline void add_equal(const T & B)     {for(size_t i=0; i<L; ++i) base_array[i] += B;}
     //min_equal scaler SELF=SELF-B
     inline void min_equal(const T & B)     {for(size_t i=0; i<L; ++i) base_array[i] -= B;}
     //min_add_equal scaler SELF=-SELF+B
     inline void min_add_equal(const T & B) {for(size_t i=0; i<L; ++i) base_array[i]=B-base_array[i];}
     //tim_equal scaler SELF=SELF*B
     inline void tim_equal(const T & B)     {for(size_t i=0; i<L; ++i) base_array[i] *= B;}
     //div_equal scaler SELF=SELF/B
     inline void div_equal(const T & B)     {for(size_t i=0; i<L; ++i) base_array[i] /= B;}
     //inv_div_equal scaler SELF=(1/SELF)B
     inline void inv_div_equal(const T & B) {for(size_t i=0; i<L; ++i) base_array[i]=B/base_array[i];}
 
 
     /**************************/
     /*PART IV: INSIDE OVERLOAD*/
     /**************************/
     //copy-assigment array   
     Matrix_base<T>& operator  = (const Matrix_base<T> & B) {copy(B);return *this;}
     //move-assigment array 
     Matrix_base<T>& operator  = (Matrix_base<T>&& B) {if(B.owns && this->owns) swap(B); else copy(B); return *this;}
 
     //+= array A=A+B
     Matrix_base<T>& operator += (const Matrix_base<T> & B) {add_equal(B);return *this;}
     //-= array A=A-B
     Matrix_base<T>& operator -= (const Matrix_base<T> & B) {min_equal(B);return *this;}
     //*= array A=A*B
     Matrix_base<T>& operator *= (const Matrix_base<T> & B) {tim_equal(B);return *this;}
     ///= array A=A/B
     Matrix_base<T>& operator /= (const Matrix_base<T> & B) {div_equal(B);return *this;}
 
     //copy-assigment scaler 
     Matrix_base<T>& operator  = (const T & B) {copy(B);return *this;}
     //+= scaler A=A+B
     Matrix_base<T>& operator += (const T & B) {add_equal(B);return *this;}
     //-= scaler A=A-B
     Matrix_base<T>& operator -= (const T & B) {min_equal(B);return *this;}
     //*= scaler A=A*B
     Matrix_base<T>& operator *= (const T & B) {tim_equal(B);return *this;}
     ///= scaler A=A/B
     Matrix_base<T>& operator /= (const T & B) {div_equal(B);return *this;}


  private:
     inline void settype()
     {
         if(typeid(T)==typeid(float))  type='s';
         else if(typeid(T)==typeid(double)) type='d';
         else if(typeid(T)==typeid(std::complex<float>))  type='c';
         else if(typeid(T)==typeid(std::complex<double>)) type='z';
         else type='?';
     }
 }; //end class Matrix_base



 /***************************/
 /*PART IV: OUTSIDE OVERLOAD*/
 /***************************/
 //for minus sign:  -(array) 
 template <class T>
 Matrix_base<T> operator -(const Matrix_base<T> & B) {return B.min_sign();}   

 //for add: (array+array)
 template <class T>
 Matrix_base<T> operator + (const Matrix_base<T>& A,const Matrix_base<T>& B) {Matrix_base<T> C;C=A; C+=B; return C;} 
 template <class T>
 Matrix_base<T> operator + (const Matrix_base<T>& A, Matrix_base<T>&& B)     {Matrix_base<T> C;C=std::move(B); C+=A; return C;} 
 template <class T>
 Matrix_base<T> operator + (Matrix_base<T>&& A,const Matrix_base<T>& B)      {Matrix_base<T> C;C=std::move(A); C+=B; return C;} 
 template <class T>
 Matrix_base<T> operator + (Matrix_base<T>&& A,Matrix_base<T>&& B)           {Matrix_base<T> C;C=std::move(A); C+=B; return C;} 

 //for minus:(array-array)
 template <class T>
 Matrix_base<T> operator - (const Matrix_base<T>& A,const Matrix_base<T>& B) {Matrix_base<T> C;C=A; C-=B; return C;}
 template <class T>
 Matrix_base<T> operator - (const Matrix_base<T>& A, Matrix_base<T>&& B)
                                                               {Matrix_base<T> C;C=std::move(B); C.min_add_equal(A); return C;} 
 template <class T>
 Matrix_base<T> operator - (Matrix_base<T>&& A,const Matrix_base<T>& B)      {Matrix_base<T> C;C=std::move(A); C-=B; return C;} 
 template <class T>
 Matrix_base<T> operator - (Matrix_base<T>&& A,Matrix_base<T>&& B)           {Matrix_base<T> C;C=std::move(A); C-=B; return C;} 

 //for time: (array*array)
 template <class T>
 Matrix_base<T> operator * (const Matrix_base<T>& A,const Matrix_base<T>& B) {Matrix_base<T> C;C=A; C*=B; return C;}
 template <class T>
 Matrix_base<T> operator * (const Matrix_base<T>& A, Matrix_base<T>&& B)     {Matrix_base<T> C;C=std::move(B); C*=A; return C;}
 template <class T>
 Matrix_base<T> operator * (Matrix_base<T>&& A,const Matrix_base<T>& B)      {Matrix_base<T> C;C=std::move(A); C*=B; return C;}
 template <class T>
 Matrix_base<T> operator * (Matrix_base<T>&& A,Matrix_base<T>&& B)           {Matrix_base<T> C;C=std::move(A); C*=B; return C;}


 //for div:(array/array)
 template <class T>
 Matrix_base<T> operator / (const Matrix_base<T>& A,const Matrix_base<T>& B) {Matrix_base<T> C;C=A; C/=B; return C;}
 template <class T>
 Matrix_base<T> operator / (const Matrix_base<T>& A, Matrix_base<T>&& B)
                                                               {Matrix_base<T> C;C=std::move(B); C.inv_div_equal(A); return C;}
 template <class T>
 Matrix_base<T> operator / (Matrix_base<T>&& A,const Matrix_base<T>& B)      {Matrix_base<T> C;C=std::move(A); C/=B; return C;}
 template <class T>
 Matrix_base<T> operator / (Matrix_base<T>&& A,Matrix_base<T>&& B)           {Matrix_base<T> C;C=std::move(A); C/=B; return C;}


 //for add: (array+scaler)
 template <class T>
 Matrix_base<T> operator + (const Matrix_base<T>& A,const T & B) {Matrix_base<T> C;C=A; C+=B; return C;}
 template <class T>
 Matrix_base<T> operator + (Matrix_base<T>&& A, const T & B)     {Matrix_base<T> C;C=std::move(A); C+=B; return C;}
 template <class T>
 Matrix_base<T> operator + (const T & B,const Matrix_base<T>& A) {Matrix_base<T> C;C=A; C+=B; return C;}
 template <class T>
 Matrix_base<T> operator + (const T & B,Matrix_base<T>&& A)      {Matrix_base<T> C;C=std::move(A); C+=B; return C;}

 //for minus: (array-scaler)
 template <class T>
 Matrix_base<T> operator - (const Matrix_base<T>& A,const T & B) {Matrix_base<T> C;C=A; C-=B; return C;}
 template <class T>
 Matrix_base<T> operator - (Matrix_base<T>&& A, const T & B)     {Matrix_base<T> C;C=std::move(A); C-=B; return C;}
 template <class T>
 Matrix_base<T> operator - (const T & B,const Matrix_base<T>& A) {Matrix_base<T> C;C=A; C.min_add_equal(B);return C;}
 template <class T>
 Matrix_base<T> operator - (const T & B,Matrix_base<T>&& A)      {Matrix_base<T> C;C=std::move(A);C.min_add_equal(B);return C;}


 //for time: (array*scaler)
 template <class T>
 Matrix_base<T> operator * (const Matrix_base<T>& A,const T & B) {Matrix_base<T> C;C=A; C*=B; return C;}
 template <class T>
 Matrix_base<T> operator * (Matrix_base<T>&& A, const T & B)     {Matrix_base<T> C;C=std::move(A); C*=B; return C;}
 template <class T>
 Matrix_base<T> operator * (const T & B,const Matrix_base<T>& A) {Matrix_base<T> C;C=A; C*=B; return C;}
 template <class T>
 Matrix_base<T> operator * (const T & B,Matrix_base<T>&& A)      {Matrix_base<T> C;C=std::move(A); C*=B; return C;}

 //for div: (array/scaler)
 template <class T>
 Matrix_base<T> operator / (const Matrix_base<T>& A,const T & B) {Matrix_base<T> C;C=A; C/=B; return C;}
 template <class T>
 Matrix_base<T> operator / (Matrix_base<T>&& A, const T & B)     {Matrix_base<T> C;C=std::move(A); C/=B; return C;}
 template <class T>
 Matrix_base<T> operator / (const T & B,const Matrix_base<T>& A) {Matrix_base<T> C;C=A; C.inv_div_equal(B);return C;}
 template <class T>
 Matrix_base<T> operator / (const T & B,Matrix_base<T>&& A)      {Matrix_base<T> C;C=std::move(A);C.inv_div_equal(B);return C;}

 //for cout
 template <class T>
 std::ostream& operator<< (std::ostream &out, const Matrix_base<T>& object)
 {
     out<<std::boolalpha;
     out<<std::scientific;
     out<<"Own: "<<object.owns<<"\n";
     out<<"Base size: "<<object.L_f()<<"\n";
     for (size_t i = 0; i<object.L_f(); ++i) out<<" "<<object.base_array[i];
     out<<"\n";
     return out;
 }


} //end namespace matrix_hao_lib

#endif
