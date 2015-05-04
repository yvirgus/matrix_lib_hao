#ifndef MATRIX_DEFINE
#define MATRIX_DEFINE

#ifdef USE_MKL
#include "mkl.h"
#define FORTRAN_NAME(x) x
//typedef  long long        BL_INT;
typedef  int              BL_INT;
typedef  float            BL_FLOAT;     
typedef  double           BL_DOUBLE;    
typedef  MKL_Complex8     BL_COMPLEX8;  
typedef  MKL_Complex16    BL_COMPLEX16; 
#endif


#ifdef USE_ACML
#include "acml.h" // change acml.h: %s/complex/ccomplex/g; Functions: cgemm_ dgemm_ sgemm_ zgemm_ zgetrs_ zheev_
#define FORTRAN_NAME(x) x##_
typedef  int              BL_INT;
typedef  float            BL_FLOAT;
typedef  double           BL_DOUBLE;
typedef  ccomplex         BL_COMPLEX8;
typedef  doubleccomplex   BL_COMPLEX16;
#endif


#endif

