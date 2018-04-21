#ifndef MAGMABLAS_CHECK_CUDAERROR_H 
#define MAGMABLAS_CHECK_CUDAERROR_H 

#define magma_check_cudaerror() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#endif 
