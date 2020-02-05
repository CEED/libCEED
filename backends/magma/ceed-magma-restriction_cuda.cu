#include <ceed.h>

#include "atomics.cuh"        
#include "magma_check_cudaerror.h"   

#define  MAX_TB_XDIM      2147483647
#define  MAX_THREADS_PTB  1024      
#define  OUR_THREADS_PTB  512
