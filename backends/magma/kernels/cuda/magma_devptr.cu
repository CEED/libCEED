#include <ceed/ceed.h>
#include <magma.h>
#include <cuda.h>
#include <cuda_runtime.h>

/***************************************************************************//**
    Determines whether a pointer points to CPU or GPU memory.

    This is very similar to magma_is_devptr, except that it does not check for 
    unified addressing support. 
    @param[in] A    pointer to test

    @return  1:  if A is a device pointer (definitely),
    @return  0:  if A is a host   pointer (definitely or inferred from error),
    @return -1:  if unknown.

    @ingroup magma_util
*******************************************************************************/
extern "C" magma_int_t
magma_isdevptr( const void* A )
{
    cudaError_t err;
    cudaPointerAttributes attr;
    int dev;  // must be int
    err = cudaGetDevice( &dev );
    if ( ! err ) {
        err = cudaPointerGetAttributes( &attr, A);
        if ( ! err ) {
            // definitely know type
            #if CUDA_VERSION >= 10000
            return (attr.type == cudaMemoryTypeDevice);
            #else
            return (attr.memoryType == cudaMemoryTypeDevice);
            #endif
        }
        else if ( err == cudaErrorInvalidValue ) {
            // clear error; see http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=529
            cudaGetLastError();
            // infer as host pointer
            return 0;
        }
    }
    // clear error
    cudaGetLastError();
    // unknown, e.g., device doesn't support unified addressing
    return -1;
}
