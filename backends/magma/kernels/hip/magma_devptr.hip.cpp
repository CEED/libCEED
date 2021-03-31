#include <ceed/ceed.h>
#include <magma_v2.h>
#include <hip/hip_runtime.h>

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
    hipError_t err;
    hipPointerAttribute_t attr;
    int dev;  // must be int
    err = hipGetDevice( &dev );
    if ( ! err ) {
        err = hipPointerGetAttributes( &attr, A);
        if ( ! err ) {
            // definitely know type
            return (attr.memoryType == hipMemoryTypeDevice);
        }
        else if ( err == hipErrorInvalidValue ) {
            // clear error; see http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=529
            hipGetLastError();
            // infer as host pointer
            return 0;
        }
    }
    // clear error
    hipGetLastError();
    // unknown, e.g., device doesn't support unified addressing
    return -1;
}
