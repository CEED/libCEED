// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../common/weight.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static magma_int_t 
magma_weight_generic_kernel_driver( 
                magma_int_t dim,   
                const T *dqweight1d, 
                T *dV, magma_int_t vstride, 
                magma_int_t batchCount, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );

    magma_int_t shmem_max, nthreads_max;

    magma_int_t pre_org  = CeedIntPow(Q, dim-0-1);
    magma_int_t post_org = CeedIntPow(Q, 0);

    magma_int_t vsize = CeedIntPow(Q, dim); 
    magma_int_t shmem = vsize * sizeof(T);  // holds dV in shared memory
    shmem += (Q * sizeof(T)); // holds qweight1d

    magma_int_t nthreads = CeedIntPow(Q, dim-1);

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_weight_generic_kernel<T, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000
 
    if ( nthreads > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(batchCount, 1, 1);
        magma_weight_generic_kernel<T, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        ( dim, pre_org, post_org, dqweight1d, dV, vstride );
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;        
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_weight_generic_q( 
    magma_int_t Q, magma_int_t dim, 
    const CeedScalar *dqweight1d, 
    CeedScalar *dV, magma_int_t vstride, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 1>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  2: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 2>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  3: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 3>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  4: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 4>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  5: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 5>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  6: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 6>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  7: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 7>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  8: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 8>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case  9: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar, 9>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        case 10: 
          launch_failed = magma_weight_generic_kernel_driver<CeedScalar,10>
          (dim, dqweight1d, dV, vstride, batchCount, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_weight_generic( 
    magma_int_t Q, magma_int_t dim, 
    const CeedScalar *dqweight1d, 
    CeedScalar *dV, magma_int_t vstride, 
    magma_int_t batchCount, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    launch_failed = magma_weight_generic_q(Q, dim, dqweight1d, dV, vstride, batchCount, queue);
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
// NonTensor weight function
extern "C" void 
magma_weight_nontensor(magma_int_t grid, magma_int_t threads, magma_int_t nelem, magma_int_t Q, 
             CeedScalar *dqweight, CeedScalar *dv, magma_queue_t queue)
{
    magma_weight_nontensor_kernel<<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>(nelem, Q, dqweight, dv);
}
