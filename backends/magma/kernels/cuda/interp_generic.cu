// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <cuda.h>    // for CUDA_VERSION
#include "../common/interp.h"

#define ipow(a,b) ( (magma_int_t)(std::pow( (float)(a), (float)(b) ) ) )

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int P, int Q>
static magma_int_t 
interp_generic_kernel_driver( 
    magma_int_t dim, magma_int_t ncomp,  
    const T *dT, magma_trans_t transT,
    const T *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          T *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );

    magma_int_t shmem_max, nthreads_max;
    magma_int_t pre = ipow(P, dim-1); //ncomp*CeedIntPow(P, dim-1);
    magma_int_t post = 1; 
    // ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1);
    // originally the exponent is (dim-1), but we use dim because 
    // we have to read the original u in shared memory
    // the original implementation access u directly
    magma_int_t tmp_size = ipow(max(P,Q), dim); //ncomp * Q * ipow(max(P,Q), dim); 
    magma_int_t shmem = P * Q * sizeof(T);
    shmem += 2 * tmp_size * sizeof(T); 

    magma_int_t nthreads = max(P, ipow(Q, dim-1) ); 
    nthreads = magma_roundup( nthreads, Q ); // nthreads must be multiple of Q

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(interp_generic_kernel<T, P, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( nthreads > nthreads_max || shmem > shmem_max ) {
        return 1;
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(nelem, ncomp, 1);
        interp_generic_kernel<T, P, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        ( dim, ncomp, pre, post, tmp_size, dT, transT, 
          dU, estrdU, cstrdU, 
          dV, estrdV, cstrdV );
          return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_interp_generic_q( 
    magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const CeedScalar *dT, magma_trans_t transT,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 1>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  2: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 2>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  3: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 3>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  4: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 4>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  5: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 5>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  6: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 6>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  7: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 7>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  8: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 8>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  9: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P, 9>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 10: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,10>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 11: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,11>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 12: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,12>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 13: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,13>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 14: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,14>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 15: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,15>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 16: 
          launch_failed = interp_generic_kernel_driver<CeedScalar, P,16>
          (dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
magma_int_t 
static magma_interp_generic_q_p( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const CeedScalar *dT, magma_trans_t transT,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (P) {
        case  1: 
          launch_failed = magma_interp_generic_q< 1>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_interp_generic_q< 2>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_interp_generic_q< 3>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_interp_generic_q< 4>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_interp_generic_q< 5>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_interp_generic_q< 6>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_interp_generic_q< 7>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_interp_generic_q< 8>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_interp_generic_q< 9>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_interp_generic_q<10>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 11: 
          launch_failed = magma_interp_generic_q<11>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 12: 
          launch_failed = magma_interp_generic_q<12>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 13: 
          launch_failed = magma_interp_generic_q<13>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 14: 
          launch_failed = magma_interp_generic_q<14>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 15: 
          launch_failed = magma_interp_generic_q<15>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 16: 
          launch_failed = magma_interp_generic_q<16>
          (Q, dim, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_interp_generic( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const CeedScalar *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;

    launch_failed = magma_interp_generic_q_p(
                        P, Q, dim, ncomp, 
                        dT, transT, 
                        dU, estrdU, cstrdU, 
                        dV, estrdV, cstrdV, 
                        nelem, queue);

    return launch_failed;
}
