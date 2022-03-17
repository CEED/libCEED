// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <cuda.h>    // for CUDA_VERSION
#include "../common/interp.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q>
static magma_int_t 
magma_interp_2d_kernel_driver(  
                const T *dT, magma_trans_t transT,
                const T *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      T *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;
    const int MAXPQ = maxpq(P,Q);

    magma_int_t nthreads = MAXPQ; 
    magma_int_t ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_2D);
    magma_int_t shmem  = 0;
    shmem += P*Q    *sizeof(T);  // for sT
    shmem += ntcol * ( P*MAXPQ*sizeof(T) );  // for reforming rU we need PxP, and for the intermediate output we need PxQ    

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_interp_2d_kernel<T,NCOMP,P,Q,MAXPQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000
   
    if ( (nthreads*ntcol) > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        magma_int_t nblocks = (nelem + ntcol-1) / ntcol;
        dim3 threads(nthreads, ntcol, 1);
        dim3 grid(nblocks, 1, 1);
        magma_interp_2d_kernel<T,NCOMP,P,Q,MAXPQ><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem);
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
magma_interp_2d_ncomp(
                magma_int_t ncomp,
                const CeedScalar *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (ncomp) {
        case 1: 
          launch_failed = magma_interp_2d_kernel_driver<CeedScalar,1,P,Q>
          (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 2: 
          launch_failed = magma_interp_2d_kernel_driver<CeedScalar,2,P,Q>
          (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 3: 
          launch_failed = magma_interp_2d_kernel_driver<CeedScalar,3,P,Q>
          (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_interp_1d_ncomp_q(
                magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_interp_2d_ncomp<P, 1>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_interp_2d_ncomp<P, 2>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_interp_2d_ncomp<P, 3>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_interp_2d_ncomp<P, 4>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_interp_2d_ncomp<P, 5>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_interp_2d_ncomp<P, 6>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_interp_2d_ncomp<P, 7>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_interp_2d_ncomp<P, 8>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_interp_2d_ncomp<P, 9>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_interp_2d_ncomp<P,10>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_interp_2d_ncomp_q_p(
                magma_int_t P, magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (P) {
        case  1: 
          launch_failed = magma_interp_1d_ncomp_q< 1>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_interp_1d_ncomp_q< 2>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_interp_1d_ncomp_q< 3>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_interp_1d_ncomp_q< 4>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_interp_1d_ncomp_q< 5>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_interp_1d_ncomp_q< 6>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_interp_1d_ncomp_q< 7>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_interp_1d_ncomp_q< 8>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_interp_1d_ncomp_q< 9>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_interp_1d_ncomp_q<10>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}


//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_interp_2d( 
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,  
    const CeedScalar *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_interp_2d_ncomp_q_p(
                        P, Q, ncomp, 
                        dT, transT, 
                        dU, estrdU, cstrdU, 
                        dV, estrdV, cstrdV, 
                        nelem, queue);

    return launch_failed;
}
