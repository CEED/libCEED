// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed.h>
#include <cuda.h>    // for CUDA_VERSION
#include <magma_v2.h>
#include "magma_common_device.cuh"
#include "grad_device.cuh"

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ CeedScalar shared_data[];
template<typename T, int DIM, int NCOMP, int P, int Q>
static __global__ void
magma_grad_1d_kernel(  
    const T *dTgrad, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, 
          T *dV, const int estrdV, const int cstrdV, const int nelem)
{
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T* sU[NCOMP];
    T* sV[NCOMP];

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    T* sT = (T*)(shared_data);
    T* sW = sT + P*Q;
    sU[0] = sW + ty * NCOMP * (P + Q);
    sV[0] = sU[0] + (NCOMP * 1 * P);
    for(int icomp = 1; icomp < NCOMP; icomp++) {
        sU[icomp] = sU[icomp-1] + (1 * P);
        sV[icomp] = sV[icomp-1] + (1 * Q);
    }

    // read T
    if (ty == 0) {
        dread_T_gm2sm<P, Q>(tx, transT, dTgrad, sT);
    }

    // read U
    read_1d<T, P, NCOMP>(dU, cstrdU, sU, tx);

    // read V if transT is magmaTrans
    if (transT == MagmaTrans) {
        read_1d<T, Q, NCOMP>(dV, cstrdV, sV, tx);
    }

    __syncthreads();
    magma_grad_1d_device<T, DIM, NCOMP, P, Q>(sT, transT, sU, sV, tx);
    __syncthreads();

    // write V
    write_1d<T, Q, NCOMP>(sV, dV, cstrdV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q>
static magma_int_t 
magma_grad_1d_kernel_driver(  
                const T *dTgrad, magma_trans_t transT,
                const T *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      T *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;

    magma_int_t nthreads = max(P, Q); 
    magma_int_t ntcol = (maxthreads < nthreads) ? 1 : (maxthreads / nthreads);
    magma_int_t shmem  = 0;
    shmem += sizeof(T) * ntcol * (NCOMP * (1*P + 1*Q)); 
    shmem += sizeof(T) * (P*Q);    

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_grad_1d_kernel<T, 1, NCOMP, P, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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
        magma_grad_1d_kernel<T, 1, NCOMP, P, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        (dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem);
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
magma_grad_1d_ncomp(
                magma_int_t ncomp,
                const CeedScalar *dTgrad, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (ncomp) {
        case 1: 
          launch_failed = magma_grad_1d_kernel_driver<CeedScalar,1,P,Q>
          (dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case 2: 
          launch_failed = magma_grad_1d_kernel_driver<CeedScalar,2,P,Q>
          (dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case 3: 
          launch_failed = magma_grad_1d_kernel_driver<CeedScalar,3,P,Q>
          (dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_grad_1d_ncomp_q(
                magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dTgrad, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_grad_1d_ncomp<P, 1>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  2: 
          launch_failed = magma_grad_1d_ncomp<P, 2>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  3: 
          launch_failed = magma_grad_1d_ncomp<P, 3>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  4: 
          launch_failed = magma_grad_1d_ncomp<P, 4>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  5: 
          launch_failed = magma_grad_1d_ncomp<P, 5>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  6: 
          launch_failed = magma_grad_1d_ncomp<P, 6>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  7: 
          launch_failed = magma_grad_1d_ncomp<P, 7>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  8: 
          launch_failed = magma_grad_1d_ncomp<P, 8>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  9: 
          launch_failed = magma_grad_1d_ncomp<P, 9>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case 10: 
          launch_failed = magma_grad_1d_ncomp<P,10>
          (ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}


//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_grad_1d_ncomp_q_p(
                magma_int_t P, magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dTgrad, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (P) {
        case  1: 
          launch_failed = magma_grad_1d_ncomp_q< 1>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  2: 
          launch_failed = magma_grad_1d_ncomp_q< 2>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  3: 
          launch_failed = magma_grad_1d_ncomp_q< 3>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  4: 
          launch_failed = magma_grad_1d_ncomp_q< 4>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  5: 
          launch_failed = magma_grad_1d_ncomp_q< 5>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  6: 
          launch_failed = magma_grad_1d_ncomp_q< 6>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  7: 
          launch_failed = magma_grad_1d_ncomp_q< 7>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  8: 
          launch_failed = magma_grad_1d_ncomp_q< 8>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case  9: 
          launch_failed = magma_grad_1d_ncomp_q< 9>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        case 10: 
          launch_failed = magma_grad_1d_ncomp_q<10>
          (Q, ncomp, dTgrad, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}



//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_grad_1d( 
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,  
    const CeedScalar *dTinterp, const CeedScalar *dTgrad, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_grad_1d_ncomp_q_p(
                        P, Q, ncomp, 
                        dTgrad, transT, 
                        dU, estrdU, cstrdU, 
                        dV, estrdV, cstrdV, 
                        nelem, maxthreads, queue);

    return launch_failed;
}
