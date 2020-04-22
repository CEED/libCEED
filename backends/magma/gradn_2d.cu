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
template<typename T, int NCOMP, int P, int Q, int MAXPQ>
static __global__ void
magma_gradn_2d_kernel(
    const T *dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, const int u_elstride, const int u_compstride, const int u_dimstride,  
          T *dV, const int v_elstride, const int v_compstride, const int v_dimstride)
{
    const int elem_id = blockIdx.x;
    const int tx      = threadIdx.x;
    T rU[1][NCOMP][MAXPQ] = { make_zero<T>() };  // here DIMU = 1, but might be different for a fused operator
    T rV[1][NCOMP][MAXPQ] = { make_zero<T>() };  // here DIMV = 1, but might be different for a fused operator
    T rTmp = make_zero<T>();

    // shift global memory pointers by elem stride
    dU += elem_id * u_elstride;
    dV += elem_id * v_elstride;

    // assign shared memory pointers
    T* sTinterp = (T*)(shared_data);
    T* sTgrad   = sTinterp + P*Q;
    T* sTmp     = sTgrad   + P*Q;

    // read T
    dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
    dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
    __syncthreads();

    // read U (u_dimstride is ignored in notrans) as a batch P of (1xP) vectors
    // vec 0  : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // vec 1  : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // ... 
    // vec P-1: [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // threads collaboratively read vec0 and then vec1 and so on
    // but for the kernel, we want
    // thread 0 to hold all of vec0 in registers, and
    // thread 1 to hold all of vec1 in registers, and and so on
    // so we need to transpose
    for(int icomp = 0; icomp < NCOMP; icomp++) {
        // read from global memory into shared memory
        if(tx < P) {
            for(int i = 0; i < P; i++) {
                sTmp[i*P + tx] = dU[0 * u_dimstride + icomp * u_compstride + i*P + tx];
            }
        }
        __syncthreads();

        if(tx < P) {
            for(int i = 0; i < P; i++) {
                rU[0][icomp][i] = sTmp[tx*P + i];
            }
        }
        __syncthreads();
    }

    // No need to read V ( required only in transposed grad )
    const T beta = make_zero<T>();

    // first call (iDIM = 0, iDIMU = 0, iDIMV = 0)
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, MAXPQ, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    __syncthreads();

    // write V for dim = 0 
    if(tx < Q) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            for(int j = 0; j < Q; j++) {
                dV[0 * v_dimstride + icomp * v_compstride + j*Q + tx] = rV[0][icomp][j];
            }
        }
    }

    // second call (iDIM = 1, iDIMU = 0, iDIMV = 0)
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, MAXPQ, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    __syncthreads();    

    // write V for dim = 1 
    if(tx < Q) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            for(int j = 0; j < Q; j++) {
                dV[1 * v_dimstride + icomp * v_compstride + j*Q + tx] = rV[0][icomp][j];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q>
static magma_int_t 
magma_gradn_2d_kernel_driver(  
                const T *dinterp1d, const T *dgrad1d, magma_trans_t transT,
                const T *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride, 
                      T *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;
    const int MAXPQ = maxpq(P,Q);

    magma_int_t shmem  = 0;
    shmem += 2*P*Q  *sizeof(T);  // for sTinterp and sTgrad
    shmem += P*MAXPQ*sizeof(T);  // for reforming rU we need PxP, and for the intermediate output we need PxQ    
    magma_int_t nthreads = MAXPQ; 

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if(shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_gradn_2d_kernel<T,NCOMP,P,Q,MAXPQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000
   
    if( nthreads > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(nelem, 1, 1);
        // IMPORTANT: we instantiate with DIM=1 instead of DIM=2 because the kernel handles one dimension at a time
        // We should instantiate with DIM >= 1 when we fuse the whole operator, because of the q-function
        magma_gradn_2d_kernel<T,NCOMP,P,Q,MAXPQ><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        (dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride);
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
magma_gradn_2d_ncomp(
                magma_int_t ncomp,
                const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride,
                      CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride,
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(ncomp) {
        case 1: launch_failed = magma_gradn_2d_kernel_driver<CeedScalar,1,P,Q>(dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case 2: launch_failed = magma_gradn_2d_kernel_driver<CeedScalar,2,P,Q>(dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case 3: launch_failed = magma_gradn_2d_kernel_driver<CeedScalar,3,P,Q>(dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_gradn_2d_ncomp_q(
                magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride,
                      CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride,
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q) {
        case  1: launch_failed = magma_gradn_2d_ncomp<P, 1>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  2: launch_failed = magma_gradn_2d_ncomp<P, 2>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  3: launch_failed = magma_gradn_2d_ncomp<P, 3>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  4: launch_failed = magma_gradn_2d_ncomp<P, 4>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  5: launch_failed = magma_gradn_2d_ncomp<P, 5>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  6: launch_failed = magma_gradn_2d_ncomp<P, 6>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  7: launch_failed = magma_gradn_2d_ncomp<P, 7>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  8: launch_failed = magma_gradn_2d_ncomp<P, 8>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  9: launch_failed = magma_gradn_2d_ncomp<P, 9>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case 10: launch_failed = magma_gradn_2d_ncomp<P,10>(ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}


//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_gradn_2d_ncomp_q_p(
                magma_int_t P, magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
               const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride,
                      CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride,
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(P) {
        case  1: launch_failed = magma_gradn_2d_ncomp_q< 1>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  2: launch_failed = magma_gradn_2d_ncomp_q< 2>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  3: launch_failed = magma_gradn_2d_ncomp_q< 3>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  4: launch_failed = magma_gradn_2d_ncomp_q< 4>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  5: launch_failed = magma_gradn_2d_ncomp_q< 5>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  6: launch_failed = magma_gradn_2d_ncomp_q< 6>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  7: launch_failed = magma_gradn_2d_ncomp_q< 7>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  8: launch_failed = magma_gradn_2d_ncomp_q< 8>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case  9: launch_failed = magma_gradn_2d_ncomp_q< 9>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        case 10: launch_failed = magma_gradn_2d_ncomp_q<10>(Q, ncomp, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}



//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_gradn_2d( 
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,  
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode, 
    const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride,
          CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride,
    magma_int_t nelem, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_gradn_2d_ncomp_q_p(
                        P, Q, ncomp, 
                        dinterp1d, dgrad1d, transT, 
                        dU, u_elstride, u_compstride, u_dimstride, 
                        dV, v_elstride, v_compstride, v_dimstride, 
                        nelem, queue);

    return launch_failed;
}
