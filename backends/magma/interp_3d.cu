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
#include "interp_device.cuh"

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ CeedScalar shared_data[];
template<typename T, int DIM, int NCOMP, int P, int Q, int MAXPQ>
static __global__ void
magma_interp_3d_kernel(
    const T *dT, magma_trans_t transT,
    const T *dU, const int u_elstride, const int u_compstride, 
          T *dV, const int v_elstride, const int v_compstride)
{
    const int elem_id = blockIdx.x;
    const int tx      = threadIdx.x;
    T rU[DIM][NCOMP][MAXPQ] = { make_zero<T>() };    // for a non fused operator DIM is always 1
    T rV[DIM][NCOMP][MAXPQ] = { make_zero<T>() };    // for a non fused operator DIM is always 1
    T rTmp[Q] = { make_zero<T>() };

    // shift global memory pointers by elem stride
    dU += elem_id * u_elstride;
    dV += elem_id * v_elstride;

    // assign shared memory pointers
    T* sT    = (T*)(shared_data);
    T* sTmp  = sT + P*Q;

    // read T
    dread_T_gm2sm<P, Q>(tx, transT, dT, sT);
    __syncthreads();

    // read U (idim = 0 for dU, iDIM = 0 for rU, u_dimstride is always 0)
    readU_3d<T, P, DIM, NCOMP, MAXPQ, 0>(0, dU, u_compstride, 0, rU, sTmp, tx);
    // there is a sync at the end of this function

    // read V if transT is magmaTrans
    if(transT == MagmaTrans) {
        readV_3d<T, Q, DIM, NCOMP, MAXPQ, 0>(0, dV, v_compstride, 0, rV, tx);
    }
    //__syncthreads();    // sync here is probably not required

    magma_interp_3d_device<T, DIM, NCOMP, P, Q, MAXPQ>(sT, transT, rU , rV, tx, rTmp, sTmp);
    __syncthreads();

    // write V
    writeV_3d<T, Q, DIM, NCOMP, MAXPQ, 0>(0, dV, v_compstride, 0, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q>
static magma_int_t 
magma_interp_3d_kernel_driver(  
                const T *dT, magma_trans_t transT,
                const T *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
                      T *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;
    const int MAXPQ = maxpq(P,Q);

    magma_int_t shmem  = 0;
    shmem += sizeof(T)* (P*Q);  // for sT
    shmem += sizeof(T)* max(P*P*MAXPQ, P*Q*Q);  // rU needs P^2xP, the intermediate output needs max(P^2xQ,PQ^2)    
    magma_int_t nthreads = MAXPQ*MAXPQ; 

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if(shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_interp_3d_kernel<T,1,NCOMP,P,Q,MAXPQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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
        // IMPORTANT: we instantiate with DIM=1 instead of DIM=3 because the kernel handles one dimension at a time
        // We should instantiate with DIM >= 1 when we fuse the whole operator, because of the q-function
        magma_interp_3d_kernel<T,1,NCOMP,P,Q,MAXPQ><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        (dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride);
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
magma_interp_3d_ncomp(
                magma_int_t ncomp,
                const CeedScalar *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
                      CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(ncomp) {
        case 1: launch_failed = magma_interp_3d_kernel_driver<CeedScalar,1,P,Q>(dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 2: launch_failed = magma_interp_3d_kernel_driver<CeedScalar,2,P,Q>(dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 3: launch_failed = magma_interp_3d_kernel_driver<CeedScalar,3,P,Q>(dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
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
                const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
                      CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q) {
        case  1: launch_failed = magma_interp_3d_ncomp<P, 1>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  2: launch_failed = magma_interp_3d_ncomp<P, 2>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  3: launch_failed = magma_interp_3d_ncomp<P, 3>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  4: launch_failed = magma_interp_3d_ncomp<P, 4>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  5: launch_failed = magma_interp_3d_ncomp<P, 5>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  6: launch_failed = magma_interp_3d_ncomp<P, 6>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  7: launch_failed = magma_interp_3d_ncomp<P, 7>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  8: launch_failed = magma_interp_3d_ncomp<P, 8>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  9: launch_failed = magma_interp_3d_ncomp<P, 9>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 10: launch_failed = magma_interp_3d_ncomp<P,10>(ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}


//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_interp_3d_ncomp_q_p(
                magma_int_t P, magma_int_t Q, magma_int_t ncomp,
                const CeedScalar *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
                      CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(P) {
        case  1: launch_failed = magma_interp_1d_ncomp_q< 1>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  2: launch_failed = magma_interp_1d_ncomp_q< 2>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  3: launch_failed = magma_interp_1d_ncomp_q< 3>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  4: launch_failed = magma_interp_1d_ncomp_q< 4>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  5: launch_failed = magma_interp_1d_ncomp_q< 5>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  6: launch_failed = magma_interp_1d_ncomp_q< 6>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  7: launch_failed = magma_interp_1d_ncomp_q< 7>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  8: launch_failed = magma_interp_1d_ncomp_q< 8>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  9: launch_failed = magma_interp_1d_ncomp_q< 9>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 10: launch_failed = magma_interp_1d_ncomp_q<10>(Q, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}



//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_interp_3d( 
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,  
    const CeedScalar *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
          CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
    magma_int_t nelem, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_interp_3d_ncomp_q_p(
                        P, Q, ncomp, 
                        dT, transT, 
                        dU, u_elstride, u_compstride, 
                        dV, v_elstride, v_compstride, 
                        nelem, queue);

    return launch_failed;
}
