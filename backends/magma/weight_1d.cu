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
#include "magma_tc_device.cuh"
#include "weight_device.cuh"

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ CeedScalar shared_data[];
template<typename T, int Q>
static __global__ void
magma_weight_1d_kernel(const T *dqweight1d, T *dV, const int v_stride)
{
    const int batchid = blockIdx.x;
    const int tx      = threadIdx.x;

    // global memory pointers
    dV += batchid * v_stride;

    // shared memory pointers
    T* sTweight = (T*)shared_data;
    T* sV = sTweight + Q;

    // read dqweight_1d
    if(tx < Q) {
        sTweight[tx] = dqweight1d[tx];
    }
    __syncthreads();

    magma_weight_1d_device<T, Q>(sTweight, sV, tx);

    // write V
    dV[ tx ] = sV[ tx ];
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static magma_int_t 
magma_weight_1d_kernel_driver(const T *dqweight1d, T *dV, magma_int_t v_stride, magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;

    magma_int_t shmem  = 0;
    shmem += sizeof(T) * (2*Q);  // for dqweight1d and output 
    magma_int_t nthreads = Q; 

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if(shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_weight_1d_kernel<T, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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
        magma_weight_1d_kernel<T, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        (dqweight1d, dV, v_stride);
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_weight_1d_q(
        magma_int_t Q, const CeedScalar *dqweight1d, 
        CeedScalar *dV, magma_int_t v_stride, 
        magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q) {
        case  1: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 1>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  2: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 2>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  3: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 3>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  4: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 4>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  5: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 5>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  6: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 6>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  7: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 7>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  8: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 8>(dqweight1d, dV, v_stride, nelem, queue); break;
        case  9: launch_failed = magma_weight_1d_kernel_driver<CeedScalar, 9>(dqweight1d, dV, v_stride, nelem, queue); break;
        case 10: launch_failed = magma_weight_1d_kernel_driver<CeedScalar,10>(dqweight1d, dV, v_stride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_weight_1d( 
    magma_int_t Q, const CeedScalar *dqweight1d, 
    CeedScalar *dV, magma_int_t v_stride, 
    magma_int_t nelem, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_weight_1d_q(Q, dqweight1d, dV, v_stride, nelem, queue);
    return launch_failed;
}
