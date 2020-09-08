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
#include "hip/hip_runtime.h"
#include <magma_v2.h>
#include "../common/magma_common_device.h"
#include "../common/weight_device.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static __global__ void
magma_weight_3d_kernel(const T *dqweight1d, T *dV, const int v_stride, const int nelem)
{
    HIP_DYNAMIC_SHARED( CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T rV[1][1][Q];    // allocate with DIM=NCOMP=1, but sizes may differ for a fused operator
    // global memory pointers
    dV += elem_id * v_stride;

    // shared memory pointers
    T* sTweight = (T*)shared_data;

    // read dqweight_1d
    if (tx < Q) {
        sTweight[tx] = dqweight1d[tx];
    }
    __syncthreads();

    magma_weight_3d_device<T, 1, 1, Q, 0, 0>(sTweight, rV, tx);

    // write V
    if (tx < (Q*Q)) {
        for(int j = 0; j < Q; j++) {
            dV[ j*(Q*Q) + tx ] = rV[0][0][j];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static magma_int_t 
magma_weight_3d_kernel_driver(
    const T *dqweight1d, T *dV, magma_int_t v_stride, 
    magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;

    magma_int_t nthreads = (Q*Q); 
    magma_int_t ntcol = (maxthreads < nthreads) ? 1 : (maxthreads / nthreads);
    magma_int_t shmem  = 0;
    shmem += sizeof(T) * Q;  // for dqweight1d 

    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);

    if ( (nthreads*ntcol) > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        magma_int_t nblocks = (nelem + ntcol-1) / ntcol;
        dim3 threads(nthreads, ntcol, 1);
        dim3 grid(nblocks, 1, 1);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(magma_weight_3d_kernel<T, Q>), dim3(grid), dim3(threads), shmem, magma_queue_get_hip_stream(queue), dqweight1d, dV, v_stride, nelem);
        return (hipPeekAtLastError() == hipSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_weight_3d_q(
        magma_int_t Q, const CeedScalar *dqweight1d, 
        CeedScalar *dV, magma_int_t v_stride, 
        magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 1>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  2: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 2>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  3: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 3>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  4: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 4>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  5: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 5>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  6: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 6>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  7: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 7>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  8: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 8>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  9: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 9>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case 10: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar,10>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_weight_3d( 
    magma_int_t Q, const CeedScalar *dqweight1d, 
    CeedScalar *dV, magma_int_t v_stride, 
    magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_weight_3d_q(Q, dqweight1d, dV, v_stride, nelem, maxthreads, queue);
    return launch_failed;
}
