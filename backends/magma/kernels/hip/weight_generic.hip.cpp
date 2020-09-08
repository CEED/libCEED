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
magma_weight_generic_kernel( 
    const int dim, const int pre_org, const int post_org, 
    const T *dqweight1d, 
    T *dV, const int vstride)
{
    HIP_DYNAMIC_SHARED( CeedScalar, shared_data)
    const int batchid = blockIdx.x; 
    magma_weight_generic_device<T, Q>
    ( dim, pre_org, post_org, dqweight1d, dV+(batchid*vstride), shared_data );
}

//////////////////////////////////////////////////////////////////////////////////////////
static __global__ void 
magma_weight_nontensor_kernel(const CeedInt nelem, const CeedInt Q,
                    const CeedScalar *__restrict__ qweight,
                    CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;
  //TODO load qweight in shared memory if blockDim.z > 1?                                           
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    d_V[elem*Q + tid] = qweight[tid];
  }
}


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

    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
 
    if ( nthreads > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(batchCount, 1, 1);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(magma_weight_generic_kernel<T, Q>), dim3(grid), dim3(threads), shmem, magma_queue_get_hip_stream(queue),  dim, pre_org, post_org, dqweight1d, dV, vstride );
        return (hipPeekAtLastError() == hipSuccess) ? 0 : 1;        
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
    hipLaunchKernelGGL(magma_weight_nontensor_kernel, dim3(grid), dim3(threads), 0, magma_queue_get_hip_stream(queue), nelem, Q, dqweight, dv);
}
