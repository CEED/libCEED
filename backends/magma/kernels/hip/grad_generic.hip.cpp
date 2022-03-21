// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "hip/hip_runtime.h"
#include "../common/grad.h"

#define hip_ipow(a,b) ( (int)(__powf( (float)(a), (float)(b) ) ) )
#define ipow(a,b) ( (magma_int_t)(std::pow( (float)(a), (float)(b) ) ) )

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int P, int Q>
static magma_int_t 
magma_grad_generic_kernel_driver( 
    magma_int_t dim, magma_int_t ncomp, 
    const T* dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, magma_int_t estrdU, const int cstrdU, 
          T *dV, magma_int_t estrdV, const int cstrdV, 
    magma_int_t dim_id, magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );

    magma_int_t shmem_max, nthreads_max;
    // ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1);
    // originally the exponent is (dim-1), but we use dim because 
    // we have to read the original u in shared memory
    // the original implementation access u directly
    magma_int_t tmp_size = CeedIntPow(max(P,Q), dim); //ncomp * Q * CeedIntPow(max(P,Q), dim); 
    magma_int_t shmem = 2 * P * Q * sizeof(T);
    shmem += 2 * tmp_size * sizeof(T); 
    
    magma_int_t pre = CeedIntPow(P, dim-1);     
    magma_int_t nthreads = max(P, CeedIntPow(Q, dim-1) ); 
    nthreads = magma_roundup( nthreads, Q ); // nthreads must be multiple of Q

    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);

    if ( nthreads > nthreads_max || shmem > shmem_max ) {
        return 1;
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(nelem, ncomp, 1);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(magma_grad_generic_kernel<T, P, Q>), dim3(grid), dim3(threads), shmem, magma_queue_get_hip_stream(queue),  dim, ncomp, pre, tmp_size, dinterp1d, dgrad1d, transT, 
          dU, estrdU, cstrdU, 
          dV, estrdV, cstrdV, 
          dim_id );
        return (hipPeekAtLastError() == hipSuccess) ? 0 : 1;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_grad_generic_q( 
    magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, 
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t dim_id, magma_int_t nelem, 
    magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q){
        case  1: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 1>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 2>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 3>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 4>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 5>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 6>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 7>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 8>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 9>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P,10>
          ( dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_grad_generic_q_p( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, 
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t dim_id, magma_int_t nelem, 
    magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(P){
        case  1: 
          launch_failed = magma_grad_generic_q< 1>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_grad_generic_q< 2>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_grad_generic_q< 3>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_grad_generic_q< 4>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_grad_generic_q< 5>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_grad_generic_q< 6>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_grad_generic_q< 7>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_grad_generic_q< 8>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_grad_generic_q< 9>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_grad_generic_q<10>
          (Q, dim, ncomp, dinterp1d, dgrad1d, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, dim_id, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t  
magma_grad_generic( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, 
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t u_dimstride, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t v_dimstride, 
    magma_int_t nelem, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    // Loop through grad dimensions only, batch call over elements and components
    for (CeedInt dim_ctr = 0; dim_ctr < dim; dim_ctr++) { 
        launch_failed = magma_grad_generic_q_p(
                P, Q, dim, ncomp, 
                dinterp1d, dgrad1d, transT, 
                dU + dim_ctr * u_dimstride, estrdU, cstrdU, 
                dV + dim_ctr * v_dimstride, estrdV, cstrdV, 
                dim_ctr, nelem, queue ); 
        if (launch_failed != 0) break;
    }

    return launch_failed;    
}
