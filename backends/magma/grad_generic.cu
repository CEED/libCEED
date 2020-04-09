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
#include <magma_v2.h>
#include "magma_tc_device.cuh"
#include "grad_device.cuh"

#define cu_ipow(a,b) ( (int)(__powf( (float)(a), (float)(b) ) ) )
#define ipow(a,b) ( (magma_int_t)(std::pow( (float)(a), (float)(b) ) ) )

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ CeedScalar shared_data[];
template<typename T, int P, int Q>
static __global__ void
magma_grad_generic_kernel( 
    const int dim, const int ncomp, const int nqpt, 
    const int pre_org, const int tmp_size, 
    const T* dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, const int u_elstride, const int u_compstride, const int u_dimstride, 
          T *dV, const int v_elstride, const int v_compstride, const int v_dimstride, 
    const int dim_id )
{
    const int elem_id = blockIdx.x;
    const int comp_id = blockIdx.y;
    int tx = threadIdx.x;
    int pre, post;
    
    // advance to the respective element in the batch
    dU += (elem_id * u_elstride) + (comp_id * u_compstride);
    dV += (elem_id * v_elstride) + (comp_id * v_compstride);

    T* sTinterp = (T*)shared_data;
    T* sTgrad = sTinterp + P * Q;
    
    // read T in shared memory
    dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp );
    dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad );
    __syncthreads();

    pre  = pre_org; // the value of pre is independent from the loop below
    post = 1;
    magma_grad_generic_device<T, P, Q>
    ( dim_id, dim, ncomp, pre, post, tmp_size, sTinterp, sTgrad, transT, dU, dV, shared_data + (2*P*Q) );
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int P, int Q>
static magma_int_t 
magma_grad_generic_kernel_driver( 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
    const T* dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, magma_int_t u_elstride, const int u_compstride, const int u_dimstride,
          T *dV, magma_int_t v_elstride, const int v_compstride, const int v_dimstride,
    magma_int_t dim_id, magma_int_t nelem, magma_queue_t queue)
{
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

    if( shmem >= 49000 ) {
        cudaFuncSetAttribute(magma_grad_generic_kernel<P,Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }

    if( nthreads > 1024 ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(nelem, ncomp, 1);
        magma_grad_generic_kernel<T, P, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        ( dim, ncomp, nqpt, pre, tmp_size, dinterp1d, dgrad1d, transT, 
          dU, u_elstride, u_compstride, u_dimstride, 
          dV, v_elstride, v_compstride, v_dimstride, 
          dim_id );
        return 0;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_grad_generic_q( 
    magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
    const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride,
          CeedScalar *dV, magma_int_t v_elstride,magma_int_t v_compstride, magma_int_t v_dimstride,
    magma_int_t dim_id, magma_int_t nelem, 
    magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q){
        case  1: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 1>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  2: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 2>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  3: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 3>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  4: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 4>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  5: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 5>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  6: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 6>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  7: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 7>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  8: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 8>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  9: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P, 9>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case 10: launch_failed = magma_grad_generic_kernel_driver<CeedScalar, P,10>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_grad_generic_q_p( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt,
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, magma_trans_t transT,
    const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride,
          CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride,
    magma_int_t dim_id, magma_int_t nelem, 
    magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(P){
        case  1: launch_failed = magma_grad_generic_q< 1>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  2: launch_failed = magma_grad_generic_q< 2>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  3: launch_failed = magma_grad_generic_q< 3>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  4: launch_failed = magma_grad_generic_q< 4>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  5: launch_failed = magma_grad_generic_q< 5>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  6: launch_failed = magma_grad_generic_q< 6>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  7: launch_failed = magma_grad_generic_q< 7>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  8: launch_failed = magma_grad_generic_q< 8>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case  9: launch_failed = magma_grad_generic_q< 9>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        case 10: launch_failed = magma_grad_generic_q<10>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, u_elstride, u_compstride, u_dimstride, dV, v_elstride, v_compstride, v_dimstride, dim_id, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t  
magma_grad_generic( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride, 
          CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride, 
    magma_int_t dim_id, magma_int_t nelem, 
    magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_grad_generic_q_p( P, Q, dim, ncomp, nqpt, 
                                            dinterp1d, dgrad1d, transT,  
                                            dU, u_elstride, u_compstride, u_dimstride, 
                                            dV, v_elstride, v_compstride, v_dimstride, 
                                            dim_id, nelem, queue);
    return launch_failed;    
}
