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
#include "interp_device.cuh"

#define ipow(a,b) ( (magma_int_t)(std::pow( (float)(a), (float)(b) ) ) )

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ double shared_data[];
template<int P, int Q>
static __global__ void
interp_generic_kernel( 
    const int dim, const int ncomp, const int pre_org, const int post_org, const int tmp_size, 
    const double *dT, magma_trans_t transT,
    const double *dU, const int u_elstride, const int u_compstride, 
          double *dV, const int v_elstride, const int v_compstride)
{
    const int elem_id = blockIdx.x; 
    const int comp_id = blockIdx.y;
    dbasis_apply_eval_interp_device< P, Q >
    ( dim, ncomp, pre_org, post_org, tmp_size, dT, transT, 
      dU + (elem_id * u_elstride) + (comp_id * u_compstride), 
      dV + (elem_id * v_elstride) + (comp_id * v_compstride), 
      shared_data );
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
interp_generic_kernel_driver( 
                magma_int_t dim, magma_int_t ncomp,  
                const double *dT, magma_trans_t transT,
                const double *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
                      double *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t pre = ipow(P, dim-1); //ncomp*CeedIntPow(P, dim-1);
    magma_int_t post = 1; 

    // ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1);
    // originally the exponent is (dim-1), but we use dim because 
    // we have to read the original u in shared memory
    // the original implementation access u directly
    magma_int_t tmp_size = ipow(max(P,Q), dim); //ncomp * Q * ipow(max(P,Q), dim); 
    magma_int_t shmem = P * Q * sizeof(double);
    shmem += 2 * tmp_size * sizeof(double); 
    
    
    magma_int_t nthreads = max(P, ipow(Q, dim-1) ); 
    nthreads = magma_roundup( nthreads, Q ); // nthreads must be multiple of Q
    
    #if 1 //CUDA_VERSION >= 9000
    if(shmem > 49152) {
        cudaFuncSetAttribute( interp_generic_kernel<P, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    if( nthreads > 1024 || shmem >= 98000 ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(nelem, ncomp, 1);
        interp_generic_kernel<P, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        ( dim, ncomp, pre, post, tmp_size, dT, transT, 
          dU, u_elstride, u_compstride, 
          dV, v_elstride, v_compstride );
        
        return 0;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_interp_generic_q( 
    magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const double *dT, magma_trans_t transT,
    const double *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
          double *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q){
        case  1: launch_failed = interp_generic_kernel_driver<P, 1>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  2: launch_failed = interp_generic_kernel_driver<P, 2>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  3: launch_failed = interp_generic_kernel_driver<P, 3>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  4: launch_failed = interp_generic_kernel_driver<P, 4>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  5: launch_failed = interp_generic_kernel_driver<P, 5>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  6: launch_failed = interp_generic_kernel_driver<P, 6>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  7: launch_failed = interp_generic_kernel_driver<P, 7>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  8: launch_failed = interp_generic_kernel_driver<P, 8>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  9: launch_failed = interp_generic_kernel_driver<P, 9>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 10: launch_failed = interp_generic_kernel_driver<P,10>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 11: launch_failed = interp_generic_kernel_driver<P,11>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 12: launch_failed = interp_generic_kernel_driver<P,12>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 13: launch_failed = interp_generic_kernel_driver<P,13>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 14: launch_failed = interp_generic_kernel_driver<P,14>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 15: launch_failed = interp_generic_kernel_driver<P,15>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 16: launch_failed = interp_generic_kernel_driver<P,16>(dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
magma_int_t 
static magma_interp_generic_q_p( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const double *dT, magma_trans_t transT,
    const double *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
          double *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(P){
        case  1: launch_failed = magma_interp_generic_q< 1>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  2: launch_failed = magma_interp_generic_q< 2>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  3: launch_failed = magma_interp_generic_q< 3>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  4: launch_failed = magma_interp_generic_q< 4>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  5: launch_failed = magma_interp_generic_q< 5>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  6: launch_failed = magma_interp_generic_q< 6>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  7: launch_failed = magma_interp_generic_q< 7>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  8: launch_failed = magma_interp_generic_q< 8>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case  9: launch_failed = magma_interp_generic_q< 9>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 10: launch_failed = magma_interp_generic_q<10>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 11: launch_failed = magma_interp_generic_q<11>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 12: launch_failed = magma_interp_generic_q<12>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 13: launch_failed = magma_interp_generic_q<13>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 14: launch_failed = magma_interp_generic_q<14>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 15: launch_failed = magma_interp_generic_q<15>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        case 16: launch_failed = magma_interp_generic_q<16>(Q, dim, ncomp, dT, transT, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_interp_generic( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const double *dT, CeedTransposeMode tmode,
    const double *dU, magma_int_t u_elstride, magma_int_t u_compstride, 
          double *dV, magma_int_t v_elstride, magma_int_t v_compstride, 
    magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;

    launch_failed = magma_interp_generic_q_p(
                        P, Q, dim, ncomp, 
                        dT, transT, 
                        dU, u_elstride, u_compstride, 
                        dV, v_elstride, v_compstride, 
                        nelem, queue);

    return launch_failed;
}
