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
#include <magma.h>
#include "magma_tc_device.cuh"
#include "magma_dbasisApply_grad_device.cuh"

#define cu_ipow(a,b) ( (int)(__powf( (float)(a), (float)(b) ) ) )
#define ipow(a,b) ( (magma_int_t)(std::pow( (float)(a), (float)(b) ) ) )

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ double shared_data[];
template<int P, int Q>
static __global__ void
dbasis_apply_eval_grad_kernel_batched( 
    const int dim, const int ncomp, const int nqpt, const int ndof,  
    const int pre_org, const int tmp_size, 
    const double* dinterp1d, const double *dgrad1d, magma_trans_t transT,
    const double *dU, const int ustride, double *dV, const int vstride, const int dim_ctr)
{
    const int batchid = blockIdx.x;
    int tx = threadIdx.x;
    int pre, post;
    
    // advance to the respective element in the batch
    dU += batchid * ustride;
    dV += batchid * vstride;

    double* sTinterp = (double*)shared_data;
    double* sTgrad = sTinterp + P * Q;
    
    // read T in shared memory
    dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp );
    dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad );
    __syncthreads();

    pre  = pre_org; // the value of pre is independent from the loop below
    post = 1;
    dbasis_apply_eval_grad_device<P, Q>
    ( dim_ctr, dim, ncomp, pre, post, tmp_size, sTinterp, sTgrad, transT, dU, dV, shared_data + (2*P*Q) );
    __syncthreads();
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
dbasis_apply_eval_grad_kernel_batched_driver( 
                magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
                const double* dinterp1d, const double *dgrad1d, magma_trans_t transT,
                const double *dU, magma_int_t ustride, 
                      double *dV, magma_int_t vstride,
                magma_int_t batchCount, magma_int_t dim_ctr)
{

    // ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1);
    // originally the exponent is (dim-1), but we use dim because 
    // we have to read the original u in shared memory
    // the original implementation access u directly
    magma_int_t tmp_size = CeedIntPow(max(P,Q), dim); //ncomp * Q * CeedIntPow(max(P,Q), dim); 
    magma_int_t shmem = 2 * P * Q * sizeof(double);
    shmem += 2 * tmp_size * sizeof(double); 
    
    // using __powf() + integer cast rounds down (like floor)
    magma_int_t pre = CeedIntPow(P, dim-1); 
    
    magma_int_t ndof = CeedIntPow(P, dim);

    magma_int_t nthreads = max(P, CeedIntPow(Q, dim-1) ); 
    nthreads = magma_roundup( nthreads, Q ); // nthreads must be multiple of Q
    
    //printf("%d threads, shared memory = %f KB\n", nthreads, (float)shmem / 1024.0);
    
    if( nthreads > 1024 || shmem >= 48000 ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(batchCount, 1, 1);
        dbasis_apply_eval_grad_kernel_batched<P, Q><<<grid, threads, shmem, 0>>>
        ( dim, ncomp, nqpt, ndof, pre, tmp_size, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, dim_ctr);
        return 0;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magmablas_dbasis_apply_batched_eval_grad_2( 
    magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
    const double* dinterp1d, const double *dgrad1d, magma_trans_t transT,
    const double *dU, magma_int_t ustride, 
          double *dV, magma_int_t vstride,
    magma_int_t batchCount, magma_int_t dim_ctr)
{
    magma_int_t launch_failed = 0;
    switch(Q){
        case  1: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 1>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  2: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 2>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  3: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 3>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  4: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 4>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  5: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 5>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  6: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 6>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  7: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 7>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  8: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 8>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  9: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P, 9>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case 10: launch_failed = dbasis_apply_eval_grad_kernel_batched_driver<P,10>( dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magmablas_dbasis_apply_batched_eval_grad_1( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
    const double* dinterp1d, const double *dgrad1d, magma_trans_t transT,
    const double *dU, magma_int_t ustride, 
          double *dV, magma_int_t vstride,
    magma_int_t batchCount, magma_int_t dim_ctr )
{
    magma_int_t launch_failed = 0;
    switch(P){
        case  1: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 1>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  2: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 2>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  3: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 3>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  4: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 4>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  5: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 5>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  6: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 6>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  7: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 7>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  8: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 8>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case  9: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2< 9>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        case 10: launch_failed = magmablas_dbasis_apply_batched_eval_grad_2<10>(Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT, dU, ustride, dV, vstride, batchCount, dim_ctr ); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" void 
magmablas_dbasis_apply_batched_eval_grad( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt, 
    const double* dinterp1d, const double *dgrad1d, CeedTransposeMode tmode,
    const double *dU, magma_int_t ustride, 
          double *dV, magma_int_t vstride,
    magma_int_t batchCount, magma_int_t dim_ctr)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magmablas_dbasis_apply_batched_eval_grad_1(P, Q, dim, ncomp, nqpt, dinterp1d, dgrad1d, transT,  dU, ustride, dV, vstride, batchCount, dim_ctr);
    
    if(launch_failed == 1) {
        // fall back to a ref. impl.
        //printf("launch failed. TODO: add ref. impl.\n");
    }
}
