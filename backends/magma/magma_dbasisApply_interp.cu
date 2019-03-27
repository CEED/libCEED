
#include "ceed-magma.h"
#include "magma_tc_device.cuh"
#include "magma_dbasisApply_interp_device.cuh"

#define ipow(a,b) ( (magma_int_t)(std::pow( (float)(a), (float)(b) ) ) )

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ double shared_data[];
template<int P, int Q>
__global__ void
dbasis_apply_eval_interp_kernel_batched( 
    const int dim, const int ncomp, const int pre_org, const int post_org, const int tmp_size, 
    const double *dT, magma_trans_t transT,
    const double *dU, const int ustride, double *dV, const int vstride)
{
    const int batchid = blockIdx.x; 
    dbasis_apply_eval_interp_device< P, Q >
    ( dim, ncomp, pre_org, post_org, tmp_size, dT, transT, dU+(batchid*ustride), dV+(batchid*vstride), shared_data );
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
magma_int_t 
dbasis_apply_eval_interp_kernel_batched_driver( 
                magma_int_t dim, magma_int_t ncomp,  
                const double *dT, magma_trans_t transT,
                const double *dU, magma_int_t ustride, double *dV, magma_int_t vstride, 
                magma_int_t batchCount )
{
    magma_int_t pre  = ncomp * ipow(P, dim-1); //ncomp*CeedIntPow(P, dim-1);
    magma_int_t post = 1; 

    // ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1);
    // originally the exponent is (dim-1), but we use dim because 
    // we have to read the original u in shared memory
    // the original implementation access u directly
    magma_int_t tmp_size = ncomp * ipow(max(P,Q), dim); //ncomp * Q * ipow(max(P,Q), dim); 
    magma_int_t shmem = P * Q * sizeof(double);
    shmem += 2 * tmp_size * sizeof(double); 
    
    
    magma_int_t nthreads = max(P, ipow(Q, dim-1) ); 
    nthreads = magma_roundup( nthreads, Q ); // nthreads must be multiple of Q
    
    //printf("%d threads, shared memory = %f KB\n", nthreads, (float)shmem / 1024.0);
    
    if( nthreads > 1024 || shmem >= 48000 ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(batchCount, 1, 1);
        dbasis_apply_eval_interp_kernel_batched<P, Q><<<grid, threads, shmem, 0>>>
        ( dim, ncomp, pre, post, tmp_size, dT, transT, dU, ustride, dV, vstride );
        
        return 0;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
magma_int_t 
magmablas_dbasis_apply_batched_eval_interp_2( 
    magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const double *dT, magma_trans_t transT,
    const double *dU, magma_int_t ustride, double *dV, magma_int_t vstride, 
    magma_int_t batchCount )
{
    magma_int_t launch_failed = 0;
    switch(Q){
        case  1: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 1>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  2: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 2>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  3: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 3>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  4: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 4>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  5: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 5>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  6: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 6>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  7: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 7>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  8: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 8>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  9: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P, 9>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case 10: launch_failed = dbasis_apply_eval_interp_kernel_batched_driver<P,10>(dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
magma_int_t 
magmablas_dbasis_apply_batched_eval_interp_1( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const double *dT, magma_trans_t transT,
    const double *dU, magma_int_t ustride, double *dV, magma_int_t vstride, 
    magma_int_t batchCount )
{
    magma_int_t launch_failed = 0;
    switch(P){
        case  1: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 1>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  2: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 2>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  3: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 3>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  4: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 4>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  5: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 5>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  6: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 6>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  7: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 7>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  8: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 8>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case  9: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2< 9>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        case 10: launch_failed = magmablas_dbasis_apply_batched_eval_interp_2<10>(Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" void 
magmablas_dbasis_apply_batched_eval_interp( 
    magma_int_t P, magma_int_t Q, 
    magma_int_t dim, magma_int_t ncomp,  
    const double *dT, CeedTransposeMode tmode,
    const double *dU, magma_int_t ustride, 
          double *dV, magma_int_t vstride, 
    magma_int_t batchCount )
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magmablas_dbasis_apply_batched_eval_interp_1(P, Q, dim, ncomp, dT, transT, dU, ustride, dV, vstride, batchCount );
    
    if(launch_failed == 1) {
        // fall back to a ref. impl.
        //printf("launch failed. TODO: add ref. impl.\n");
    }
}


