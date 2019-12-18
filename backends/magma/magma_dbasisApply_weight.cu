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

//////////////////////////////////////////////////////////////////////////////////////////
template<int Q>
static __device__ __inline__ void
dbasis_apply_eval_weight_device( 
    const int dim, 
    const int pre_org, const int post_org,  
    const double *dqweight1d, double *dV, 
    double* shared_data )
{
    const int nthreads = blockDim.x;
    const int tx = threadIdx.x;
    int pre  = pre_org;
    int post = post_org;

    int tx_      = tx % post;
    int slice_id = tx / post;
    
    // the size of V is Q^dim, which is pre * post
    double* sVorg      = (double*)shared_data;
    double* sqweight1d = sVorg + (pre*post*Q);
    double* sV;

    // read qweight1d into shared memory
    int i = 0;
    #pragma unroll
    for(i = 0; i < Q-nthreads; i += nthreads) {
        sqweight1d[i+tx] = dqweight1d[i+tx];
    }

    if(tx < Q-i) {
        sqweight1d[i+tx] = dqweight1d[i+tx];
    }
     __syncthreads();

    // first iteration -- special case
    sV = sVorg + slice_id * post * Q;
    #pragma unroll
    for(int j = 0; j < Q; j++) {
        sV[j * post + tx_] = sqweight1d[j];
    }
    __syncthreads();
    
    // rest of iterations
    #pragma unroll
    for(int d = 1; d < dim; d++) {
        // remapping
        pre  /= Q;
        post *= Q;
        tx_      = tx % post; 
        slice_id = tx / post;
        sV = sVorg + slice_id * post * Q;
        #pragma unroll
        for(int j = 0; j < Q; j++) {
            sV[j * post + tx_] *= sqweight1d[j];
        }
        __syncthreads();
    }

    // write V back, advance dV and 
    // use the values of pre, post, tx_, and sV
    dV += slice_id * post * Q;
    #pragma unroll
    for(int j = 0; j < Q; j++) {
        dV[j * post + tx_] = sV[j * post + tx_];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
extern __shared__ double shared_data[];
template<int Q>
static __global__ void
dbasis_apply_eval_weight_kernel_batched( 
    const int dim, const int pre_org, const int post_org, 
    const double *dqweight1d, 
    double *dV, const int vstride)
{
    const int batchid = blockIdx.x; 
    dbasis_apply_eval_weight_device< Q >
    ( dim, pre_org, post_org, dqweight1d, dV+(batchid*vstride), shared_data );
}

//////////////////////////////////////////////////////////////////////////////////////////

static __global__ void 
magma_weight_kernel(const CeedInt nelem, const CeedInt Q,
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
template<int Q>
static magma_int_t 
dbasis_apply_eval_weight_kernel_batched_driver( 
                magma_int_t dim,   
                const double *dqweight1d, 
                double *dV, magma_int_t vstride, 
                magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t pre_org  = CeedIntPow(Q, dim-0-1);
    magma_int_t post_org = CeedIntPow(Q, 0);

    magma_int_t vsize = CeedIntPow(Q, dim); 
    magma_int_t shmem = vsize * sizeof(double);  // holds dV in shared memory
    shmem += (Q * sizeof(double)); // holds qweight1d

    magma_int_t nthreads = CeedIntPow(Q, dim-1);
    //printf("nthreads = %d, shmem = %f\n", nthreads, shmem/1024.0);

    if( nthreads > 1024 || shmem >= 48000 ) {
        return 1;    // launch failed
    }
    else { 
        dim3 threads(nthreads, 1, 1);
        dim3 grid(batchCount, 1, 1);
        dbasis_apply_eval_weight_kernel_batched<Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        ( dim, pre_org, post_org, dqweight1d, dV, vstride );
        
        return 0;
    } 
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magmablas_dbasis_apply_batched_eval_weight_1( 
    magma_int_t Q, magma_int_t dim, 
    const double *dqweight1d, 
    double *dV, magma_int_t vstride, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch(Q){
        case  1: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 1>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  2: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 2>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  3: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 3>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  4: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 4>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  5: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 5>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  6: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 6>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  7: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 7>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  8: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 8>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case  9: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver< 9>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        case 10: launch_failed = dbasis_apply_eval_weight_kernel_batched_driver<10>(dim, dqweight1d, dV, vstride, batchCount, queue); break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" void 
magmablas_dbasis_apply_batched_eval_weight( 
    magma_int_t Q, magma_int_t dim, 
    const double *dqweight1d, 
    double *dV, magma_int_t vstride, 
    magma_int_t batchCount, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    launch_failed = magmablas_dbasis_apply_batched_eval_weight_1(Q, dim, dqweight1d, dV, vstride, batchCount, queue);
    
    if(launch_failed == 1) {
        // fall back to a ref. impl.
        //printf("launch failed. TODO: add ref. impl.\n");
    }
}

// NonTensor weight function
extern "C" void 
magma_weight(magma_int_t grid, magma_int_t threads, magma_int_t nelem, magma_int_t Q, 
             double *dqweight, double *dv, magma_queue_t queue)
{
    magma_weight_kernel<<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>(nelem, Q, dqweight, dv);
}
