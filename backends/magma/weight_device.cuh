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

#ifndef MAGMA_WEIGHT_DEVICE_CUH
#define MAGMA_WEIGHT_DEVICE_CUH

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 1D
template<typename T, int Q>
__device__ __inline__ void
magma_weight_1d_device(const T* sTweight, T* sV, const int tx) 
{
    // Assumptions
    // 1. 1D thread configuration of size Q
    // 2. The output sV is in shared memory -- size 1xQ
    if(tx < Q){
        sV[tx] = sTweight[tx];
    }
    __syncthreads();
}

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 2D
template<typename T, int DIM, int NCOMP, int Q, int iDIM, int iCOMP>
__device__ __inline__ void
magma_weight_2d_device(const T* sTweight, T* rV[DIM][NCOMP][Q], const int tx) 
{
    // Assumptions
    // 1. 1D thread configuration of size Q
    // 2. rV[][][] matches the storage used in other actions (interp, grad, ... etc)
    // 3. iDIM and iCOMP specify which indexes to use in rV, 
    //    since the output per thread is a register array of size Q
    // 4. Sync is recommended after the call (to make sure sTweight can be overwritten)

    if(tx < Q) {
        // first update
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] = sTweight[j];
        }

        // second update
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] *= sTweight[tx];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 2D
template<typename T, int DIM, int NCOMP, int Q, int iDIM, int iCOMP>
__device__ __inline__ void
magma_weight_3d_device(const T* sTweight, T* rV[DIM][NCOMP][Q], const int tx) 
{
    // Assumptions
    // 1. 1D thread configuration of size Q^2
    // 2. rV[][][] matches the storage used in other actions (interp, grad, ... etc)
    // 3. iDIM and iCOMP specify which indexes to use in rV, 
    //    since the output per thread is a register array of size Q
    // 4. Sync is recommended after the call (to make sure sTweight can be overwritten)

    if(tx < (Q*Q)) {
        int tx_;
        // first update
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] = sTweight[j];
        }

        // second update
        tx_ = tx % Q;
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] *= sTweight[tx_];
        }

        // second update
        tx_ = tx / Q;
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] *= sTweight[tx_];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static __device__ __inline__ void
magma_weight_generic_device( 
    const int dim, 
    const int pre_org, const int post_org,  
    const T *dqweight1d, T *dV, 
    T* shared_data )
{
    const int nthreads = blockDim.x;
    const int tx = threadIdx.x;
    int pre  = pre_org;
    int post = post_org;

    int tx_      = tx % post;
    int slice_id = tx / post;
    
    // the size of V is Q^dim, which is pre * post
    T* sVorg      = (T*)shared_data;
    T* sqweight1d = sVorg + (pre*post*Q);
    T* sV;

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

#endif    // MAGMA_WEIGHT_DEVICE_CUH
