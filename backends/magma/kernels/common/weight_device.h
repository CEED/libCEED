// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_WEIGHT_DEVICE_H
#define CEED_MAGMA_WEIGHT_DEVICE_H

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 1D
template<typename T, int Q>
__device__ __inline__ void
magma_weight_1d_device(const T* sTweight, T* sV, const int tx) 
{
    // Assumptions
    // 1. 1D thread configuration of size Q
    // 2. The output sV is in shared memory -- size 1xQ
    if (tx < Q){
        sV[tx] = sTweight[tx];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 2D
template<typename T, int DIM, int NCOMP, int Q, int iDIM, int iCOMP>
__device__ __inline__ void
magma_weight_2d_device(const T* sTweight, T rV[DIM][NCOMP][Q], const int tx) 
{
    // Assumptions
    // 1. 1D thread configuration of size Q
    // 2. rV[][][] matches the storage used in other actions (interp, grad, ... etc)
    // 3. iDIM and iCOMP specify which indexes to use in rV, 
    //    since the output per thread is a register array of size Q
    // 4. Sync is recommended after the call (to make sure sTweight can be overwritten)

    if (tx < Q) {
        // x sTweight[j]  for first update
        // x sTweight[tx] for second update
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] = sTweight[j] * sTweight[tx];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 2D
template<typename T, int DIM, int NCOMP, int Q, int iDIM, int iCOMP>
__device__ __inline__ void
magma_weight_3d_device(const T* sTweight, T rV[DIM][NCOMP][Q], const int tx) 
{
    // Assumptions
    // 1. 1D thread configuration of size Q^2
    // 2. rV[][][] matches the storage used in other actions (interp, grad, ... etc)
    // 3. iDIM and iCOMP specify which indexes to use in rV, 
    //    since the output per thread is a register array of size Q
    // 4. Sync is recommended after the call (to make sure sTweight can be overwritten)

    if (tx < (Q*Q)) {
        // x sTweight[j]    for first update
        // x sTweight[tx%Q] for second update
        // x sTweight[tx/Q] for third update
        for(int j = 0; j < Q; j++) {
            rV[iDIM][iCOMP][j] = sTweight[j] * sTweight[tx%Q] * sTweight[tx/Q];
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
    for(i = 0; i < Q-nthreads; i += nthreads) {
        sqweight1d[i+tx] = dqweight1d[i+tx];
    }

    if (tx < Q-i) {
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

#endif    // CEED_MAGMA_WEIGHT_DEVICE_H
