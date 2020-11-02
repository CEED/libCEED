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

#ifndef CEED_MAGMA_WEIGHT_H
#define CEED_MAGMA_WEIGHT_H

#include <ceed.h>
#include <magma_v2.h>
#include "magma_common_device.h"
#include "weight_device.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static __global__ void
magma_weight_1d_kernel(const T *dqweight1d, T *dV, const int v_stride, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    // global memory pointers
    dV += elem_id * v_stride;

    // shared memory pointers
    T* sTweight = (T*)shared_data;
    T* sV = sTweight + Q;
    sV   += ty * Q;

    // read dqweight_1d
    if (ty == 0 && tx < Q) {
        sTweight[tx] = dqweight1d[tx];
    }

    __syncthreads();
    magma_weight_1d_device<T, Q>(sTweight, sV, tx);
    __syncthreads();

    // write V
    dV[ tx ] = sV[ tx ];
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static __global__ void
magma_weight_2d_kernel(const T *dqweight1d, T *dV, const int v_stride, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

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
    if (ty == 0 && tx < Q) {
        sTweight[tx] = dqweight1d[tx];
    }

    __syncthreads();
    magma_weight_2d_device<T, 1, 1, Q, 0, 0>(sTweight, rV, tx);

    // write V
    if (tx < Q) {
        for(int j = 0; j < Q; j++) {
            dV[ j*Q + tx ] = rV[0][0][j];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static __global__ void
magma_weight_3d_kernel(const T *dqweight1d, T *dV, const int v_stride, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

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
static __global__ void
magma_weight_generic_kernel( 
    const int dim, const int pre_org, const int post_org, 
    const T *dqweight1d, 
    T *dV, const int vstride)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)
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

#endif    // CEED_MAGMA_WEIGHT_H
