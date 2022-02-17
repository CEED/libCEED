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

#ifndef CEED_MAGMA_INTERP_H
#define CEED_MAGMA_INTERP_H

#include <ceed/ceed.h>
#include <magma_v2.h>
#include "magma_common_device.h"
#include "interp_device.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int DIM, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_1D)) __global__ void
magma_interp_1d_kernel(  
    const T *dT, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, 
          T *dV, const int estrdV, const int cstrdV, const int nelem)
{

    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T* sU[NCOMP];
    T* sV[NCOMP];

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    T* sT = (T*)(shared_data);
    T* sW = sT + P*Q;
    sU[0] = sW + ty * NCOMP * (P + Q);
    sV[0] = sU[0] + (NCOMP * 1 * P);
    for(int icomp = 1; icomp < NCOMP; icomp++) {
        sU[icomp] = sU[icomp-1] + (1 * P);
        sV[icomp] = sV[icomp-1] + (1 * Q);
    }

    // read T
    if (ty == 0) {
        dread_T_gm2sm<P, Q>(tx, transT, dT, sT);
    }
    
    // read U
    read_1d<T, P, NCOMP>(dU, cstrdU, sU, tx);

    // read V if transT is magmaTrans
    if (transT == MagmaTrans) {
        read_1d<T, Q, NCOMP>(dV, cstrdV, sV, tx);
    }

    __syncthreads();
    magma_interp_1d_device<T, DIM, NCOMP, P, Q>(sT, transT, sU, sV, tx);
    __syncthreads();

    // write V
    write_1d<T, Q, NCOMP>(sV, dV, cstrdV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__ void
magma_interp_2d_kernel(
    const T *dT, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, 
          T *dV, const int estrdV, const int cstrdV, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T rU[1][NCOMP][P] = { make_zero<T>() };    // for a non fused operator DIM is always 1
    T rV[1][NCOMP][Q] = { make_zero<T>() };    // for a non fused operator DIM is always 1
    T rTmp = make_zero<T>();

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    T* sT    = (T*)(shared_data);
    T* sTmp  = sT + P*Q;
    sTmp    += ty * (P * MAXPQ);

    // read T
    if (ty == 0) {
        dread_T_gm2sm<P, Q>(tx, transT, dT, sT);
    }

    // read V if transT is magmaTrans
    if (transT == MagmaTrans) {
        readV_2d<T, Q, 1, NCOMP, Q, 0>(dV, cstrdV, rV, tx);
    }

    // read U -- there is a sync at the end of this function
    readU_2d<T, P, 1, NCOMP, P, 0>(dU, cstrdU, rU, sTmp, tx);

    // no sync needed here -- readU_2d already syncs at the end
    magma_interp_2d_device<T, 1, 1, NCOMP, P, Q, P, Q>(sT, transT, rU, rV, tx, rTmp, sTmp);
    __syncthreads();

    // write V
    writeV_2d<T, Q, 1, NCOMP, Q, 0>(dV, cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename TC, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ*MAXPQ, MAGMA_MAXTHREADS_3D)) __global__ void
magma_interp_3d_kernel(
    const TC *dT, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, 
          T *dV, const int estrdV, const int cstrdV, const int nelem)
{
    MAGMA_DEVICE_SHARED( CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    TC rU[1][NCOMP][P] = { make_zero<TC>() };    // for a non fused operator DIM is always 1
    TC rV[1][NCOMP][Q] = { make_zero<TC>() };    // for a non fused operator DIM is always 1
    TC rTmp[Q] = { make_zero<TC>() };

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    TC* sT    = (TC*)(shared_data);
    TC* sTmp  = sT + P*Q;
    sTmp    += ty * (max(P*P*MAXPQ, P*Q*Q));

    // read T
    if (ty == 0) {
        dread_TC_gm2sm<P, Q>(tx, transT, dT, sT);
    }

    // read V if transT is magmaTrans
    if (transT == MagmaTrans) {
        readV_3d<T, TC, Q, 1, NCOMP, Q, 0>(dV, cstrdV, rV, tx);
    }

    // read U (idim = 0 for dU, iDIM = 0 for rU, u_dimstride is always 0)
    readU_3d<T, TC, P, 1, NCOMP, P, 0>(dU, cstrdU, rU, sTmp, tx);
    // there is a sync at the end of this function

    magma_interp_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q>(sT, transT, rU , rV, tx, rTmp, sTmp);
    __syncthreads();

    // write V
    writeV_3d<T, TC, Q, 1, NCOMP, Q, 0>(dV, cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int P, int Q>
static __global__ void
interp_generic_kernel( 
    const int dim, const int ncomp, const int pre_org, const int post_org, const int tmp_size, 
    const T *dT, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, 
          T *dV, const int estrdV, const int cstrdV)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int elem_id = blockIdx.x; 
    const int comp_id = blockIdx.y;
    magma_interp_generic_device< P, Q >
    ( dim, ncomp, pre_org, post_org, tmp_size, dT, transT, 
      dU + (elem_id * estrdU) + (comp_id * cstrdU), 
      dV + (elem_id * estrdV) + (comp_id * cstrdV), 
      shared_data );
}

#endif    // CEED_MAGMA_INTERP_H
