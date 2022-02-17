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

#ifndef CEED_MAGMA_GRAD_H
#define CEED_MAGMA_GRAD_H

#include <ceed/ceed.h>
#include <magma_v2.h>
#include "magma_common_device.h"
#include "grad_device.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int DIM, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_1D)) __global__ void
magma_grad_1d_kernel(  
    const T *dTgrad, magma_trans_t transT,
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
        dread_T_gm2sm<P, Q>(tx, transT, dTgrad, sT);
    }

    // read U
    read_1d<T, P, NCOMP>(dU, cstrdU, sU, tx);

    // read V if transT is magmaTrans
    if (transT == MagmaTrans) {
        read_1d<T, Q, NCOMP>(dV, cstrdV, sV, tx);
    }

    __syncthreads();
    magma_grad_1d_device<T, DIM, NCOMP, P, Q>(sT, transT, sU, sV, tx);
    __syncthreads();

    // write V
    write_1d<T, Q, NCOMP>(sV, dV, cstrdV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__ void
magma_gradn_2d_kernel(
    const T *dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, const int dstrdU,  
          T *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem)
{

    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T rU[1][NCOMP][P] = { make_zero<T>() };  // here DIMU = 1, but might be different for a fused operator
    T rV[1][NCOMP][Q] = { make_zero<T>() };  // here DIMV = 1, but might be different for a fused operator
    T rTmp = make_zero<T>();

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    T* sTinterp = (T*)(shared_data);
    T* sTgrad   = sTinterp + P*Q;
    T* sTmp     = sTgrad   + P*Q;
    sTmp       += ty * (P * MAXPQ);

    // read T
    if (ty == 0) {
        dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
        dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
    }

    // No need to read V ( required only in transposed grad )
    const T beta = make_zero<T>();

    /* read U (idim = 0 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_2d<T, P, 1, NCOMP, P, 0>
    (dU + (0*dstrdU), cstrdU, rU, sTmp, tx);

    /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) -- 
       output from rV[0][][] into dV (idim = 0) */
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp); 
    /* there is a sync at the end of magma_grad_2d_device */
    writeV_2d<T, Q, 1, NCOMP, Q, 0>
    (dV+(0*dstrdV), cstrdV, rV, tx);

    /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) -- 
    output from rV[0][][] into dV (idim = 1) */
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_2d_device */
    writeV_2d<T, Q, 1, NCOMP, Q, 0>
    (dV+(1*dstrdV), cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__ void
magma_gradt_2d_kernel(
    const T *dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, const int dstrdU,  
          T *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    T rU[1][NCOMP][P] = { make_zero<T>() };  // here DIMU = 1, but might be different for a fused operator
    T rV[1][NCOMP][Q] = { make_zero<T>() };  // here DIMV = 1, but might be different for a fused operator
    T rTmp = make_zero<T>();

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    T* sTinterp = (T*)(shared_data);
    T* sTgrad   = sTinterp + P*Q;
    T* sTmp     = sTgrad   + P*Q;
    sTmp       += ty * (P*MAXPQ);

    // read T
    if (ty == 0) {
        dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
        dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
    }
    __syncthreads();

    /* read V (since this is transposed mode -- 
       idim = 0 for dV, iDIM = 0 for rV) */
    const T beta = make_one<T>();
    readV_2d<T, Q, 1, NCOMP, Q, 0>
    (dV + (0*dstrdV), cstrdV, rV, tx);

    /* read U (idim = 0 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_2d<T, P, 1, NCOMP, P, 0>
    (dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);
    /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) */
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_2d_device */

    /* read U (idim = 1 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_2d<T, P, 1, NCOMP, P, 0>
    (dU + (1*dstrdU), cstrdU, rU, sTmp, tx);
    /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) */
    magma_grad_2d_device<T, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_2d_device */

    // write V
    writeV_2d<T, Q, 1, NCOMP, Q, 0>
    (dV + (0*dstrdV), cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename TC, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ*MAXPQ, MAGMA_MAXTHREADS_3D)) __global__ void
magma_gradn_3d_kernel(
    const TC* dinterp1d, const TC* dgrad1d, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, const int dstrdU,  
          T *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    TC rU[1][NCOMP][P] = {make_zero<TC>()};  // here DIMU = 1, but might be different for a fused operator
    TC rV[1][NCOMP][Q] = {make_zero<TC>()};  // here DIMV = 1, but might be different for a fused operator
    TC rTmp = make_zero<TC>();

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    TC* sTinterp = (TC*)(shared_data);
    TC* sTgrad   = sTinterp + P*Q;
    TC* sTmp     = sTgrad   + P*Q;
    sTmp       += ty * (max(P*P*P, (P*P*Q) + (P*Q*Q)));

    // read T
    if (ty == 0) {
        dread_TC_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
        dread_TC_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
    }
    __syncthreads();

    // No need to read V ( required only in transposed grad )
    const TC beta = make_zero<T>();

    /* read U (idim = 0 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_3d<T, TC, P, 1, NCOMP, P, 0>
    (dU + (0*dstrdU), cstrdU, rU, sTmp, tx);

    /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) -- 
       output from rV[0][][] into dV (idim = 0) */
    magma_grad_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_3d_device */
    writeV_3d<T, TC, Q, 1, NCOMP, Q, 0>
    (dV+ (0*dstrdV), cstrdV, rV, tx);

    /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) -- 
       output from rV[0][][] into dV (idim = 1) */
    magma_grad_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_3d_device */
    writeV_3d<T, TC, Q, 1, NCOMP, Q, 0>
    (dV+ (1*dstrdV), cstrdV, rV, tx); 

    /* third call (iDIM = 2, iDIMU = 0, iDIMV = 0) -- 
       output from rV[0][][] into dV (idim = 2) */
    magma_grad_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q, 2, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_3d_device */
    writeV_3d<T, TC, Q, 1, NCOMP, Q, 0>
    (dV+ (2*dstrdV), cstrdV, rV, tx); 
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename TC, int NCOMP, int P, int Q, int MAXPQ>
static __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ*MAXPQ, MAGMA_MAXTHREADS_3D)) __global__ void
magma_gradt_3d_kernel(
    const TC *dinterp1d, const TC *dgrad1d, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, const int dstrdU,  
          T *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem)
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int elem_id = (blockIdx.x * blockDim.y) + ty;

    if (elem_id >= nelem) return;

    TC rU[1][NCOMP][P] = { make_zero<TC>() };  // here DIMU = 1, but might be different for a fused operator
    TC rV[1][NCOMP][Q] = { make_zero<TC>() };  // here DIMV = 1, but might be different for a fused operator
    TC rTmp = make_zero<TC>();

    // shift global memory pointers by elem stride
    dU += elem_id * estrdU;
    dV += elem_id * estrdV;

    // assign shared memory pointers
    TC* sTinterp = (TC*)(shared_data);
    TC* sTgrad   = sTinterp + P*Q;
    TC* sTmp     = sTgrad   + P*Q;
    sTmp       += ty * (max(P*P*P, (P*P*Q) + (P*Q*Q)));

    // read T
    if (ty == 0) {
        dread_TC_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
        dread_TC_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
    }
    __syncthreads();

    // read V (since this is transposed mode)
    const TC beta = make_one<T>();
    readV_3d<T, TC, Q, 1, NCOMP, Q, 0>
    (dV + (0*dstrdV), cstrdV, rV, tx);

    /* read U (idim = 0 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_3d<T, TC, P, 1, NCOMP, P, 0>
    (dU + (0 * dstrdU), cstrdU, rU, sTmp, tx); 
    /* then first call (iDIM = 0, iDIMU = 0, iDIMV = 0) */
    magma_grad_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_3d_device */

    /* read U (idim = 1 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_3d<T, TC, P, 1, NCOMP, P, 0>
    (dU + (1 * dstrdU), cstrdU, rU, sTmp, tx); 
    /* then second call (iDIM = 1, iDIMU = 0, iDIMV = 0) */
    magma_grad_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_3d_device */

    /* read U (idim = 2 for dU, iDIM = 0 for rU) -- 
       there is a sync at the end of this function */
    readU_3d<T, TC, P, 1, NCOMP, P, 0>
    (dU + (2 * dstrdU), cstrdU, rU, sTmp, tx); 
    /* then third call (iDIM = 2, iDIMU = 0, iDIMV = 0) */
    magma_grad_3d_device<TC, 1, 1, NCOMP, P, Q, P, Q, 2, 0, 0>
    (sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
    /* there is a sync at the end of magma_grad_3d_device */

    // write V 
    writeV_3d<T, TC, Q, 1, NCOMP, Q, 0>
    (dV + (0 * dstrdV), cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int P, int Q>
static __global__ void
magma_grad_generic_kernel( 
    const int dim, const int ncomp, 
    const int pre_org, const int tmp_size, 
    const T* dinterp1d, const T *dgrad1d, magma_trans_t transT,
    const T *dU, const int estrdU, const int cstrdU, 
          T *dV, const int estrdV, const int cstrdV, 
    const int dim_id )
{
    MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

    const int elem_id = blockIdx.x;
    const int comp_id = blockIdx.y;
    int tx = threadIdx.x;
    int pre, post;
    
    // advance to the respective element in the batch
    dU += (elem_id * estrdU) + (comp_id * cstrdU);
    dV += (elem_id * estrdV) + (comp_id * cstrdV);

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

#endif    // CEED_MAGMA_GRAD_H
