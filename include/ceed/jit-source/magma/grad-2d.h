// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j)*P_ + (i)]
#define sTmp(i, j, ldw) sTmp[(j) * (ldw) + (i)]

//////////////////////////////////////////////////////////////////////////////////////////
// grad basis action (2D)
// This function is called two times at a higher level for 2D
// DIM_U  -- for the size of rU[DIM_U * NCOMP_ * MAXP_Q_]
// DIM_V  -- for the size of rV[DIM_V * NCOMP_ * MAXP_Q_]
// iDIM_  -- the index of the outermost loop over dimensions in grad
// iDIM_U -- which dim index of rU is accessed (always 0 for notrans, 0 or 1 for trans)
// iDIM_V -- which dim index of rV is accessed (0 or 1 for notrans, always 0 for trans)
// the scalar beta is used to specify whether to accumulate to rV, or overwrite it
template <typename T, int DIM_U, int DIM_V, int NCOMP_, int P_, int Q_, int rUsize, int rVsize, int iDIM_, int iDIM_U, int iDIM_V>
static __device__ __inline__ void magma_grad_2d_device(const T *sTinterp, const T *sTgrad, T rU[DIM_U][NCOMP_][rUsize], T rV[DIM_V][NCOMP_][rVsize],
                                                       T beta, const int tx, T rTmp, T *swork) {
  // Assumptions
  // 0. This device routine applies grad for one dim only (iDIM_), so it should be called twice for 2D
  // 1. 1D threads of size max(P_,Q_)
  // 2. input:  rU[DIM_U x NCOMP_ x P_] in registers (per thread)
  // 3. output: rV[DIM_V x NCOMP_ x Q_] in registers (per thread)
  // 4. Two products per each (dim,component) pair
  //  4.1 Batch P_ of (1xP_) matrices times (P_xQ_) matrix => Batch P_ of (1xQ_) matrices
  //  4.2 Batch 1 of (Q_xP_) matrix   times (P_xQ_) matrix => (Q_xQ_) matrix
  // 6. Each thread computes one row of the output of each product
  // 7. Sync is recommended before and after the call

  for (int icomp = 0; icomp < NCOMP_; icomp++) {
    // 1st product -- Batch P_ of (1xP_) matrices [reg] x (P_xQ_) [shmem] => Batch P_ of (1xQ_) matrices
    // the batch output P_ x (1xQ_) is written on the fly to shmem
    if (tx < P_) {
      const int batchid = tx;
      const int sld     = 1;
      const T  *sT      = (iDIM_ == 0) ? sTgrad : sTinterp;
      T        *sTmp    = swork + batchid * (1 * Q_);
      for (int j = 0; j < Q_; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P_; i++) {
          rTmp += rU[iDIM_U][icomp][i] * sT(i, j);
        }
        sTmp(0, j, sld) = rTmp;
      }
    }  // end of: if (tx < P_)
    __syncthreads();

    // 2nd product -- Batch 1 of a (Q_xP_) matrix [shmem] x (P_xQ_) [shmem] => (Q_xQ_) matrix [reg]
    if (tx < Q_) {
      const int batchid = 0;
      const int sld     = Q_;
      const T  *sT      = (iDIM_ == 1) ? sTgrad : sTinterp;
      T        *sTmp    = swork + batchid * (Q_ * P_);
      for (int j = 0; j < Q_; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P_; i++) {
          rTmp += sTmp(tx, i, sld) * sT(i, j);
        }
        rV[iDIM_V][icomp][j] *= beta;
        rV[iDIM_V][icomp][j] += rTmp;
      }
    }
    __syncthreads();
  }  // loop over NCOMP_
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__
    void magma_gradn_2d_kernel(const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, const CeedScalar *dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaNoTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][NCOMP][P] = {0.0};  // here DIMU = 1, but might be different for a fused operator
  CeedScalar rV[1][NCOMP][Q] = {0.0};  // here DIMV = 1, but might be different for a fused operator
  CeedScalar rTmp            = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sTinterp = (CeedScalar *)(shared_data);
  CeedScalar *sTgrad   = sTinterp + P * Q;
  CeedScalar *sTmp     = sTgrad + P * Q;
  sTmp += ty * (P * MAXPQ);

  // read T
  if (ty == 0) {
    dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
    dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
  }

  // No need to read V ( required only in transposed grad )
  const CeedScalar beta = 0.0;

  /* read U (idim = 0 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_2d<CeedScalar, P, 1, NCOMP, P, 0>(dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);

  /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) --
     output from rV[0][][] into dV (idim = 0) */
  magma_grad_2d_device<CeedScalar, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_2d_device */
  writeV_2d<CeedScalar, Q, 1, NCOMP, Q, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);

  /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) --
  output from rV[0][][] into dV (idim = 1) */
  magma_grad_2d_device<CeedScalar, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_2d_device */
  writeV_2d<CeedScalar, Q, 1, NCOMP, Q, 0>(dV + (1 * dstrdV), cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__
    void magma_gradt_2d_kernel(const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, const CeedScalar *dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][NCOMP][Q] = {0.0};  // here DIMU = 1, but might be different for a fused operator
  CeedScalar rV[1][NCOMP][P] = {0.0};  // here DIMV = 1, but might be different for a fused operator
  CeedScalar rTmp            = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sTinterp = (CeedScalar *)(shared_data);
  CeedScalar *sTgrad   = sTinterp + Q * P;
  CeedScalar *sTmp     = sTgrad + Q * P;
  sTmp += ty * (Q * MAXPQ);

  // read T
  if (ty == 0) {
    dread_T_gm2sm<Q, P>(tx, transT, dinterp1d, sTinterp);
    dread_T_gm2sm<Q, P>(tx, transT, dgrad1d, sTgrad);
  }
  __syncthreads();

  /* read V (since this is transposed mode --
     idim = 0 for dV, iDIM = 0 for rV) */
  const CeedScalar beta = 1.0;
  readV_2d<CeedScalar, P, 1, NCOMP, P, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);

  /* read U (idim = 0 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_2d<CeedScalar, Q, 1, NCOMP, Q, 0>(dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);
  /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) */
  magma_grad_2d_device<CeedScalar, 1, 1, NCOMP, Q, P, Q, P, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_2d_device */

  /* read U (idim = 1 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_2d<CeedScalar, Q, 1, NCOMP, Q, 0>(dU + (1 * dstrdU), cstrdU, rU, sTmp, tx);
  /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) */
  magma_grad_2d_device<CeedScalar, 1, 1, NCOMP, Q, P, Q, P, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_2d_device */

  // write V
  writeV_2d<CeedScalar, P, 1, NCOMP, P, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);
}
