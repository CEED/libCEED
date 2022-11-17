// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j)*P_ + (i)]
#define sTmp(i, j, ldw) sTmp[(j) * (ldw) + (i)]
#define sTmp2(i, j, ldw) sTmp2[(j) * (ldw) + (i)]

//////////////////////////////////////////////////////////////////////////////////////////
// grad basis action (3D)
// This function is called three times at a higher level for 3D
// DIM_U  -- for the size of rU[DIM_U * NCOMP_ * MAXP_Q_]
// DIM_V  -- for the size of rV[DIM_V * NCOMP_ * MAXP_Q_]
// iDIM_  -- the index of the outermost loop over dimensions in grad
// iDIM_U -- which dim index of rU is accessed (always 0 for notrans, 0, 1, or 2 for trans)
// iDIM_V -- which dim index of rV is accessed (0, 1, or 2 for notrans, always 0 for trans)
// the scalar beta is used to specify whether to accumulate to rV, or overwrite it
template <typename T, int DIM_U, int DIM_V, int NCOMP_, int P_, int Q_, int rUsize, int rVsize, int iDIM_, int iDIM_U, int iDIM_V>
static __device__ __inline__ void magma_grad_3d_device(const T* sTinterp, const T* sTgrad, T rU[DIM_U][NCOMP_][rUsize], T rV[DIM_V][NCOMP_][rVsize],
                                                       T beta, const int tx, T rTmp, T* swork) {
  // Assumptions
  // 0. This device routine applies grad for one dim only (iDIM_), so it should be thrice for 3D
  // 1. 1D threads of size max(P_,Q_)^2
  // 2. input:  rU[DIM_U x NCOMP_ x rUsize] in registers (per thread)
  // 3. output: rV[DIM_V x NCOMP_ x rVsize] in registers (per thread)
  // 4. Three products per each (dim,component) pair
  //  4.1 Batch P_^2 of (1xP_) matrices times (P_xQ_) matrix => Batch P_^2 of (1xQ_) matrices
  //  4.2 Batch P_   of (Q_xP_) matrices times (P_xQ_) matrix => Batch P_   of (Q_xQ_) matrices
  //  4.3 Batch 1   of (Q_^2xP_) matrix times (P_xQ_) matrix => (Q_^2xQ_) matrix
  // 6. Each thread computes one row of the output of each product
  // 7. Sync is recommended before and after the call

  T* sW1 = swork;
  T* sW2 = sW1 + P_ * P_ * Q_;
  for (int icomp = 0; icomp < NCOMP_; icomp++) {
    // Batch P_^2 of (1xP_) matrices [reg] times (P_xQ_) matrix [shmem] => Batch P_^2 of (1xQ_) matrices [shmem]
    if (tx < (P_ * P_)) {
      const int batchid = tx;
      const int sld     = 1;
      const T*  sT      = (iDIM_ == 0) ? sTgrad : sTinterp;
      T*        sTmp    = sW1 + batchid * (1 * Q_);
      for (int j = 0; j < Q_; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P_; i++) {
          rTmp += rU[iDIM_U][icomp][i] * sT(i, j);
        }
        sTmp(0, j, sld) = rTmp;
      }
    }  // end of: if (tx < P_*P_)
    __syncthreads();

    // Batch P_ of (Q_xP_) matrices [shmem] times (P_xQ_) matrix [shmem] => Batch P_ of (Q_xQ_) matrices [reg]
    if (tx < (P_ * Q_)) {
      const int batchid = tx / Q_;
      const int tx_     = tx % Q_;
      const int sld     = Q_;
      const T*  sT      = (iDIM_ == 1) ? sTgrad : sTinterp;
      T*        sTmp    = sW1 + batchid * (Q_ * P_);  // sTmp is input
      T*        sTmp2   = sW2 + batchid * (Q_ * Q_);  // sTmp2 is output
      for (int j = 0; j < Q_; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P_; i++) {
          rTmp += sTmp(tx_, i, sld) * sT(i, j);
        }
        sTmp2(tx_, j, sld) = rTmp;
      }
    }
    __syncthreads();

    // Batch 1 of (Q_^2xP_) matrices [shmem] times (P_xQ_) matrix [shmem] => Batch 1 of (Q_^2xQ_) matrices [reg]
    if (tx < (Q_ * Q_)) {
      // No need to declare batchid = (tx  / Q_^2) = always zero
      // No need to declare tx_     = (tx_ % Q_^2) = always tx
      const int sld  = Q_ * Q_;
      const T*  sT   = (iDIM_ == 2) ? sTgrad : sTinterp;
      T*        sTmp = sW2;  // sTmp is input
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
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ* MAXPQ, MAGMA_MAXTHREADS_3D)) __global__
    void magma_gradn_3d_kernel(const CeedScalar* dinterp1d, const CeedScalar* dgrad1d, const CeedScalar* dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar* dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
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
  CeedScalar* sTinterp = (CeedScalar*)(shared_data);
  CeedScalar* sTgrad   = sTinterp + P * Q;
  CeedScalar* sTmp     = sTgrad + P * Q;
  sTmp += ty * (max(P * P * P, (P * P * Q) + (P * Q * Q)));

  // read T
  if (ty == 0) {
    dread_T_gm2sm<P, Q>(tx, transT, dinterp1d, sTinterp);
    dread_T_gm2sm<P, Q>(tx, transT, dgrad1d, sTgrad);
  }
  __syncthreads();

  // No need to read V ( required only in transposed grad )
  const CeedScalar beta = 0.0;

  /* read U (idim = 0 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_3d<CeedScalar, P, 1, NCOMP, P, 0>(dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);

  /* first call (iDIM = 0, iDIMU = 0, iDIMV = 0) --
     output from rV[0][][] into dV (idim = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, NCOMP, P, Q, P, Q, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */
  writeV_3d<CeedScalar, Q, 1, NCOMP, Q, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);

  /* second call (iDIM = 1, iDIMU = 0, iDIMV = 0) --
     output from rV[0][][] into dV (idim = 1) */
  magma_grad_3d_device<CeedScalar, 1, 1, NCOMP, P, Q, P, Q, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */
  writeV_3d<CeedScalar, Q, 1, NCOMP, Q, 0>(dV + (1 * dstrdV), cstrdV, rV, tx);

  /* third call (iDIM = 2, iDIMU = 0, iDIMV = 0) --
     output from rV[0][][] into dV (idim = 2) */
  magma_grad_3d_device<CeedScalar, 1, 1, NCOMP, P, Q, P, Q, 2, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */
  writeV_3d<CeedScalar, Q, 1, NCOMP, Q, 0>(dV + (2 * dstrdV), cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ* MAXPQ, MAGMA_MAXTHREADS_3D)) __global__
    void magma_gradt_3d_kernel(const CeedScalar* dinterp1d, const CeedScalar* dgrad1d, const CeedScalar* dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar* dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
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
  CeedScalar* sTinterp = (CeedScalar*)(shared_data);
  CeedScalar* sTgrad   = sTinterp + Q * P;
  CeedScalar* sTmp     = sTgrad + Q * P;
  sTmp += ty * (max(Q * Q * Q, (Q * Q * P) + (Q * P * P)));

  // read T
  if (ty == 0) {
    dread_T_gm2sm<Q, P>(tx, transT, dinterp1d, sTinterp);
    dread_T_gm2sm<Q, P>(tx, transT, dgrad1d, sTgrad);
  }
  __syncthreads();

  // read V (since this is transposed mode)
  const CeedScalar beta = 1.0;
  readV_3d<CeedScalar, P, 1, NCOMP, P, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);

  /* read U (idim = 0 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_3d<CeedScalar, Q, 1, NCOMP, Q, 0>(dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);
  /* then first call (iDIM = 0, iDIMU = 0, iDIMV = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, NCOMP, Q, P, Q, P, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */

  /* read U (idim = 1 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_3d<CeedScalar, Q, 1, NCOMP, Q, 0>(dU + (1 * dstrdU), cstrdU, rU, sTmp, tx);
  /* then second call (iDIM = 1, iDIMU = 0, iDIMV = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, NCOMP, Q, P, Q, P, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */

  /* read U (idim = 2 for dU, iDIM = 0 for rU) --
     there is a sync at the end of this function */
  readU_3d<CeedScalar, Q, 1, NCOMP, Q, 0>(dU + (2 * dstrdU), cstrdU, rU, sTmp, tx);
  /* then third call (iDIM = 2, iDIMU = 0, iDIMV = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, NCOMP, Q, P, Q, P, 2, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */

  // write V
  writeV_3d<CeedScalar, P, 1, NCOMP, P, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);
}
