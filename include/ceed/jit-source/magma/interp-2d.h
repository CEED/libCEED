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
// interp basis action (2D)
template <typename T, int DIM_U, int DIM_V, int NCOMP_, int P_, int Q_, int rUsize, int rVsize>
static __device__ __inline__ void magma_interp_2d_device(const T* sT, magma_trans_t transT, T rU[DIM_U][NCOMP_][rUsize], T rV[DIM_V][NCOMP_][rVsize],
                                                         const int tx, T rTmp, T* swork) {
  // Assumptions
  // 1. 1D threads of size max(P_,Q_)
  // 2. input:  rU[DIM_U x NCOMP_ x rUsize] in registers (per thread)
  // 3. output: rV[DIM_V x NCOMP_ x rVsize] in registers (per thread)
  // 4. Two products per component
  //  4.1 Batch P_ of (1xP_) matrices times (P_xQ_) matrix => Batch P_ of (1xQ_) matrices
  //  4.2 Batch 1 of (Q_xP_) matrix   times (P_xQ_) matrix => (Q_xQ_) matrix
  // 5. Each thread computes one row of the output of each product
  // 6. Sync is recommended before and after the call

  for (int icomp = 0; icomp < NCOMP_; icomp++) {
    // 1st product -- Batch P_ of (1xP_) matrices [reg] x (P_xQ_) [shmem] => Batch P_ of (1xQ_) matrices
    // the batch output P_ x (1xQ_) is written on the fly to shmem
    if (tx < P_) {
      const int batchid = tx;
      const int sld     = 1;
      T*        sTmp    = swork + batchid * (1 * Q_);
      for (int j = 0; j < Q_; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P_; i++) {
          rTmp += rU[0][icomp][i] * sT(i, j);
        }
        sTmp(0, j, sld) = rTmp;
      }
    }  // end of: if (tx < P_)
    __syncthreads();

    // 2nd product -- Batch 1 of a (Q_xP_) matrix [shmem] x (P_xQ_) [shmem] => (Q_xQ_) matrix [reg]
    if (tx < Q_) {
      const int batchid = 0;
      const int sld     = Q_;
      T*        sTmp    = swork + batchid * (Q_ * P_);
      for (int j = 0; j < Q_; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P_; i++) {
          rTmp += sTmp(tx, i, sld) * sT(i, j);
        }
        rV[0][icomp][j] += rTmp;
      }
    }
    __syncthreads();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__
    void magma_interpn_2d_kernel(const CeedScalar* dT, const CeedScalar* dU, const int estrdU, const int cstrdU, CeedScalar* dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaNoTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][NCOMP][P] = {0.0};  // for a non fused operator DIM is always 1
  CeedScalar rV[1][NCOMP][Q] = {0.0};  // for a non fused operator DIM is always 1
  CeedScalar rTmp            = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar* sT   = (CeedScalar*)(shared_data);
  CeedScalar* sTmp = sT + P * Q;
  sTmp += ty * (P * MAXPQ);

  // read T
  if (ty == 0) {
    dread_T_gm2sm<P, Q>(tx, transT, dT, sT);
  }

  // read U -- there is a sync at the end of this function
  readU_2d<CeedScalar, P, 1, NCOMP, P, 0>(dU, cstrdU, rU, sTmp, tx);

  // no sync needed here -- readU_2d already syncs at the end
  magma_interp_2d_device<CeedScalar, 1, 1, NCOMP, P, Q, P, Q>(sT, transT, rU, rV, tx, rTmp, sTmp);
  __syncthreads();

  // write V
  writeV_2d<CeedScalar, Q, 1, NCOMP, Q, 0>(dV, cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_2D)) __global__
    void magma_interpt_2d_kernel(const CeedScalar* dT, const CeedScalar* dU, const int estrdU, const int cstrdU, CeedScalar* dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][NCOMP][Q] = {0.0};  // for a non fused operator DIM is always 1
  CeedScalar rV[1][NCOMP][P] = {0.0};  // for a non fused operator DIM is always 1
  CeedScalar rTmp            = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar* sT   = (CeedScalar*)(shared_data);
  CeedScalar* sTmp = sT + Q * P;
  sTmp += ty * (Q * MAXPQ);

  // read T
  if (ty == 0) {
    dread_T_gm2sm<Q, P>(tx, transT, dT, sT);
  }

  // read V
  readV_2d<CeedScalar, P, 1, NCOMP, P, 0>(dV, cstrdV, rV, tx);

  // read U -- there is a sync at the end of this function
  readU_2d<CeedScalar, Q, 1, NCOMP, Q, 0>(dU, cstrdU, rU, sTmp, tx);

  // no sync needed here -- readU_2d already syncs at the end
  magma_interp_2d_device<CeedScalar, 1, 1, NCOMP, Q, P, Q, P>(sT, transT, rU, rV, tx, rTmp, sTmp);
  __syncthreads();

  // write V
  writeV_2d<CeedScalar, P, 1, NCOMP, P, 0>(dV, cstrdV, rV, tx);
}
