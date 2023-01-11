// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j)*P_ + (i)]

//////////////////////////////////////////////////////////////////////////////////////////
// grad basis action (1D)
template <typename T, int DIM_, int NCOMP_, int P_, int Q_>
static __device__ __inline__ void magma_grad_1d_device(const T* sT, magma_trans_t transT, T* sU[NCOMP_], T* sV[NCOMP_], const int tx) {
  // Assumptions
  // 1. 1D threads of size max(P_,Q_)
  // 2. sU[i] is 1xP_: in shared memory
  // 3. sV[i] is 1xQ_: in shared memory
  // 4. P_roduct per component is one row (1xP_) times T matrix (P_xQ_) => one row (1xQ_)
  // 5. Each thread computes one entry in sV[i]
  // 6. Must sync before and after call
  // 7. Note that the layout for U and V is different from 2D/3D problem

  T rv;
  if (tx < Q_) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      rv = (transT == MagmaTrans) ? sV[icomp][tx] : 0.0;
      for (int i = 0; i < P_; i++) {
        rv += sU[icomp][i] * sT(i, tx);
      }
      sV[icomp][tx] = rv;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_1D)) __global__
    void magma_gradn_1d_kernel(const CeedScalar* dTinterp, const CeedScalar* dTgrad, const CeedScalar* dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar* dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaNoTrans;

  if (elem_id >= nelem) return;

  CeedScalar* sU[NCOMP];
  CeedScalar* sV[NCOMP];

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar* sT = (CeedScalar*)(shared_data);
  CeedScalar* sW = sT + P * Q;
  sU[0]          = sW + ty * NCOMP * (P + Q);
  sV[0]          = sU[0] + (NCOMP * 1 * P);
  for (int icomp = 1; icomp < NCOMP; icomp++) {
    sU[icomp] = sU[icomp - 1] + (1 * P);
    sV[icomp] = sV[icomp - 1] + (1 * Q);
  }

  // read T
  if (ty == 0) {
    dread_T_gm2sm<P, Q>(tx, transT, dTgrad, sT);
  }

  // read U
  read_1d<CeedScalar, P, NCOMP>(dU, cstrdU, sU, tx);

  __syncthreads();
  magma_grad_1d_device<CeedScalar, DIM, NCOMP, P, Q>(sT, transT, sU, sV, tx);
  __syncthreads();

  // write V
  write_1d<CeedScalar, Q, NCOMP>(sV, dV, cstrdV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(MAXPQ, MAGMA_MAXTHREADS_1D)) __global__
    void magma_gradt_1d_kernel(const CeedScalar* dTinterp, const CeedScalar* dTgrad, const CeedScalar* dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar* dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaTrans;

  if (elem_id >= nelem) return;

  CeedScalar* sU[NCOMP];
  CeedScalar* sV[NCOMP];

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar* sT = (CeedScalar*)(shared_data);
  CeedScalar* sW = sT + Q * P;
  sU[0]          = sW + ty * NCOMP * (Q + P);
  sV[0]          = sU[0] + (NCOMP * 1 * Q);
  for (int icomp = 1; icomp < NCOMP; icomp++) {
    sU[icomp] = sU[icomp - 1] + (1 * Q);
    sV[icomp] = sV[icomp - 1] + (1 * P);
  }

  // read T
  if (ty == 0) {
    dread_T_gm2sm<Q, P>(tx, transT, dTgrad, sT);
  }

  // read U
  read_1d<CeedScalar, Q, NCOMP>(dU, cstrdU, sU, tx);

  // read V
  read_1d<CeedScalar, P, NCOMP>(dV, cstrdV, sV, tx);

  __syncthreads();
  magma_grad_1d_device<CeedScalar, DIM, NCOMP, Q, P>(sT, transT, sU, sV, tx);
  __syncthreads();

  // write V
  write_1d<CeedScalar, P, NCOMP>(sV, dV, cstrdV, tx);
}
