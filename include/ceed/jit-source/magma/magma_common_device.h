// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_COMMON_DEVICE_H
#define CEED_MAGMA_COMMON_DEVICE_H

#ifdef CEED_MAGMA_USE_HIP
#define MAGMA_DEVICE_SHARED(type, name) HIP_DYNAMIC_SHARED(type, name)
#else
#define MAGMA_DEVICE_SHARED(type, name) extern __shared__ type name[];
#endif

typedef enum { MagmaNoTrans = 111, MagmaTrans = 112, MagmaConjTrans = 113, Magma_ConjTrans = MagmaConjTrans } magma_trans_t;

#define MAGMA_MAXTHREADS_1D 128
#define MAGMA_MAXTHREADS_2D 128
#define MAGMA_MAXTHREADS_3D 64
// Define macro for determining number of threads in y-direction
// for basis kernels
#define MAGMA_BASIS_NTCOL(x, maxt) (((maxt) < (x)) ? 1 : ((maxt) / (x)))
// Define macro for computing the total threads in a block
// for use with __launch_bounds__()
#define MAGMA_BASIS_BOUNDS(x, maxt) (x * MAGMA_BASIS_NTCOL(x, maxt))

#define MAGMA_D_ZERO 0.0
#define MAGMA_D_ONE 1.0

//////////////////////////////////////////////////////////////////////////////////////////
// read U or V of a 1D element into shared memory sU[][] or sV[][] --  for all components
// the devptr is assumed to point directly to the element
// must sync after call
template <typename T, int LENGTH, int NCOMP_>
__device__ __inline__ void read_1d(const T* devptr, const int compstride, T* sBuffer[NCOMP_], const int tx) {
  if (tx < LENGTH) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      sBuffer[icomp][tx] = devptr[icomp * compstride + tx];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// write V of a 1D element into global memory from sV[][] --  for all components
// the devptr is assumed to point directly to the element
template <typename T, int LENGTH, int NCOMP_>
__device__ __inline__ void write_1d(T* sBuffer[NCOMP_], T* devptr, const int compstride, const int tx) {
  if (tx < LENGTH) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      devptr[icomp * compstride + tx] = sBuffer[icomp][tx];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read U of a 2D element into registers rU[][][] --  for all components of a single dim
// dU is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rU[DIMU][NCOMP_][rUsize]
// iDIM specifies which dimension is being read into in rU
// rUsize can be different from P_ (e.g. MAXP_Q)
// sTmp is a shared memory workspace of size P_^2
template <typename T, int P_, int DIMU, int NCOMP_, int rUsize, int iDIM>
__device__ __inline__ void readU_2d(const T* dU, const int compstride, T rU[DIMU][NCOMP_][rUsize], T* sTmp, const int tx) {
  // read U as a batch P_ of (1xP_) vectors
  // vec 0  : [u0, u1, u2, ... u_(P_-1)] -- contiguous in memory
  // vec 1  : [u0, u1, u2, ... u_(P_-1)] -- contiguous in memory
  // ...
  // vec P_-1: [u0, u1, u2, ... u_(P_-1)] -- contiguous in memory
  // threads collaboratively read vec0 and then vec1 and so on
  // but for the kernel, we want
  // thread 0 to hold all of vec0 in registers, and
  // thread 1 to hold all of vec1 in registers, and and so on
  // so we need to transpose
  for (int icomp = 0; icomp < NCOMP_; icomp++) {
    // read from global memory into shared memory
    if (tx < P_) {
      for (int i = 0; i < P_; i++) {
        sTmp[i * P_ + tx] = dU[icomp * compstride + i * P_ + tx];
      }
    }
    __syncthreads();

    if (tx < P_) {
      for (int i = 0; i < P_; i++) {
        rU[iDIM][icomp][i] = sTmp[tx * P_ + i];
      }
    }
    __syncthreads();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read V of a 2D element into registers rV[][][] --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIMV][NCOMP_][rVsize]
// iDIM specifies which dimension is being read into in rV
// rVsize can be different from P_ (e.g. MAXP_Q)
template <typename T, int Q_, int DIMV, int NCOMP_, int rVsize, int iDIM>
__device__ __inline__ void readV_2d(const T* dV, const int compstride, T rV[DIMV][NCOMP_][rVsize], const int tx) {
  if (tx < Q_) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      for (int j = 0; j < Q_; j++) {
        rV[iDIM][icomp][j] = dV[icomp * compstride + j * Q_ + tx];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// write V of a 2D element from registers rV[][][] to global memory --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIMV][NCOMP_][rVsize]
// iDIM specifies which dimension is being read from in rV
// idim specifies which dimension is being written to in dV
// rVsize can be different from P_ (e.g. MAXP_Q)
template <typename T, int Q_, int DIMV, int NCOMP_, int rVsize, int iDIM>
__device__ __inline__ void writeV_2d(T* dV, const int compstride, T rV[DIMV][NCOMP_][rVsize], const int tx) {
  if (tx < Q_) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      for (int j = 0; j < Q_; j++) {
        dV[icomp * compstride + j * Q_ + tx] = rV[iDIM][icomp][j];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read U of a 3D element into registers rU[][][] --  for all components of a single dim
// dU is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rU[DIMU][NCOMP_][rUsize]
// iDIM specifies which dimension is being read into in rU
// rUsize can be different from P_ (e.g. MAXP_Q)
// sTmp is a shared memory workspace of size P_^3
template <typename T, int P_, int DIMU, int NCOMP_, int rUsize, int iDIM>
__device__ __inline__ void readU_3d(const T* dU, const int compstride, T rU[DIMU][NCOMP_][rUsize], T* sTmp, const int tx) {
  // read U as a batch P_^2 of (1xP_) vectors
  // vec 0    : [u0, u1, u2, ... u_(P_-1)] -- contiguous in memory
  // vec 1    : [u0, u1, u2, ... u_(P_-1)] -- contiguous in memory
  // ...
  // vec P_^2-1: [u0, u1, u2, ... u_(P_-1)] -- contiguous in memory
  // threads collaboratively read vec0 and then vec1 and so on
  // but for the kernel, we want
  // thread 0 to hold all of vec0 in registers, and
  // thread 1 to hold all of vec1 in registers, and and so on
  // so we need to transpose
  for (int icomp = 0; icomp < NCOMP_; icomp++) {
    // read from global memory into shared memory
    if (tx < P_ * P_) {
      for (int i = 0; i < P_; i++) {
        sTmp[i * P_ * P_ + tx] = dU[icomp * compstride + i * P_ * P_ + tx];
      }
    }
    __syncthreads();

    if (tx < P_ * P_) {
      for (int i = 0; i < P_; i++) {
        rU[iDIM][icomp][i] = sTmp[tx * P_ + i];
      }
    }
    __syncthreads();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read V of a 3D element into registers rV[][][] --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIMV][NCOMP_][rVsize]
// iDIM specifies which dimension is being read into in rV
// rVsize can be different from P_ (e.g. MAXP_Q)
template <typename T, int Q_, int DIMV, int NCOMP_, int rVsize, int iDIM>
__device__ __inline__ void readV_3d(const T* dV, const int compstride, T rV[DIMV][NCOMP_][rVsize], const int tx) {
  if (tx < Q_ * Q_) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      for (int j = 0; j < Q_; j++) {
        rV[iDIM][icomp][j] = dV[icomp * compstride + j * (Q_ * Q_) + tx];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// write V of a 3D element from registers rV[][][] to global memory --  for all components of a single dim
// dV is assumed to point directly to the element (i.e. already offset by elem-stride)
// register is assumed to be rV[DIMV][NCOMP_][rVsize]
// iDIM specifies which dimension is being read from in rV
// idim specifies which dimension is being written to in dV
// rVsize can be different from P_ (e.g. MAXP_Q)
template <typename T, int Q_, int DIMV, int NCOMP_, int rVsize, int iDIM>
__device__ __inline__ void writeV_3d(T* dV, const int compstride, T rV[DIMV][NCOMP_][rVsize], const int tx) {
  if (tx < (Q_ * Q_)) {
    for (int icomp = 0; icomp < NCOMP_; icomp++) {
      for (int j = 0; j < Q_; j++) {
        dV[icomp * compstride + j * (Q_ * Q_) + tx] = rV[iDIM][icomp][j];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// reads T into shared memory
// must sync after call
template <int B, int J>
__device__ __inline__ void dread_T_gm2sm(const int tx, const magma_trans_t transT, const CeedScalar* dT, CeedScalar* sT) {
  if (transT == MagmaNoTrans) {
    // T is B x J
    if (tx < B) {
      for (int i = 0; i < J; i++) {
        sT[i * B + tx] = dT[i * B + tx];
      }
    }
  } else {
    // T is J x B
    if (tx < J) {
      for (int i = 0; i < B; i++) {
        sT[tx * B + i] = dT[i * J + tx];
      }
    }
  }
  // must sync after call
}

//////////////////////////////////////////////////////////////////////////////////////////
// reads a slice of U from shared/global memory into registers
// the correct pointer U must be precomputed
template <int B>
__device__ __inline__ void dread_U_gsm2reg(const int C, const int tx_, const CeedScalar* U, CeedScalar rU[B]) {
  for (int i = 0; i < B; i++) {
    rU[i] = U[i * C + tx_];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// reads a slice of V from shared/global memory into registers with scaling
// the correct pointer V must be precomputed
template <int J>
__device__ __inline__ void dread_V_gsm2reg(const int C, const int tx_, const CeedScalar* V, CeedScalar rV[J]) {
  for (int i = 0; i < J; i++) {
    rV[i] = V[i * C + tx_];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// writes a slice of V from reg to shared/global memory
// the correct pointer V must be precomputed
template <int J>
__device__ __inline__ void dwrite_V_reg2gsm(const int C, const int tx_, CeedScalar rV[J], CeedScalar* V) {
  for (int i = 0; i < J; i++) {
    V[i * C + tx_] = rV[i];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// multiply a slice of U times T to produce a slice of V
template <int B, int J>
__device__ __inline__ void dgemm_slice(CeedScalar alpha, CeedScalar* sT, CeedScalar rU[B], CeedScalar beta, CeedScalar rV[J]) {
  CeedScalar rTmp;
  for (int j = 0; j < J; j++) {
    rTmp = 0.0;
    for (int b = 0; b < B; b++) {
      rTmp += rU[b] * sT[j * B + b];
    }
    rV[j] *= beta;
    rV[j] += alpha * rTmp;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
template <int B, int J>
__device__ __inline__ void dgemm_ceed_device(const int tx, const int A, const int C, magma_trans_t transT, CeedScalar* sT, const CeedScalar alpha,
                                             const CeedScalar beta, const CeedScalar* dU, CeedScalar* dV, CeedScalar rU[B], CeedScalar rV[J]) {
  const int tx_      = tx % C;
  const int slice_id = tx / C;

  // advance pointers for U and V
  dU += slice_id * C * B;
  dV += slice_id * C * J;

  // read V if beta is non-zero
  if (beta != 0.0) {
    dread_V_gsm2reg<J>(C, tx_, (const CeedScalar*)dV, rV);
  }

  // read U
  dread_U_gsm2reg<B>(C, tx_, dU, rU);

  // multiply
  dgemm_slice<B, J>(alpha, sT, rU, beta, rV);

  // write V back
  dwrite_V_reg2gsm<J>(C, tx_, rV, dV);
}

#endif  // CEED_MAGMA_COMMON_DEVICE_H
