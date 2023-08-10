// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma-gemm-nontensor.h"

#include "ceed-magma-gemm-selector.h"

#ifdef CEED_MAGMA_USE_HIP
#define devblasDgemmStridedBatched hipblasDgemmStridedBatched
#define devblasSgemmStridedBatched hipblasSgemmStridedBatched
#define magma_queue_get_devblas_handle magma_queue_get_hipblas_handle
#define devblas_trans_const hipblas_trans_const
#else
#define devblasDgemmStridedBatched cublasDgemmStridedBatched
#define devblasSgemmStridedBatched cublasSgemmStridedBatched
#define magma_queue_get_devblas_handle magma_queue_get_cublas_handle
#define devblas_trans_const cublas_trans_const
#endif

////////////////////////////////////////////////////////////////////////////////
static inline int magmablas_gemm(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                 const CeedScalar *d_A, magma_int_t ldda, const CeedScalar *d_B, magma_int_t lddb, CeedScalar beta, CeedScalar *d_C,
                                 magma_int_t lddc, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    magmablas_sgemm(trans_A, trans_B, m, n, k, (float)alpha, (const float *)d_A, ldda, (const float *)d_B, lddb, (float)beta, (float *)d_C, lddc,
                    queue);
  } else {
    magmablas_dgemm(trans_A, trans_B, m, n, k, (double)alpha, (const double *)d_A, ldda, (const double *)d_B, lddb, (double)beta, (double *)d_C, lddc,
                    queue);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
static inline int magmablas_gemm_batched_strided(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k,
                                                 CeedScalar alpha, const CeedScalar *d_A, magma_int_t ldda, magma_int_t strideA,
                                                 const CeedScalar *d_B, magma_int_t lddb, magma_int_t strideB, CeedScalar beta, CeedScalar *d_C,
                                                 magma_int_t lddc, magma_int_t strideC, magma_int_t batchCount, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    magmablas_sgemm_batched_strided(trans_A, trans_B, m, n, k, (float)alpha, (const float *)d_A, ldda, strideA, (const float *)d_B, lddb, strideB,
                                    (float)beta, (float *)d_C, lddc, strideC, batchCount, queue);
  } else {
    magmablas_dgemm_batched_strided(trans_A, trans_B, m, n, k, (double)alpha, (const double *)d_A, ldda, strideA, (const double *)d_B, lddb, strideB,
                                    (double)beta, (double *)d_C, lddc, strideC, batchCount, queue);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
static inline int devblas_gemm(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                               const CeedScalar *d_A, magma_int_t ldda, const CeedScalar *d_B, magma_int_t lddb, CeedScalar beta, CeedScalar *d_C,
                               magma_int_t lddc, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    magma_sgemm(trans_A, trans_B, m, n, k, (float)alpha, (const float *)d_A, ldda, (const float *)d_B, lddb, (float)beta, (float *)d_C, lddc, queue);
  } else {
    magma_dgemm(trans_A, trans_B, m, n, k, (double)alpha, (const double *)d_A, ldda, (const double *)d_B, lddb, (double)beta, (double *)d_C, lddc,
                queue);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
static inline int devblas_gemm_batched_strided(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k,
                                               CeedScalar alpha, const CeedScalar *d_A, magma_int_t ldda, magma_int_t strideA, const CeedScalar *d_B,
                                               magma_int_t lddb, magma_int_t strideB, CeedScalar beta, CeedScalar *d_C, magma_int_t lddc,
                                               magma_int_t strideC, magma_int_t batchCount, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    devblasSgemmStridedBatched(magma_queue_get_devblas_handle(queue), devblas_trans_const(trans_A), devblas_trans_const(trans_B), (int)m, (int)n,
                               (int)k, (const float *)&alpha, (const float *)d_A, (int)ldda, strideA, (const float *)d_B, (int)lddb, strideB,
                               (const float *)&beta, (float *)d_C, (int)lddc, strideC, (int)batchCount);
  } else {
    devblasDgemmStridedBatched(magma_queue_get_devblas_handle(queue), devblas_trans_const(trans_A), devblas_trans_const(trans_B), (int)m, (int)n,
                               (int)k, (const double *)&alpha, (const double *)d_A, (int)ldda, strideA, (const double *)d_B, (int)lddb, strideB,
                               (const double *)&beta, (double *)d_C, (int)lddc, strideC, (int)batchCount);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
int magma_gemm_nontensor(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                         const CeedScalar *d_A, magma_int_t ldda, const CeedScalar *d_B, magma_int_t lddb, CeedScalar beta, CeedScalar *d_C,
                         magma_int_t lddc, magma_queue_t queue) {
  magma_int_t nbatch, use_magmablas;
  magma_int_t arch = magma_getdevice_arch();

  // check for specific transpositions (NN and TN only)
  bool NN = trans_A == MagmaNoTrans && trans_B == MagmaNoTrans;
  bool TN = trans_A == MagmaTrans && trans_B == MagmaNoTrans;
  if (!(NN || TN)) {
    // default case -- no specific tuning
    devblas_gemm(trans_A, trans_B, m, n, k, alpha, d_A, ldda, d_B, lddb, beta, d_C, lddc, queue);
    return 0;
  }

  // get tuning decision
  char trans     = (trans_A == MagmaNoTrans) ? 'n' : 't';
  char precision = (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) ? 's' : 'd';
  gemm_selector(arch, precision, trans, m, n, k, &nbatch, &use_magmablas);

  // perform the gemm operation
  if (nbatch == n) {
    // no batching
    if (use_magmablas) {
      magmablas_gemm(trans_A, trans_B, m, n, k, alpha, d_A, ldda, d_B, lddb, beta, d_C, lddc, queue);
    } else {
      devblas_gemm(trans_A, trans_B, m, n, k, alpha, d_A, ldda, d_B, lddb, beta, d_C, lddc, queue);
    }
  } else {
    // use batch kernels
    magma_int_t batchCount = n / nbatch;
    magma_int_t n2         = n - (batchCount * nbatch);
    magma_int_t strideA    = 0;
    magma_int_t strideB    = lddb * nbatch;
    magma_int_t strideC    = lddc * nbatch;

    if (use_magmablas) {
      if (batchCount > 0) {
        magmablas_gemm_batched_strided(trans_A, trans_B, m, nbatch, k, alpha, d_A, ldda, strideA, d_B, lddb, strideB, beta, d_C, lddc, strideC,
                                       batchCount, queue);
      }

      // cleanup
      if (n2 > 0) {
        devblas_gemm(trans_A, trans_B, m, n2, k, alpha, d_A, ldda, d_B + batchCount * strideB, lddb, beta, d_C + batchCount * strideC, lddc, queue);
      }
    } else {
      if (batchCount > 0) {
        devblas_gemm_batched_strided(trans_A, trans_B, m, nbatch, k, alpha, d_A, ldda, strideA, d_B, lddb, strideB, beta, d_C, lddc, strideC,
                                     batchCount, queue);
      }

      // cleanup
      if (n2 > 0) {
        devblas_gemm_batched_strided(trans_A, trans_B, m, n2, k, alpha, d_A, ldda, strideA, d_B + batchCount * strideB, lddb, strideB, beta,
                                     d_C + batchCount * strideC, lddc, strideC, 1, queue);
      }
    }
  }

  // wait for the operation to complete
  ceed_magma_queue_sync(queue);

  return 0;
}
