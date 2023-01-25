// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"

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
static int magmablas_gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                          const CeedScalar *dA, magma_int_t ldda, const CeedScalar *dB, magma_int_t lddb, CeedScalar beta, CeedScalar *dC,
                          magma_int_t lddc, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    magmablas_sgemm(transA, transB, m, n, k, (float)alpha, (const float *)dA, ldda, (const float *)dB, lddb, (float)beta, (float *)dC, lddc, queue);
  } else {
    magmablas_dgemm(transA, transB, m, n, k, (double)alpha, (const double *)dA, ldda, (const double *)dB, lddb, (double)beta, (double *)dC, lddc,
                    queue);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
static int magmablas_gemm_batched_strided(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                          const CeedScalar *dA, magma_int_t ldda, magma_int_t strideA, const CeedScalar *dB, magma_int_t lddb,
                                          magma_int_t strideB, CeedScalar beta, CeedScalar *dC, magma_int_t lddc, magma_int_t strideC,
                                          magma_int_t batchCount, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    magmablas_sgemm_batched_strided(transA, transB, m, n, k, (float)alpha, (const float *)dA, ldda, strideA, (const float *)dB, lddb, strideB,
                                    (float)beta, (float *)dC, lddc, strideC, batchCount, queue);
  } else {
    magmablas_dgemm_batched_strided(transA, transB, m, n, k, (double)alpha, (const double *)dA, ldda, strideA, (const double *)dB, lddb, strideB,
                                    (double)beta, (double *)dC, lddc, strideC, batchCount, queue);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
static int devblas_gemm(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                        const CeedScalar *dA, magma_int_t ldda, const CeedScalar *dB, magma_int_t lddb, CeedScalar beta, CeedScalar *dC,
                        magma_int_t lddc, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    magma_sgemm(transA, transB, m, n, k, (float)alpha, (const float *)dA, ldda, (const float *)dB, lddb, (float)beta, (float *)dC, lddc, queue);
  } else {
    magma_dgemm(transA, transB, m, n, k, (double)alpha, (const double *)dA, ldda, (const double *)dB, lddb, (double)beta, (double *)dC, lddc, queue);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
static int devblas_gemm_batched_strided(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                        const CeedScalar *dA, magma_int_t ldda, magma_int_t strideA, const CeedScalar *dB, magma_int_t lddb,
                                        magma_int_t strideB, CeedScalar beta, CeedScalar *dC, magma_int_t lddc, magma_int_t strideC,
                                        magma_int_t batchCount, magma_queue_t queue) {
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    devblasSgemmStridedBatched(magma_queue_get_devblas_handle(queue), devblas_trans_const(transA), devblas_trans_const(transB), (int)m, (int)n,
                               (int)k, (const float *)&alpha, (const float *)dA, (int)ldda, strideA, (const float *)dB, (int)lddb, strideB,
                               (const float *)&beta, (float *)dC, (int)lddc, strideC, (int)batchCount);
  } else {
    devblasDgemmStridedBatched(magma_queue_get_devblas_handle(queue), devblas_trans_const(transA), devblas_trans_const(transB), (int)m, (int)n,
                               (int)k, (const double *)&alpha, (const double *)dA, (int)ldda, strideA, (const double *)dB, (int)lddb, strideB,
                               (const double *)&beta, (double *)dC, (int)lddc, strideC, (int)batchCount);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
int magma_gemm_nontensor(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                         const CeedScalar *dA, magma_int_t ldda, const CeedScalar *dB, magma_int_t lddb, CeedScalar beta, CeedScalar *dC,
                         magma_int_t lddc, magma_queue_t queue) {
  magma_int_t nbatch, use_magmablas;
  magma_int_t arch = magma_getdevice_arch();

  // check for specific transpositions (NN and TN only)
  bool NN = transA == MagmaNoTrans && transB == MagmaNoTrans;
  bool TN = transA == MagmaTrans && transB == MagmaNoTrans;
  if (!(NN || TN)) {
    // default case -- no specific tuning
    devblas_gemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
    return 0;
  }

  // get tuning decision
  char trans     = (transA == MagmaNoTrans) ? 'n' : 't';
  char precision = (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) ? 's' : 'd';
  gemm_selector(arch, precision, trans, m, n, k, &nbatch, &use_magmablas);

  // perform the gemm operation
  if (nbatch == n) {
    // no batching
    if (use_magmablas) {
      magmablas_gemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
    } else {
      devblas_gemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
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
        magmablas_gemm_batched_strided(transA, transB, m, nbatch, k, alpha, dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, strideC, batchCount,
                                       queue);
      }

      // cleanup
      if (n2 > 0) {
        devblas_gemm(transA, transB, m, n2, k, alpha, dA, ldda, dB + batchCount * strideB, lddb, beta, dC + batchCount * strideC, lddc, queue);
      }
    } else {
      if (batchCount > 0) {
        devblas_gemm_batched_strided(transA, transB, m, nbatch, k, alpha, dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, strideC, batchCount,
                                     queue);
      }

      // cleanup
      if (n2 > 0) {
        devblas_gemm_batched_strided(transA, transB, m, n2, k, alpha, dA, ldda, strideA, dB + batchCount * strideB, lddb, strideB, beta,
                                     dC + batchCount * strideC, lddc, strideC, 1, queue);
      }
    }
  }

  // wait for the operation to complete
  ceed_magma_queue_sync(queue);

  return 0;
}
