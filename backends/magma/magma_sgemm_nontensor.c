// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"

#ifdef CEED_MAGMA_USE_HIP
#define devblasSgemmStridedBatched hipblasSgemmStridedBatched
#define magma_queue_get_devblas_handle magma_queue_get_hipblas_handle
#define devblas_trans_const hipblas_trans_const
#else
#define devblasSgemmStridedBatched cublasSgemmStridedBatched
#define magma_queue_get_devblas_handle magma_queue_get_cublas_handle
#define devblas_trans_const cublas_trans_const
#endif

int magma_sgemm_nontensor(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, float alpha, const float *dA,
                          magma_int_t ldda, const float *dB, magma_int_t lddb, float beta, float *dC, magma_int_t lddc, magma_queue_t queue) {
  magma_int_t nbatch, use_magmablas;
  magma_int_t arch = magma_getdevice_arch();

  // check for specific transpositions (NN and TN only)
  bool NN = transA == MagmaNoTrans && transB == MagmaNoTrans;
  bool TN = transA == MagmaTrans && transB == MagmaNoTrans;
  if (!(NN || TN)) {
    // default case -- no specific tuning
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
    return 0;
  }

  // get tuning decision
  char trans     = (transA == MagmaNoTrans) ? 'n' : 't';
  char precision = 'd';
  gemm_selector(arch, precision, trans, m, n, k, &nbatch, &use_magmablas);

#if 0
  printf("%c %c -- (%3d, %3d, %3d) -- nbatch = %3d, use_magma = %d\n", trans,
         precision, m, n, k, nbatch, use_magmablas);
#endif

  // perform the sgemm operation
  if (nbatch == n) {
    // no batching
    if (use_magmablas) {
      magmablas_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
    } else {
      magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
    }
  } else {
    // use batch kernels
    magma_int_t batchCount = n / nbatch;
    magma_int_t n2         = n - (batchCount * nbatch);
    magma_int_t strideA    = 0;
    magma_int_t strideB    = lddb * nbatch;
    magma_int_t strideC    = lddc * nbatch;

    if (use_magmablas) {
      magmablas_sgemm_batched_strided(transA, transB, m, nbatch, k, alpha, dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, strideC, batchCount,
                                      queue);

      // cleanup
      if (n2 > 0) {
        magma_sgemm(transA, transB, m, n2, k, alpha, dA, ldda, dB + batchCount * strideB, lddb, beta, dC + batchCount * strideC, lddc, queue);
      }
    } else {
      devblasSgemmStridedBatched(magma_queue_get_devblas_handle(queue), devblas_trans_const(transA), devblas_trans_const(transB), (int)m, (int)nbatch,
                                 (int)k, &alpha, (const float *)dA, (int)ldda, strideA, (const float *)dB, (int)lddb, strideB, &beta, dC, (int)lddc,
                                 strideC, (int)batchCount);

      // cleanup
      if (n2 > 0) {
        devblasSgemmStridedBatched(magma_queue_get_devblas_handle(queue), devblas_trans_const(transA), devblas_trans_const(transB), (int)m, (int)n2,
                                   (int)k, &alpha, (const float *)dA, (int)ldda, strideA, (const float *)dB + batchCount * strideB, (int)lddb,
                                   strideB, &beta, dC + batchCount * strideC, (int)lddc, strideC, 1);
      }
    }
  }

  // wait for the operation to complete
  ceed_magma_queue_sync(queue);

  return 0;
}
