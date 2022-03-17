// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"

#ifdef HAVE_HIP
// TODO: Tune for HIP
int
magma_sgemm_nontensor(
  magma_trans_t transA, magma_trans_t transB,
  magma_int_t m, magma_int_t n, magma_int_t k,
  float alpha, const float *dA, magma_int_t ldda,
  const float *dB, magma_int_t lddb,
  float beta,  float *dC, magma_int_t lddc,
  magma_queue_t queue ) {

  // check for specific transpositions (NN and TN only)
  bool NN = transA == MagmaNoTrans && transB == MagmaNoTrans;
  bool TN = transA == MagmaTrans   && transB == MagmaNoTrans;
  if ( !(NN || TN) ) {
    // default case -- no specific tuning
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc,
                queue);
    return 0;
  }
  // select batching vs. no-batching (based on offline tuning)
  magma_int_t n1 = n;
  bool use_magmablas = false;

  // Always use hipblas for now, pending further tuning
  if ( NN ) {
    if     (m <=   2 && k <=    3) { use_magmablas = false; n1 = 128;}
    else if (m <=   3 && k <=    4) { use_magmablas = false; n1 = 128;}
    else if (m <=   4 && k <=    5) { use_magmablas = false; n1 = 128;}
    else if (m <=   4 && k <=    9) { use_magmablas = false;  n1 =  16;}
    else if (m <=   5 && k <=    6) { use_magmablas = false; n1 = 128;}
    else if (m <=   6 && k <=    7) { use_magmablas = false; n1 = 128;}
    else if (m <=   7 && k <=    8) { use_magmablas = false; n1 = 128;}
    else if (m <=   8 && k <=    9) { use_magmablas = false;  n1 =  16;}
    else if (m <=   8 && k <=   27) { use_magmablas = false;  n1 =  16;}
    else if (m <=   9 && k <=   10) { use_magmablas = false;  n1 = 128;}
    else if (m <=   9 && k <=   16) { use_magmablas = false; n1 = 128;}
    else if (m <=  16 && k <=   25) { use_magmablas = false;  n1 =  16;}
    else if (m <=  25 && k <=   36) { use_magmablas = false;  n1 = 128;}
    else if (m <=  27 && k <=   64) { use_magmablas = false;  n1 = 128;}
    else if (m <=  36 && k <=   49) { use_magmablas = false; n1 = 128;}
    else if (m <=  49 && k <=   64) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=   81) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=  125) { use_magmablas = false; n1 = 128;}
    else if (m <=  81 && k <=  100) { use_magmablas = false; n1 = 128;}
    else if (m <= 125 && k <=  216) { use_magmablas = false; n1 = 256;}
    else if (m <= 216 && k <=  343) { use_magmablas = false; n1 = 256;}
    else if (m <= 343 && k <=  512) { use_magmablas = false; n1 = 256;}
    else if (m <= 512 && k <=  729) { use_magmablas = false; n1 = 256;}
    else if (m <= 729 && k <= 1000) { use_magmablas = false; n1 =   n;}
    else { use_magmablas = false; n1 =   n;}
  } else if ( TN ) {
    if     (m <=   2 && k <=    3) { use_magmablas = false; n1 =   4;}
    else if (m <=   3 && k <=    4) { use_magmablas = false; n1 =   4;}
    else if (m <=   4 && k <=    5) { use_magmablas = false; n1 = 128;}
    else if (m <=   4 && k <=    9) { use_magmablas = false; n1 =   4;}
    else if (m <=   5 && k <=    6) { use_magmablas = false; n1 = 128;}
    else if (m <=   6 && k <=    7) { use_magmablas = false; n1 = 128;}
    else if (m <=   7 && k <=    8) { use_magmablas = false; n1 = 128;}
    else if (m <=   8 && k <=    9) { use_magmablas = false;  n1 = 128;}
    else if (m <=   8 && k <=   27) { use_magmablas = false;  n1 = 128;}
    else if (m <=   9 && k <=   10) { use_magmablas = false;  n1 = 128;}
    else if (m <=   9 && k <=   16) { use_magmablas = false; n1 = 128;}
    else if (m <=  16 && k <=   25) { use_magmablas = false;  n1 = 128;}
    else if (m <=  25 && k <=   36) { use_magmablas = false;  n1 = 128;}
    else if (m <=  27 && k <=   64) { use_magmablas = false;  n1 = 128;}
    else if (m <=  36 && k <=   49) { use_magmablas = false;  n1 = 128;}
    else if (m <=  49 && k <=   64) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=   81) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=  125) { use_magmablas = false; n1 = 128;}
    else if (m <=  81 && k <=  100) { use_magmablas = false; n1 = 128;}
    else if (m <= 125 && k <=  216) { use_magmablas = false; n1 = 128;}
    else if (m <= 216 && k <=  343) { use_magmablas = false; n1 = 128;}
    else if (m <= 343 && k <=  512) { use_magmablas = false; n1 = 128;}
    else if (m <= 512 && k <=  729) { use_magmablas = false; n1 = 128;}
    else if (m <= 729 && k <= 1000) { use_magmablas = false; n1 =   n;}
    else { use_magmablas = false; n1 =   n;}
  }

  // perform the sgemm operation
  if ( n1 == n) {
    // no batching, do not use magmablas
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc,
                queue);
  } else {
    // use batch kernels
    magma_int_t batchCount = n/n1;
    magma_int_t n2 = n - (batchCount * n1);
    magma_int_t strideA = 0;
    magma_int_t strideB = lddb*n1;
    magma_int_t strideC = lddc*n1;

    if ( use_magmablas ) {
      magmablas_sgemm_batched_strided(
        transA, transB, m, n1, k,
        alpha, dA, ldda, strideA,
        dB, lddb, strideB,
        beta,  dC, lddc, strideC,
        batchCount, queue);

      // cleanup
      if (n2 > 0) {
        magma_sgemm(
          transA, transB, m, n2, k,
          alpha, dA, ldda,
          dB + batchCount * strideB, lddb,
          beta,  dC + batchCount * strideC, lddc, queue);
      }
    } else {
      hipblasSgemmStridedBatched(
        magma_queue_get_hipblas_handle( queue ),
        hipblas_trans_const(transA), hipblas_trans_const(transB),
        (int)m, (int)n1, (int)k,
        &alpha, (const float *) dA, (int)ldda, strideA,
        (const float *) dB, (int)lddb, strideB,
        &beta,                  dC, (int)lddc, strideC, (int)batchCount );

      // cleanup
      if (n2 > 0) {
        hipblasSgemmStridedBatched(
          magma_queue_get_hipblas_handle( queue ),
          hipblas_trans_const(transA), hipblas_trans_const(transB),
          (int)m, (int)n2, (int)k,
          &alpha, (const float *) dA, (int)ldda, strideA,
          (const float *) dB + batchCount * strideB, (int)lddb, strideB,
          &beta,                  dC + batchCount * strideC, (int)lddc, strideC, 1 );
      }
    }
  }

  // wait for the operation to complete
  ceed_magma_queue_sync( queue );

  return 0;
}

#else
int
magma_sgemm_nontensor(
  magma_trans_t transA, magma_trans_t transB,
  magma_int_t m, magma_int_t n, magma_int_t k,
  float alpha, const float *dA, magma_int_t ldda,
  const float *dB, magma_int_t lddb,
  float beta,  float *dC, magma_int_t lddc,
  magma_queue_t queue ) {

  // check for specific transpositions (NN and TN only)
  bool NN = transA == MagmaNoTrans && transB == MagmaNoTrans;
  bool TN = transA == MagmaTrans   && transB == MagmaNoTrans;
  if ( !(NN || TN) ) {
    // default case -- no specific tuning
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc,
                queue);
    return 0;
  }
  // select batching vs. no-batching (based on offline tuning)
  magma_int_t n1 = n;
  bool use_magmablas = false;

  if ( NN ) {
    if     (m <=   2 && k <=    3) { use_magmablas = false; n1 = 128;}
    else if (m <=   3 && k <=    4) { use_magmablas = false; n1 = 128;}
    else if (m <=   4 && k <=    5) { use_magmablas = false; n1 = 128;}
    else if (m <=   4 && k <=    9) { use_magmablas = true;  n1 =  16;}
    else if (m <=   5 && k <=    6) { use_magmablas = false; n1 = 128;}
    else if (m <=   6 && k <=    7) { use_magmablas = false; n1 = 128;}
    else if (m <=   7 && k <=    8) { use_magmablas = false; n1 = 128;}
    else if (m <=   8 && k <=    9) { use_magmablas = true;  n1 =  16;}
    else if (m <=   8 && k <=   27) { use_magmablas = true;  n1 =  16;}
    else if (m <=   9 && k <=   10) { use_magmablas = true;  n1 = 128;}
    else if (m <=   9 && k <=   16) { use_magmablas = false; n1 = 128;}
    else if (m <=  16 && k <=   25) { use_magmablas = true;  n1 =  16;}
    else if (m <=  25 && k <=   36) { use_magmablas = true;  n1 = 128;}
    else if (m <=  27 && k <=   64) { use_magmablas = true;  n1 = 128;}
    else if (m <=  36 && k <=   49) { use_magmablas = false; n1 = 128;}
    else if (m <=  49 && k <=   64) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=   81) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=  125) { use_magmablas = false; n1 = 128;}
    else if (m <=  81 && k <=  100) { use_magmablas = false; n1 = 128;}
    else if (m <= 125 && k <=  216) { use_magmablas = false; n1 = 256;}
    else if (m <= 216 && k <=  343) { use_magmablas = false; n1 = 256;}
    else if (m <= 343 && k <=  512) { use_magmablas = false; n1 = 256;}
    else if (m <= 512 && k <=  729) { use_magmablas = false; n1 = 256;}
    else if (m <= 729 && k <= 1000) { use_magmablas = false; n1 =   n;}
    else { use_magmablas = false; n1 =   n;}
  } else if ( TN ) {
    if     (m <=   2 && k <=    3) { use_magmablas = false; n1 =   4;}
    else if (m <=   3 && k <=    4) { use_magmablas = false; n1 =   4;}
    else if (m <=   4 && k <=    5) { use_magmablas = false; n1 = 128;}
    else if (m <=   4 && k <=    9) { use_magmablas = false; n1 =   4;}
    else if (m <=   5 && k <=    6) { use_magmablas = false; n1 = 128;}
    else if (m <=   6 && k <=    7) { use_magmablas = false; n1 = 128;}
    else if (m <=   7 && k <=    8) { use_magmablas = false; n1 = 128;}
    else if (m <=   8 && k <=    9) { use_magmablas = true;  n1 = 128;}
    else if (m <=   8 && k <=   27) { use_magmablas = true;  n1 = 128;}
    else if (m <=   9 && k <=   10) { use_magmablas = true;  n1 = 128;}
    else if (m <=   9 && k <=   16) { use_magmablas = false; n1 = 128;}
    else if (m <=  16 && k <=   25) { use_magmablas = true;  n1 = 128;}
    else if (m <=  25 && k <=   36) { use_magmablas = true;  n1 = 128;}
    else if (m <=  27 && k <=   64) { use_magmablas = true;  n1 = 128;}
    else if (m <=  36 && k <=   49) { use_magmablas = true;  n1 = 128;}
    else if (m <=  49 && k <=   64) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=   81) { use_magmablas = false; n1 = 128;}
    else if (m <=  64 && k <=  125) { use_magmablas = false; n1 = 128;}
    else if (m <=  81 && k <=  100) { use_magmablas = false; n1 = 128;}
    else if (m <= 125 && k <=  216) { use_magmablas = false; n1 = 128;}
    else if (m <= 216 && k <=  343) { use_magmablas = false; n1 = 128;}
    else if (m <= 343 && k <=  512) { use_magmablas = false; n1 = 128;}
    else if (m <= 512 && k <=  729) { use_magmablas = false; n1 = 128;}
    else if (m <= 729 && k <= 1000) { use_magmablas = false; n1 =   n;}
    else { use_magmablas = false; n1 =   n;}
  }

  // perform the sgemm operation
  if ( n1 == n) {
    // no batching, do not use magmablas
    magma_sgemm(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc,
                queue);
  } else {
    // use batch kernels
    magma_int_t batchCount = n/n1;
    magma_int_t n2 = n - (batchCount * n1);
    magma_int_t strideA = 0;
    magma_int_t strideB = lddb*n1;
    magma_int_t strideC = lddc*n1;

    if ( use_magmablas ) {
      magmablas_sgemm_batched_strided(
        transA, transB, m, n1, k,
        alpha, dA, ldda, strideA,
        dB, lddb, strideB,
        beta,  dC, lddc, strideC,
        batchCount, queue);

      // cleanup
      if (n2 > 0) {
        magma_sgemm(
          transA, transB, m, n2, k,
          alpha, dA, ldda,
          dB + batchCount * strideB, lddb,
          beta,  dC + batchCount * strideC, lddc, queue);
      }
    } else {
      cublasSgemmStridedBatched(
        magma_queue_get_cublas_handle( queue ),
        cublas_trans_const(transA), cublas_trans_const(transB),
        (int)m, (int)n1, (int)k,
        &alpha, (const float *) dA, (int)ldda, strideA,
        (const float *) dB, (int)lddb, strideB,
        &beta,                  dC, (int)lddc, strideC, (int)batchCount );

      // cleanup
      if (n2 > 0) {
        cublasSgemmStridedBatched(
          magma_queue_get_cublas_handle( queue ),
          cublas_trans_const(transA), cublas_trans_const(transB),
          (int)m, (int)n2, (int)k,
          &alpha, (const float *) dA, (int)ldda, strideA,
          (const float *) dB + batchCount * strideB, (int)lddb, strideB,
          &beta,                  dC + batchCount * strideC, (int)lddc, strideC, 1 );
      }
    }
  }

  // wait for the operation to complete
  ceed_magma_queue_sync( queue );

  return 0;
}
#endif
