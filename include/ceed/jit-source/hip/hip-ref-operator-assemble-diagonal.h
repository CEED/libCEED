// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

#if CEEDSIZE
typedef CeedSize IndexType;
#else
typedef CeedInt IndexType;
#endif

//------------------------------------------------------------------------------
// Get Basis Emode Pointer
//------------------------------------------------------------------------------
extern "C" __device__ void CeedOperatorGetBasisPointer_Hip(const CeedScalar **basisptr, CeedEvalMode emode, const CeedScalar *identity,
                                                           const CeedScalar *interp, const CeedScalar *grad) {
  switch (emode) {
    case CEED_EVAL_NONE:
      *basisptr = identity;
      break;
    case CEED_EVAL_INTERP:
      *basisptr = interp;
      break;
    case CEED_EVAL_GRAD:
      *basisptr = grad;
      break;
    case CEED_EVAL_WEIGHT:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      break;  // Caught by QF Assembly
  }
}

//------------------------------------------------------------------------------
// Core code for diagonal assembly
//------------------------------------------------------------------------------
__device__ void diagonalCore(const CeedInt nelem, const bool pointBlock, const CeedScalar *identity, const CeedScalar *interpin,
                             const CeedScalar *gradin, const CeedScalar *interpout, const CeedScalar *gradout, const CeedEvalMode *emodein,
                             const CeedEvalMode *emodeout, const CeedScalar *__restrict__ assembledqfarray, CeedScalar *__restrict__ elemdiagarray) {
  const int tid = threadIdx.x;  // running with P threads, tid is evec node
  if (tid >= NNODES) return;

  // Compute the diagonal of B^T D B
  // Each element
  for (IndexType e = blockIdx.x * blockDim.z + threadIdx.z; e < nelem; e += gridDim.x * blockDim.z) {
    IndexType dout = -1;
    // Each basis eval mode pair
    for (IndexType eout = 0; eout < NUMEMODEOUT; eout++) {
      const CeedScalar *bt = NULL;
      if (emodeout[eout] == CEED_EVAL_GRAD) dout += 1;
      CeedOperatorGetBasisPointer_Hip(&bt, emodeout[eout], identity, interpout, &gradout[dout * NQPTS * NNODES]);
      IndexType din = -1;
      for (IndexType ein = 0; ein < NUMEMODEIN; ein++) {
        const CeedScalar *b = NULL;
        if (emodein[ein] == CEED_EVAL_GRAD) din += 1;
        CeedOperatorGetBasisPointer_Hip(&b, emodein[ein], identity, interpin, &gradin[din * NQPTS * NNODES]);
        // Each component
        for (IndexType compOut = 0; compOut < NCOMP; compOut++) {
          // Each qpoint/node pair
          if (pointBlock) {
            // Point Block Diagonal
            for (IndexType compIn = 0; compIn < NCOMP; compIn++) {
              CeedScalar evalue = 0.;
              for (IndexType q = 0; q < NQPTS; q++) {
                const CeedScalar qfvalue =
                    assembledqfarray[((((ein * NCOMP + compIn) * NUMEMODEOUT + eout) * NCOMP + compOut) * nelem + e) * NQPTS + q];
                evalue += bt[q * NNODES + tid] * qfvalue * b[q * NNODES + tid];
              }
              elemdiagarray[((compOut * NCOMP + compIn) * nelem + e) * NNODES + tid] += evalue;
            }
          } else {
            // Diagonal Only
            CeedScalar evalue = 0.;
            for (IndexType q = 0; q < NQPTS; q++) {
              const CeedScalar qfvalue =
                  assembledqfarray[((((ein * NCOMP + compOut) * NUMEMODEOUT + eout) * NCOMP + compOut) * nelem + e) * NQPTS + q];
              evalue += bt[q * NNODES + tid] * qfvalue * b[q * NNODES + tid];
            }
            elemdiagarray[(compOut * nelem + e) * NNODES + tid] += evalue;
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Linear diagonal
//------------------------------------------------------------------------------
extern "C" __global__ void linearDiagonal(const CeedInt nelem, const CeedScalar *identity, const CeedScalar *interpin, const CeedScalar *gradin,
                                          const CeedScalar *interpout, const CeedScalar *gradout, const CeedEvalMode *emodein,
                                          const CeedEvalMode *emodeout, const CeedScalar *__restrict__ assembledqfarray,
                                          CeedScalar *__restrict__ elemdiagarray) {
  diagonalCore(nelem, false, identity, interpin, gradin, interpout, gradout, emodein, emodeout, assembledqfarray, elemdiagarray);
}

//------------------------------------------------------------------------------
// Linear point block diagonal
//------------------------------------------------------------------------------
extern "C" __global__ void linearPointBlockDiagonal(const CeedInt nelem, const CeedScalar *identity, const CeedScalar *interpin,
                                                    const CeedScalar *gradin, const CeedScalar *interpout, const CeedScalar *gradout,
                                                    const CeedEvalMode *emodein, const CeedEvalMode *emodeout,
                                                    const CeedScalar *__restrict__ assembledqfarray, CeedScalar *__restrict__ elemdiagarray) {
  diagonalCore(nelem, true, identity, interpin, gradin, interpout, gradout, emodein, emodeout, assembledqfarray, elemdiagarray);
}

//------------------------------------------------------------------------------
