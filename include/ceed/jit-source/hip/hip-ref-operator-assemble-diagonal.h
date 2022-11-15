// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

//------------------------------------------------------------------------------
// Diagonal assembly kernels
//------------------------------------------------------------------------------
typedef enum {
  /// Perform no evaluation (either because there is no data or it is already at
  /// quadrature points)
  CEED_EVAL_NONE = 0,
  /// Interpolate from nodes to quadrature points
  CEED_EVAL_INTERP = 1,
  /// Evaluate gradients at quadrature points from input in a nodal basis
  CEED_EVAL_GRAD = 2,
  /// Evaluate divergence at quadrature points from input in a nodal basis
  CEED_EVAL_DIV = 4,
  /// Evaluate curl at quadrature points from input in a nodal basis
  CEED_EVAL_CURL = 8,
  /// Using no input, evaluate quadrature weights on the reference element
  CEED_EVAL_WEIGHT = 16,
} CeedEvalMode;

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
  for (CeedInt e = blockIdx.x * blockDim.z + threadIdx.z; e < nelem; e += gridDim.x * blockDim.z) {
    CeedInt dout = -1;
    // Each basis eval mode pair
    for (CeedInt eout = 0; eout < NUMEMODEOUT; eout++) {
      const CeedScalar *bt = NULL;
      if (emodeout[eout] == CEED_EVAL_GRAD) dout += 1;
      CeedOperatorGetBasisPointer_Hip(&bt, emodeout[eout], identity, interpout, &gradout[dout * NQPTS * NNODES]);
      CeedInt din = -1;
      for (CeedInt ein = 0; ein < NUMEMODEIN; ein++) {
        const CeedScalar *b = NULL;
        if (emodein[ein] == CEED_EVAL_GRAD) din += 1;
        CeedOperatorGetBasisPointer_Hip(&b, emodein[ein], identity, interpin, &gradin[din * NQPTS * NNODES]);
        // Each component
        for (CeedInt compOut = 0; compOut < NCOMP; compOut++) {
          // Each qpoint/node pair
          if (pointBlock) {
            // Point Block Diagonal
            for (CeedInt compIn = 0; compIn < NCOMP; compIn++) {
              CeedScalar evalue = 0.;
              for (CeedInt q = 0; q < NQPTS; q++) {
                const CeedScalar qfvalue =
                    assembledqfarray[((((ein * NCOMP + compIn) * NUMEMODEOUT + eout) * NCOMP + compOut) * nelem + e) * NQPTS + q];
                evalue += bt[q * NNODES + tid] * qfvalue * b[q * NNODES + tid];
              }
              elemdiagarray[((compOut * NCOMP + compIn) * nelem + e) * NNODES + tid] += evalue;
            }
          } else {
            // Diagonal Only
            CeedScalar evalue = 0.;
            for (CeedInt q = 0; q < NQPTS; q++) {
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
