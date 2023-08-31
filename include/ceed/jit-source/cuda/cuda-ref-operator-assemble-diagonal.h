// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA operator diagonal assembly
#ifndef CEED_CUDA_REF_OPERATOR_ASSEMBLE_DIAGONAL_H
#define CEED_CUDA_REF_OPERATOR_ASSEMBLE_DIAGONAL_H

#include <ceed.h>

#if USE_CEEDSIZE
typedef CeedSize IndexType;
#else
typedef CeedInt IndexType;
#endif

//------------------------------------------------------------------------------
// Get Basis Emode Pointer
//------------------------------------------------------------------------------
extern "C" __device__ void CeedOperatorGetBasisPointer_Cuda(const CeedScalar **basis_ptr, CeedEvalMode e_mode, const CeedScalar *identity,
                                                            const CeedScalar *interp, const CeedScalar *grad) {
  switch (e_mode) {
    case CEED_EVAL_NONE:
      *basis_ptr = identity;
      break;
    case CEED_EVAL_INTERP:
      *basis_ptr = interp;
      break;
    case CEED_EVAL_GRAD:
      *basis_ptr = grad;
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
__device__ void diagonalCore(const CeedInt num_elem, const bool is_point_block, const CeedScalar *identity, const CeedScalar *interp_in,
                             const CeedScalar *grad_in, const CeedScalar *interp_out, const CeedScalar *grad_out, const CeedEvalMode *e_mode_in,
                             const CeedEvalMode *e_mode_out, const CeedScalar *__restrict__ assembled_qf_array,
                             CeedScalar *__restrict__ elem_diag_array) {
  const int tid = threadIdx.x;  // running with P threads, tid is evec node
  if (tid >= NUM_NODES) return;

  // Compute the diagonal of B^T D B
  // Each element
  for (IndexType e = blockIdx.x * blockDim.z + threadIdx.z; e < num_elem; e += gridDim.x * blockDim.z) {
    IndexType d_out = -1;

    // Each basis eval mode pair
    for (IndexType e_out = 0; e_out < NUM_E_MODE_OUT; e_out++) {
      const CeedScalar *b_t = NULL;

      if (e_mode_out[e_out] == CEED_EVAL_GRAD) d_out += 1;
      CeedOperatorGetBasisPointer_Cuda(&b_t, e_mode_out[e_out], identity, interp_out, &grad_out[d_out * NUM_QPTS * NUM_NODES]);
      IndexType d_in = -1;

      for (IndexType e_in = 0; e_in < NUM_E_MODE_IN; e_in++) {
        const CeedScalar *b = NULL;

        if (e_mode_in[e_in] == CEED_EVAL_GRAD) d_in += 1;
        CeedOperatorGetBasisPointer_Cuda(&b, e_mode_in[e_in], identity, interp_in, &grad_in[d_in * NUM_QPTS * NUM_NODES]);
        // Each component
        for (IndexType comp_out = 0; comp_out < NUM_COMP; comp_out++) {
          // Each qpoint/node pair
          if (is_point_block) {
            // Point Block Diagonal
            for (IndexType comp_in = 0; comp_in < NUM_COMP; comp_in++) {
              CeedScalar e_value = 0.;

              for (IndexType q = 0; q < NUM_QPTS; q++) {
                const CeedScalar qf_value =
                    assembled_qf_array[((((e_in * NUM_COMP + comp_in) * NUM_E_MODE_OUT + e_out) * NUM_COMP + comp_out) * num_elem + e) * NUM_QPTS +
                                       q];

                e_value += b_t[q * NUM_NODES + tid] * qf_value * b[q * NUM_NODES + tid];
              }
              elem_diag_array[((comp_out * NUM_COMP + comp_in) * num_elem + e) * NUM_NODES + tid] += e_value;
            }
          } else {
            // Diagonal Only
            CeedScalar e_value = 0.;

            for (IndexType q = 0; q < NUM_QPTS; q++) {
              const CeedScalar qf_value =
                  assembled_qf_array[((((e_in * NUM_COMP + comp_out) * NUM_E_MODE_OUT + e_out) * NUM_COMP + comp_out) * num_elem + e) * NUM_QPTS + q];

              e_value += b_t[q * NUM_NODES + tid] * qf_value * b[q * NUM_NODES + tid];
            }
            elem_diag_array[(comp_out * num_elem + e) * NUM_NODES + tid] += e_value;
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Linear diagonal
//------------------------------------------------------------------------------
extern "C" __global__ void linearDiagonal(const CeedInt num_elem, const CeedScalar *identity, const CeedScalar *interp_in, const CeedScalar *grad_in,
                                          const CeedScalar *interp_out, const CeedScalar *grad_out, const CeedEvalMode *e_mode_in,
                                          const CeedEvalMode *e_mode_out, const CeedScalar *__restrict__ assembled_qf_array,
                                          CeedScalar *__restrict__ elem_diag_array) {
  diagonalCore(num_elem, false, identity, interp_in, grad_in, interp_out, grad_out, e_mode_in, e_mode_out, assembled_qf_array, elem_diag_array);
}

//------------------------------------------------------------------------------
// Linear point block diagonal
//------------------------------------------------------------------------------
extern "C" __global__ void linearPointBlockDiagonal(const CeedInt num_elem, const CeedScalar *identity, const CeedScalar *interp_in,
                                                    const CeedScalar *grad_in, const CeedScalar *interp_out, const CeedScalar *grad_out,
                                                    const CeedEvalMode *e_mode_in, const CeedEvalMode *e_mode_out,
                                                    const CeedScalar *__restrict__ assembled_qf_array, CeedScalar *__restrict__ elem_diag_array) {
  diagonalCore(num_elem, true, identity, interp_in, grad_in, interp_out, grad_out, e_mode_in, e_mode_out, assembled_qf_array, elem_diag_array);
}

//------------------------------------------------------------------------------

#endif  // CEED_CUDA_REF_OPERATOR_ASSEMBLE_DIAGONAL_H
