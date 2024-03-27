// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP operator diagonal assembly
#ifndef CEED_HIP_REF_OPERATOR_ASSEMBLE_DIAGONAL_H
#define CEED_HIP_REF_OPERATOR_ASSEMBLE_DIAGONAL_H

#include <ceed.h>

#if USE_CEEDSIZE
typedef CeedSize IndexType;
#else
typedef CeedInt IndexType;
#endif

//------------------------------------------------------------------------------
// Get basis pointer
//------------------------------------------------------------------------------
static __device__ __inline__ void GetBasisPointer(const CeedScalar **basis_ptr, CeedEvalMode eval_modes, const CeedScalar *identity,
                                                  const CeedScalar *interp, const CeedScalar *grad, const CeedScalar *div, const CeedScalar *curl) {
  switch (eval_modes) {
    case CEED_EVAL_NONE:
      *basis_ptr = identity;
      break;
    case CEED_EVAL_INTERP:
      *basis_ptr = interp;
      break;
    case CEED_EVAL_GRAD:
      *basis_ptr = grad;
      break;
    case CEED_EVAL_DIV:
      *basis_ptr = div;
      break;
    case CEED_EVAL_CURL:
      *basis_ptr = curl;
      break;
    case CEED_EVAL_WEIGHT:
      break;  // Caught by QF assembly
  }
}

//------------------------------------------------------------------------------
// Core code for diagonal assembly
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BLOCK_SIZE) __global__
    void LinearDiagonal(const CeedInt num_elem, const CeedScalar *identity, const CeedScalar *interp_in, const CeedScalar *grad_in,
                        const CeedScalar *div_in, const CeedScalar *curl_in, const CeedScalar *interp_out, const CeedScalar *grad_out,
                        const CeedScalar *div_out, const CeedScalar *curl_out, const CeedEvalMode *eval_modes_in, const CeedEvalMode *eval_modes_out,
                        const CeedScalar *__restrict__ assembled_qf_array, CeedScalar *__restrict__ elem_diag_array) {
  const int tid = threadIdx.x;  // Running with P threads

  if (tid >= NUM_NODES) return;

  // Compute the diagonal of B^T D B
  // Each element
  for (IndexType e = blockIdx.x * blockDim.z + threadIdx.z; e < num_elem; e += gridDim.x * blockDim.z) {
    // Each basis eval mode pair
    IndexType    d_out               = 0;
    CeedEvalMode eval_modes_out_prev = CEED_EVAL_NONE;

    for (IndexType e_out = 0; e_out < NUM_EVAL_MODES_OUT; e_out++) {
      IndexType         d_in               = 0;
      CeedEvalMode      eval_modes_in_prev = CEED_EVAL_NONE;
      const CeedScalar *b_t                = NULL;

      GetBasisPointer(&b_t, eval_modes_out[e_out], identity, interp_out, grad_out, div_out, curl_out);
      if (e_out == 0 || eval_modes_out[e_out] != eval_modes_out_prev) d_out = 0;
      else b_t = &b_t[(++d_out) * NUM_QPTS * NUM_NODES];
      eval_modes_out_prev = eval_modes_out[e_out];

      for (IndexType e_in = 0; e_in < NUM_EVAL_MODES_IN; e_in++) {
        const CeedScalar *b = NULL;

        GetBasisPointer(&b, eval_modes_in[e_in], identity, interp_in, grad_in, div_in, curl_in);
        if (e_in == 0 || eval_modes_in[e_in] != eval_modes_in_prev) d_in = 0;
        else b = &b[(++d_in) * NUM_QPTS * NUM_NODES];
        eval_modes_in_prev = eval_modes_in[e_in];

        // Each component
        for (IndexType comp_out = 0; comp_out < NUM_COMP; comp_out++) {
#if USE_POINT_BLOCK
          // Point block diagonal
          for (IndexType comp_in = 0; comp_in < NUM_COMP; comp_in++) {
            CeedScalar e_value = 0.;

            // Each qpoint/node pair
            for (IndexType q = 0; q < NUM_QPTS; q++) {
              const CeedScalar qf_value =
                  assembled_qf_array[((((e_in * NUM_COMP + comp_in) * NUM_EVAL_MODES_OUT + e_out) * NUM_COMP + comp_out) * num_elem + e) * NUM_QPTS +
                                     q];

              e_value += b_t[q * NUM_NODES + tid] * qf_value * b[q * NUM_NODES + tid];
            }
            elem_diag_array[((comp_out * NUM_COMP + comp_in) * num_elem + e) * NUM_NODES + tid] += e_value;
          }
#else
          // Diagonal only
          CeedScalar e_value = 0.;

          // Each qpoint/node pair
          for (IndexType q = 0; q < NUM_QPTS; q++) {
            const CeedScalar qf_value =
                assembled_qf_array[((((e_in * NUM_COMP + comp_out) * NUM_EVAL_MODES_OUT + e_out) * NUM_COMP + comp_out) * num_elem + e) * NUM_QPTS +
                                   q];

            e_value += b_t[q * NUM_NODES + tid] * qf_value * b[q * NUM_NODES + tid];
          }
          elem_diag_array[(comp_out * num_elem + e) * NUM_NODES + tid] += e_value;
#endif
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

#endif  // CEED_HIP_REF_OPERATOR_ASSEMBLE_DIAGONAL_H
