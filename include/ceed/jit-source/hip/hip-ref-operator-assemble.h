// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP operator full assembly
#ifndef CEED_HIP_REF_OPERATOR_ASSEMBLE_H
#define CEED_HIP_REF_OPERATOR_ASSEMBLE_H

#include <ceed.h>

#if USE_CEEDSIZE
typedef CeedSize IndexType;
#else
typedef CeedInt IndexType;
#endif

//------------------------------------------------------------------------------
// Matrix assembly kernel
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BLOCK_SIZE) __global__
    void LinearAssemble(const CeedInt num_elem, const CeedScalar *B_in, const CeedScalar *B_out, const bool *orients_in,
                        const CeedInt8 *curl_orients_in, const bool *orients_out, const CeedInt8 *curl_orients_out,
                        const CeedScalar *__restrict__ qf_array, CeedScalar *__restrict__ values_array) {
  extern __shared__ CeedScalar s_CT[];
  CeedScalar                  *s_C = s_CT + NUM_NODES_OUT * NUM_NODES_IN;

  const int l = threadIdx.x;  // The output column index of each B^T D B operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il

  // Strides for final output ordering, determined by the reference (interface) implementation of the symbolic assembly, slowest --> fastest: e,
  // comp_in, comp_out, node_row, node_col
  const IndexType comp_out_stride = NUM_NODES_OUT * NUM_NODES_IN;
  const IndexType comp_in_stride  = comp_out_stride * NUM_COMP_OUT;
  const IndexType e_stride        = comp_in_stride * NUM_COMP_IN;

  // Strides for QF array, slowest --> fastest: e_in, comp_in, e_out, comp_out, e, q
  const IndexType q_e_stride             = NUM_QPTS;
  const IndexType q_comp_out_stride      = num_elem * q_e_stride;
  const IndexType q_eval_mode_out_stride = q_comp_out_stride * NUM_COMP_OUT;
  const IndexType q_comp_in_stride       = q_eval_mode_out_stride * NUM_EVAL_MODES_OUT;
  const IndexType q_eval_mode_in_stride  = q_comp_in_stride * NUM_COMP_IN;

  // Loop over each element (if necessary)
  for (IndexType e = blockIdx.x * blockDim.z + threadIdx.z; e < num_elem; e += gridDim.x * blockDim.z) {
    for (IndexType comp_in = 0; comp_in < NUM_COMP_IN; comp_in++) {
      for (IndexType comp_out = 0; comp_out < NUM_COMP_OUT; comp_out++) {
        for (IndexType i = threadIdx.y; i < NUM_NODES_OUT; i += BLOCK_SIZE_Y) {
          CeedScalar result        = 0.0;
          IndexType  qf_index_comp = q_comp_in_stride * comp_in + q_comp_out_stride * comp_out + q_e_stride * e;

          for (IndexType e_in = 0; e_in < NUM_EVAL_MODES_IN; e_in++) {
            IndexType b_in_index = e_in * NUM_QPTS * NUM_NODES_IN;

            for (IndexType e_out = 0; e_out < NUM_EVAL_MODES_OUT; e_out++) {
              IndexType b_out_index = e_out * NUM_QPTS * NUM_NODES_OUT;
              IndexType qf_index    = qf_index_comp + q_eval_mode_out_stride * e_out + q_eval_mode_in_stride * e_in;

              // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
              for (IndexType j = 0; j < NUM_QPTS; j++) {
                result += B_out[b_out_index + j * NUM_NODES_OUT + i] * qf_array[qf_index + j] * B_in[b_in_index + j * NUM_NODES_IN + l];
              }
            }  // end of out eval mode
          }    // end of in eval mode
          if (orients_in) {
            result *= orients_in[NUM_NODES_IN * e + l] ? -1.0 : 1.0;
          }
          if (orients_out) {
            result *= orients_out[NUM_NODES_OUT * e + i] ? -1.0 : 1.0;
          }
          if (!curl_orients_in && !curl_orients_out) {
            IndexType val_index = e_stride * e + comp_in_stride * comp_in + comp_out_stride * comp_out + NUM_NODES_IN * i + l;

            values_array[val_index] = result;
          } else if (curl_orients_in) {
            s_C[NUM_NODES_IN * threadIdx.y + l] = result;
            __syncthreads();
            s_CT[NUM_NODES_IN * i + l] =
                (l > 0 ? s_C[NUM_NODES_IN * threadIdx.y + l - 1] * curl_orients_in[3 * NUM_NODES_IN * e + 3 * l - 1] : 0.0) +
                s_C[NUM_NODES_IN * threadIdx.y + l] * curl_orients_in[3 * NUM_NODES_IN * e + 3 * l + 1] +
                (l < (NUM_NODES_IN - 1) ? s_C[NUM_NODES_IN * threadIdx.y + l + 1] * curl_orients_in[3 * NUM_NODES_IN * e + 3 * l + 3] : 0.0);
          } else {
            s_CT[NUM_NODES_IN * i + l] = result;
          }
        }  // end of loop over element node index, i
        if (curl_orients_in || curl_orients_out) {
          // Compute and store the final T^T (B^T D B T) using the fully computed C T product in shared memory
          if (curl_orients_out) __syncthreads();
          for (IndexType i = threadIdx.y; i < NUM_NODES_OUT; i += BLOCK_SIZE_Y) {
            IndexType val_index = e_stride * e + comp_in_stride * comp_in + comp_out_stride * comp_out + NUM_NODES_IN * i + l;

            if (curl_orients_out) {
              values_array[val_index] =
                  (i > 0 ? s_CT[NUM_NODES_IN * (i - 1) + l] * curl_orients_out[3 * NUM_NODES_OUT * e + 3 * i - 1] : 0.0) +
                  s_CT[NUM_NODES_IN * i + l] * curl_orients_out[3 * NUM_NODES_OUT * e + 3 * i + 1] +
                  (i < (NUM_NODES_OUT - 1) ? s_CT[NUM_NODES_IN * (i + 1) + l] * curl_orients_out[3 * NUM_NODES_OUT * e + 3 * i + 3] : 0.0);
            } else {
              values_array[val_index] = s_CT[NUM_NODES_IN * i + l];
            }
          }
        }
      }  // end of out component
    }    // end of in component
  }      // end of element loop
}

//------------------------------------------------------------------------------

#endif  // CEED_HIP_REF_OPERATOR_ASSEMBLE_H
