// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA operator full assembly
#ifndef CEED_CUDA_REF_OPERATOR_ASSEMBLE_H
#define CEED_CUDA_REF_OPERATOR_ASSEMBLE_H

#include <ceed.h>

#if USE_CEEDSIZE
typedef CeedSize IndexType;
#else
typedef CeedInt IndexType;
#endif

//------------------------------------------------------------------------------
// Matrix assembly kernel for low-order elements (2D thread block)
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BLOCK_SIZE) __global__
    void linearAssemble(const CeedScalar *B_in, const CeedScalar *B_out, const CeedScalar *__restrict__ qf_array,
                        CeedScalar *__restrict__ values_array) {
  // This kernel assumes B_in and B_out have the same number of quadrature points and basis points.
  // TODO: expand to more general cases
  const int i = threadIdx.x;  // The output row index of each B^TDB operation
  const int l = threadIdx.y;  // The output column index of each B^TDB operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il

  // Strides for final output ordering, determined by the reference (interface) implementation of the symbolic assembly, slowest --> fastest: element,
  // comp_in, comp_out, node_row, node_col
  const IndexType comp_out_stride = NUM_NODES * NUM_NODES;
  const IndexType comp_in_stride  = comp_out_stride * NUM_COMP;
  const IndexType e_stride        = comp_in_stride * NUM_COMP;
  // Strides for QF array, slowest --> fastest:  e_mode_in, comp_in, e_mode_out, comp_out, elem, qpt
  const IndexType q_e_stride          = NUM_QPTS;
  const IndexType q_comp_out_stride   = NUM_ELEM * q_e_stride;
  const IndexType q_e_mode_out_stride = q_comp_out_stride * NUM_COMP;
  const IndexType q_comp_in_stride    = q_e_mode_out_stride * NUM_E_MODE_OUT;
  const IndexType q_e_mode_in_stride  = q_comp_in_stride * NUM_COMP;

  // Loop over each element (if necessary)
  for (IndexType e = blockIdx.x * blockDim.z + threadIdx.z; e < NUM_ELEM; e += gridDim.x * blockDim.z) {
    for (IndexType comp_in = 0; comp_in < NUM_COMP; comp_in++) {
      for (IndexType comp_out = 0; comp_out < NUM_COMP; comp_out++) {
        CeedScalar result        = 0.0;
        IndexType  qf_index_comp = q_comp_in_stride * comp_in + q_comp_out_stride * comp_out + q_e_stride * e;

        for (IndexType e_mode_in = 0; e_mode_in < NUM_E_MODE_IN; e_mode_in++) {
          IndexType b_in_index = e_mode_in * NUM_QPTS * NUM_NODES;

          for (IndexType e_mode_out = 0; e_mode_out < NUM_E_MODE_OUT; e_mode_out++) {
            IndexType b_out_index = e_mode_out * NUM_QPTS * NUM_NODES;
            IndexType qf_index    = qf_index_comp + q_e_mode_out_stride * e_mode_out + q_e_mode_in_stride * e_mode_in;

            // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
            for (IndexType j = 0; j < NUM_QPTS; j++) {
              result += B_out[b_out_index + j * NUM_NODES + i] * qf_array[qf_index + j] * B_in[b_in_index + j * NUM_NODES + l];
            }
          }  // end of e_mode_out
        }    // end of e_mode_in
        IndexType val_index = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + NUM_NODES * i + l;

        values_array[val_index] = result;
      }  // end of out component
    }    // end of in component
  }      // end of element loop
}

//------------------------------------------------------------------------------
// Fallback kernel for larger orders (1D thread block)
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BLOCK_SIZE) __global__
    void linearAssembleFallback(const CeedScalar *B_in, const CeedScalar *B_out, const CeedScalar *__restrict__ qf_array,
                                CeedScalar *__restrict__ values_array) {
  // This kernel assumes B_in and B_out have the same number of quadrature points and basis points.
  // TODO: expand to more general cases
  const int l = threadIdx.x;  // The output column index of each B^TDB operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il

  // Strides for final output ordering, determined by the reference (interface) implementation of the symbolic assembly, slowest --> fastest: element,
  // comp_in, comp_out, node_row, node_col
  const IndexType comp_out_stride = NUM_NODES * NUM_NODES;
  const IndexType comp_in_stride  = comp_out_stride * NUM_COMP;
  const IndexType e_stride        = comp_in_stride * NUM_COMP;
  // Strides for QF array, slowest --> fastest:  e_mode_in, comp_in, e_mode_out, comp_out, elem, qpt
  const IndexType q_e_stride          = NUM_QPTS;
  const IndexType q_comp_out_stride   = NUM_ELEM * q_e_stride;
  const IndexType q_e_mode_out_stride = q_comp_out_stride * NUM_COMP;
  const IndexType q_comp_in_stride    = q_e_mode_out_stride * NUM_E_MODE_OUT;
  const IndexType q_e_mode_in_stride  = q_comp_in_stride * NUM_COMP;

  // Loop over each element (if necessary)
  for (IndexType e = blockIdx.x * blockDim.z + threadIdx.z; e < NUM_ELEM; e += gridDim.x * blockDim.z) {
    for (IndexType comp_in = 0; comp_in < NUM_COMP; comp_in++) {
      for (IndexType comp_out = 0; comp_out < NUM_COMP; comp_out++) {
        for (IndexType i = 0; i < NUM_NODES; i++) {
          CeedScalar result        = 0.0;
          IndexType  qf_index_comp = q_comp_in_stride * comp_in + q_comp_out_stride * comp_out + q_e_stride * e;

          for (IndexType e_mode_in = 0; e_mode_in < NUM_E_MODE_IN; e_mode_in++) {
            IndexType b_in_index = e_mode_in * NUM_QPTS * NUM_NODES;

            for (IndexType e_mode_out = 0; e_mode_out < NUM_E_MODE_OUT; e_mode_out++) {
              IndexType b_out_index = e_mode_out * NUM_QPTS * NUM_NODES;
              IndexType qf_index    = qf_index_comp + q_e_mode_out_stride * e_mode_out + q_e_mode_in_stride * e_mode_in;

              // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
              for (IndexType j = 0; j < NUM_QPTS; j++) {
                result += B_out[b_out_index + j * NUM_NODES + i] * qf_array[qf_index + j] * B_in[b_in_index + j * NUM_NODES + l];
              }
            }  // end of e_mode_out
          }    // end of e_mode_in
          IndexType val_index = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + NUM_NODES * i + l;

          values_array[val_index] = result;
        }  // end of loop over element node index, i
      }    // end of out component
    }      // end of in component
  }        // end of element loop
}

//------------------------------------------------------------------------------

#endif  // CEED_CUDA_REF_OPERATOR_ASSEMBLE_H
