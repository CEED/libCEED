// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

//------------------------------------------------------------------------------
// Matrix assembly kernel for low-order elements (2D thread block)
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BLOCK_SIZE) __global__
    void linearAssemble(const CeedScalar *B_in, const CeedScalar *B_out, const CeedScalar *__restrict__ qf_array,
                        CeedScalar *__restrict__ values_array) {
  // This kernel assumes B_in and B_out have the same number of quadrature points and
  // basis points.
  // TODO: expand to more general cases
  const int i = threadIdx.x;  // The output row index of each B^TDB operation
  const int l = threadIdx.y;  // The output column index of each B^TDB operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il

  // Strides for final output ordering, determined by the reference (interface) implementation of
  // the symbolic assembly, slowest --> fastest: element, comp_in, comp_out, node_row, node_col
  const CeedInt comp_out_stride = NNODES * NNODES;
  const CeedInt comp_in_stride  = comp_out_stride * NCOMP;
  const CeedInt e_stride        = comp_in_stride * NCOMP;
  // Strides for QF array, slowest --> fastest:  emode_in, comp_in, emode_out, comp_out, elem, qpt
  const CeedInt qe_stride         = NQPTS;
  const CeedInt qcomp_out_stride  = NELEM * qe_stride;
  const CeedInt qemode_out_stride = qcomp_out_stride * NCOMP;
  const CeedInt qcomp_in_stride   = qemode_out_stride * NUMEMODEOUT;
  const CeedInt qemode_in_stride  = qcomp_in_stride * NCOMP;

  // Loop over each element (if necessary)
  for (CeedInt e = blockIdx.x * blockDim.z + threadIdx.z; e < NELEM; e += gridDim.x * blockDim.z) {
    for (CeedInt comp_in = 0; comp_in < NCOMP; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < NCOMP; comp_out++) {
        CeedScalar result        = 0.0;
        CeedInt    qf_index_comp = qcomp_in_stride * comp_in + qcomp_out_stride * comp_out + qe_stride * e;
        for (CeedInt emode_in = 0; emode_in < NUMEMODEIN; emode_in++) {
          CeedInt b_in_index = emode_in * NQPTS * NNODES;
          for (CeedInt emode_out = 0; emode_out < NUMEMODEOUT; emode_out++) {
            CeedInt b_out_index = emode_out * NQPTS * NNODES;
            CeedInt qf_index    = qf_index_comp + qemode_out_stride * emode_out + qemode_in_stride * emode_in;
            // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
            for (CeedInt j = 0; j < NQPTS; j++) {
              result += B_out[b_out_index + j * NNODES + i] * qf_array[qf_index + j] * B_in[b_in_index + j * NNODES + l];
            }
          }  // end of emode_out
        }    // end of emode_in
        CeedInt val_index       = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + NNODES * i + l;
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
  // This kernel assumes B_in and B_out have the same number of quadrature points and
  // basis points.
  // TODO: expand to more general cases
  const int l = threadIdx.x;  // The output column index of each B^TDB operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il

  // Strides for final output ordering, determined by the reference (interface) implementation of
  // the symbolic assembly, slowest --> fastest: element, comp_in, comp_out, node_row, node_col
  const CeedInt comp_out_stride = NNODES * NNODES;
  const CeedInt comp_in_stride  = comp_out_stride * NCOMP;
  const CeedInt e_stride        = comp_in_stride * NCOMP;
  // Strides for QF array, slowest --> fastest:  emode_in, comp_in, emode_out, comp_out, elem, qpt
  const CeedInt qe_stride         = NQPTS;
  const CeedInt qcomp_out_stride  = NELEM * qe_stride;
  const CeedInt qemode_out_stride = qcomp_out_stride * NCOMP;
  const CeedInt qcomp_in_stride   = qemode_out_stride * NUMEMODEOUT;
  const CeedInt qemode_in_stride  = qcomp_in_stride * NCOMP;

  // Loop over each element (if necessary)
  for (CeedInt e = blockIdx.x * blockDim.z + threadIdx.z; e < NELEM; e += gridDim.x * blockDim.z) {
    for (CeedInt comp_in = 0; comp_in < NCOMP; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < NCOMP; comp_out++) {
        for (CeedInt i = 0; i < NNODES; i++) {
          CeedScalar result        = 0.0;
          CeedInt    qf_index_comp = qcomp_in_stride * comp_in + qcomp_out_stride * comp_out + qe_stride * e;
          for (CeedInt emode_in = 0; emode_in < NUMEMODEIN; emode_in++) {
            CeedInt b_in_index = emode_in * NQPTS * NNODES;
            for (CeedInt emode_out = 0; emode_out < NUMEMODEOUT; emode_out++) {
              CeedInt b_out_index = emode_out * NQPTS * NNODES;
              CeedInt qf_index    = qf_index_comp + qemode_out_stride * emode_out + qemode_in_stride * emode_in;
              // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
              for (CeedInt j = 0; j < NQPTS; j++) {
                result += B_out[b_out_index + j * NNODES + i] * qf_array[qf_index + j] * B_in[b_in_index + j * NNODES + l];
              }
            }  // end of emode_out
          }    // end of emode_in
          CeedInt val_index       = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + NNODES * i + l;
          values_array[val_index] = result;
        }  // end of loop over element node index, i
      }    // end of out component
    }      // end of in component
  }        // end of element loop
}

//------------------------------------------------------------------------------
