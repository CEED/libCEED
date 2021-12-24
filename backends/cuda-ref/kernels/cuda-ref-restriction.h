// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedNoTranspose(const CeedInt num_elem,
                                              const CeedScalar *__restrict__ u,
                                              CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < num_elem*RESTR_ELEM_SIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RESTR_ELEM_SIZE;
    const CeedInt elem = node / RESTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[loc_node + comp*RESTR_ELEM_SIZE*RESTR_NUM_ELEM +
        elem*RESTR_ELEM_SIZE] =
          u[loc_node*RESTR_STRIDE_NODES + comp*RESTR_STRIDE_COMP + elem*RESTR_STRIDE_ELEM];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetNoTranspose(const CeedInt num_elem,
                                             const CeedInt *__restrict__ indices,
                                             const CeedScalar *__restrict__ u,
                                             CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < num_elem*RESTR_ELEM_SIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt ind = indices[node];
    const CeedInt loc_node = node % RESTR_ELEM_SIZE;
    const CeedInt elem = node / RESTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[loc_node + comp*RESTR_ELEM_SIZE*RESTR_NUM_ELEM +
        elem*RESTR_ELEM_SIZE] =
          u[ind + comp*RESTR_COMP_STRIDE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedTranspose(const CeedInt num_elem,
    const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < num_elem*RESTR_ELEM_SIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RESTR_ELEM_SIZE;
    const CeedInt elem = node / RESTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[loc_node*RESTR_STRIDE_NODES + comp*RESTR_STRIDE_COMP + elem*RESTR_STRIDE_ELEM] +=
          u[loc_node + comp*RESTR_ELEM_SIZE*RESTR_NUM_ELEM +
            elem*RESTR_ELEM_SIZE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetTranspose(const CeedInt *__restrict__ lvec_indices,
                                           const CeedInt *__restrict__ t_indices,
                                           const CeedInt *__restrict__ t_offsets,
                                           const CeedScalar *__restrict__ u,
                                           CeedScalar *__restrict__ v) {
  CeedScalar value[RESTR_NUM_COMP];

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x;
       i < RESTR_NUM_NODES;
       i += blockDim.x * gridDim.x) {
    const CeedInt ind = lvec_indices[i];
    const CeedInt range_1 = t_offsets[i];
    const CeedInt range_N = t_offsets[i+1];

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      value[comp] = 0.0;

    for (CeedInt j = range_1; j < range_N; j++) {
      const CeedInt t_ind = t_indices[j];
      CeedInt loc_node = t_ind % RESTR_ELEM_SIZE;
      CeedInt elem = t_ind / RESTR_ELEM_SIZE;

      for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
        value[comp] += u[loc_node + comp*RESTR_ELEM_SIZE*RESTR_NUM_ELEM +
                         elem*RESTR_ELEM_SIZE];
    }

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[ind + comp*RESTR_COMP_STRIDE] += value[comp];
  }
}

//------------------------------------------------------------------------------
