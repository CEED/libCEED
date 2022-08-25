// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

template <int SIZE>
//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
inline __device__ void readQuads(const CeedInt quad, const CeedInt num_qpts, const CeedScalar* d_u, CeedScalar* r_u) {
  for (CeedInt comp = 0; comp < SIZE; comp++) {
    r_u[comp] = d_u[quad + num_qpts * comp];
  }
}

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline __device__ void writeQuads(const CeedInt quad, const CeedInt num_qpts, const CeedScalar* r_v, CeedScalar* d_v) {
  for (CeedInt comp = 0; comp < SIZE; comp++) {
    d_v[quad + num_qpts * comp] = r_v[comp];
  }
}

//------------------------------------------------------------------------------
