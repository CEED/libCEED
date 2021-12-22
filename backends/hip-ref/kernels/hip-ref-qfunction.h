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

template <int SIZE>
//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
inline __device__ void readQuads(const CeedInt quad, const CeedInt num_qpts,
                                 const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt comp = 0; comp < SIZE; ++comp) {
    r_u[comp] = d_u[quad + num_qpts * comp];
  }
}

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline __device__ void writeQuads(const CeedInt quad, const CeedInt num_qpts,
                                  const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt comp = 0; comp < SIZE; ++comp) {
    d_v[quad + num_qpts * comp] = r_v[comp];
  }
}

//------------------------------------------------------------------------------
