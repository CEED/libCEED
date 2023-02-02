// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
#ifndef READ_QUADS
#define READ_QUADS(ncomp, q, nq, d_u, r_u) \
  for (CeedInt comp = 0; comp < ncomp; comp++) r_u[comp] = d_u[q + nq * comp];
#endif

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
#ifndef WRITE_QUADS
#define WRITE_QUADS(ncomp, q, nq, r_v, d_v) \
  for (CeedInt comp = 0; comp < ncomp; comp++) d_v[q + nq * comp] = r_v[comp];
#endif
//------------------------------------------------------------------------------
