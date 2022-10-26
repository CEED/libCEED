// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef vortexshedding_h
#define vortexshedding_h

#include <ceed.h>
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

typedef struct VortexsheddingContext_ *VortexsheddingContext;
struct VortexsheddingContext_ {
  bool       implicit; // !< Using implicit timesteping or not
  bool       weakT;    // !< flag to set Temperature weakly at inflow
  CeedScalar x_inflow; // !< Location of inflow in x
  CeedScalar P0;       // !< Pressure at outflow
  struct NewtonianIdealGasContext_ newtonian_ctx;
};









#endif // vortexshedding_h