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
#include "ceed/ceed-f64.h"
#include "ceed/types.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

typedef struct VortexsheddingContext_ *VortexsheddingContext;
struct VortexsheddingContext_ {
  bool       implicit; // !< Using implicit timesteping or not
  bool       weakT;    // !< flag to set Temperature weakly at inflow
  CeedScalar U_in;     // !< Velocity at inflow
  CeedScalar T_in;     // !< Temperature at inflow
  CeedScalar P0;       // !< Pressure at outflow
  CeedScalar L;        // !< Length of the rectangular channel
  CeedScalar H;        // !< Height of the rectangular channel
  CeedScalar d;        // !< Diameter of the cylinder
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

// *****************************************************************************
// This QFunction set the initial condition for the cylinder
// *****************************************************************************
CEED_QFUNCTION(ICsVortexshedding)(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // Context
  const VortexsheddingContext context = (VortexsheddingContext)ctx;
  const CeedScalar T_in        = context->T_in;
  const CeedScalar P0          = context->P0;
  const CeedScalar U_in        = context->U_in;
  const CeedScalar H           = context->H;
  const CeedScalar L           = context->L;
  const CeedScalar d           = context->d;
  const CeedScalar cv          = context->newtonian_ctx.cv;
  const CeedScalar mu          = context->newtonian_ctx.mu;
  const CeedScalar gamma       = HeatCapacityRatio(&context->newtonian_ctx);
  const CeedScalar e_internal  = cv * T_in;
  const CeedScalar rho         = P0 / ((gamma - 1) * e_internal);
  const CeedScalar Re          = (rho * U_in * d) / mu;

  // Quadrature point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};

    // State s

    CeedScalar q[5] = {0};
    UnpackState_U(s.U, q);
    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];

  } // End of Quadrature Point Loop
  return 0;
}




#endif // vortexshedding_h