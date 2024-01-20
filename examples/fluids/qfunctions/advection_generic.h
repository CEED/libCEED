// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection_generic_h
#define advection_generic_h

#include <ceed.h>
#include <math.h>

#include "advection_types.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization_types.h"
#include "utils.h"

// *****************************************************************************
// This QFunction sets the initial conditions and the boundary conditions
//   for two test cases: ROTATION and TRANSLATION
//
// -- ROTATION (default)
//      Initial Conditions:
//        Mass Density:
//          Constant mass density of 1.0
//        Momentum Density:
//          Rotational field in x,y
//        Energy Density:
//          Maximum of 1. x0 decreasing linearly to 0. as radial distance
//            increases to (1.-r/rc), then 0. everywhere else
//
//      Boundary Conditions:
//        Mass Density:
//          0.0 flux
//        Momentum Density:
//          0.0
//        Energy Density:
//          0.0 flux
//
// -- TRANSLATION
//      Initial Conditions:
//        Mass Density:
//          Constant mass density of 1.0
//        Momentum Density:
//           Constant rectilinear field in x,y
//        Energy Density:
//          Maximum of 1. x0 decreasing linearly to 0. as radial distance
//            increases to (1.-r/rc), then 0. everywhere else
//
//      Boundary Conditions:
//        Mass Density:
//          0.0 flux
//        Momentum Density:
//          0.0
//        Energy Density:
//          Inflow BCs:
//            E = E_wind
//          Outflow BCs:
//            E = E(boundary)
//          Both In/Outflow BCs for E are applied weakly in the
//            QFunction "Advection2d_Sur"
//
// *****************************************************************************

// *****************************************************************************
// This helper function provides the exact, time-dependent solution and IC formulation for 2D advection
// *****************************************************************************
CEED_QFUNCTION_HELPER CeedInt Exact_AdvectionGeneric(CeedInt dim, CeedScalar time, const CeedScalar X[], CeedInt Nf, CeedScalar q[], void *ctx) {
  const SetupContextAdv context = (SetupContextAdv)ctx;
  const CeedScalar      rc      = context->rc;
  const CeedScalar      lx      = context->lx;
  const CeedScalar      ly      = context->ly;
  const CeedScalar      lz      = dim == 2 ? 0. : context->lz;
  const CeedScalar     *wind    = context->wind;

  const CeedScalar center[3] = {0.5 * lx, 0.5 * ly, 0.5 * lz};
  const CeedScalar theta     = dim == 2 ? M_PI / 3 : M_PI;
  const CeedScalar x0[3]     = {center[0] + .25 * lx * cos(theta + time), center[1] + .25 * ly * sin(theta + time), 0.5 * lz};

  const CeedScalar x = X[0], y = X[1], z = dim == 2 ? 0. : X[2];

  CeedScalar r = 0.;
  switch (context->initial_condition_type) {
    case ADVECTIONIC_BUBBLE_SPHERE:
    case ADVECTIONIC_BUBBLE_CYLINDER:
      r = sqrt(Square(x - x0[0]) + Square(y - x0[1]) + Square(z - x0[2]));
      break;
    case ADVECTIONIC_COSINE_HILL:
      r = sqrt(Square(x - center[0]) + Square(y - center[1]));
      break;
    case ADVECTIONIC_SKEW:
      break;
  }

  switch (context->wind_type) {
    case WIND_ROTATION:
      q[0] = 1.;
      q[1] = -(y - center[1]);
      q[2] = (x - center[0]);
      q[3] = 0;
      break;
    case WIND_TRANSLATION:
      q[0] = 1.;
      q[1] = wind[0];
      q[2] = wind[1];
      q[3] = dim == 2 ? 0. : wind[2];
      break;
    default:
      return 1;
  }

  switch (context->initial_condition_type) {
    case ADVECTIONIC_BUBBLE_SPHERE:
    case ADVECTIONIC_BUBBLE_CYLINDER:
      switch (context->bubble_continuity_type) {
        // original continuous, smooth shape
        case BUBBLE_CONTINUITY_SMOOTH:
          q[4] = r <= rc ? (1. - r / rc) : 0.;
          break;
        // discontinuous, sharp back half shape
        case BUBBLE_CONTINUITY_BACK_SHARP:
          q[4] = ((r <= rc) && (y < center[1])) ? (1. - r / rc) : 0.;
          break;
        // attempt to define a finite thickness that will get resolved under grid refinement
        case BUBBLE_CONTINUITY_THICK:
          q[4] = ((r <= rc) && (y < center[1])) ? (1. - r / rc) * fmin(1.0, (center[1] - y) / 1.25) : 0.;
          break;
        case BUBBLE_CONTINUITY_COSINE:
          q[4] = r <= rc ? .5 + .5 * cos(r * M_PI / rc) : 0;
          break;
      }
      break;
    case ADVECTIONIC_COSINE_HILL: {
      CeedScalar half_width = context->lx / 2;
      q[4]                  = r > half_width ? 0. : cos(2 * M_PI * r / half_width + M_PI) + 1.;
    } break;
    case ADVECTIONIC_SKEW: {
      CeedScalar       skewed_barrier[3]  = {wind[0], wind[1], 0};
      CeedScalar       inflow_to_point[3] = {x - context->lx / 2, y, 0};
      CeedScalar       cross_product[3]   = {0};
      const CeedScalar boundary_threshold = 20 * CEED_EPSILON;
      Cross3(skewed_barrier, inflow_to_point, cross_product);

      q[4] = cross_product[2] > boundary_threshold ? 0 : 1;
      if ((x < boundary_threshold && wind[0] < boundary_threshold) ||                // outflow at -x boundary
          (y < boundary_threshold && wind[1] < boundary_threshold) ||                // outflow at -y boundary
          (x > context->lx - boundary_threshold && wind[0] > boundary_threshold) ||  // outflow at +x boundary
          (y > context->ly - boundary_threshold && wind[1] > boundary_threshold)     // outflow at +y boundary
      ) {
        q[4] = 0;
      }
    } break;
  }
  return 0;
}

#endif  // advection_generic_h
