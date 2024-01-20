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

CEED_QFUNCTION_HELPER void QdataUnpack_ND(CeedInt N, CeedInt Q, CeedInt i, const CeedScalar *q_data, CeedScalar *wdetJ, CeedScalar *dXdx) {
  switch (N) {
    case 2:
      QdataUnpack_2D(Q, i, q_data, wdetJ, (CeedScalar(*)[2])dXdx);
      break;
    case 3:
      QdataUnpack_3D(Q, i, q_data, wdetJ, (CeedScalar(*)[3])dXdx);
      break;
  }
}

CEED_QFUNCTION_HELPER void StatePhysicalGradientFromReference_ND(CeedInt N, CeedInt Q, CeedInt i, NewtonianIdealGasContext gas, State s,
                                                                 StateVariable state_var, const CeedScalar *grad_q, const CeedScalar *dXdx,
                                                                 State *grad_s) {
  switch (N) {
    case 2: {
      for (CeedInt k = 0; k < 2; k++) {
        CeedScalar dqi[5];
        for (CeedInt j = 0; j < 5; j++) {
          dqi[j] = grad_q[(Q * 5) * 0 + Q * j + i] * dXdx[0 * N + k] + grad_q[(Q * 5) * 1 + Q * j + i] * dXdx[1 * N + k];
        }
        grad_s[k] = StateFromQ_fwd(gas, s, dqi, state_var);
      }
      CeedScalar U[5] = {0.};
      grad_s[2]       = StateFromU(gas, U);
    } break;
    case 3:
      StatePhysicalGradientFromReference(Q, i, gas, s, state_var, grad_q, (CeedScalar(*)[3])dXdx, grad_s);
      break;
  }
}

// *****************************************************************************
// This QFunction implements Advection for implicit time stepping method
// *****************************************************************************
CEED_QFUNCTION_HELPER void IFunction_AdvectionGeneric(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, CeedInt dim) {
  const CeedScalar(*q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*grad_q)            = in[1];
  const CeedScalar(*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*q_data)            = in[3];

  CeedScalar(*v)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  CeedScalar *jac_data               = out[2];

  AdvectionContext                 context   = (AdvectionContext)ctx;
  const CeedScalar                 CtauS     = context->CtauS;
  const CeedScalar                 zeros[14] = {0.};
  NewtonianIdealGasContext         gas;
  struct NewtonianIdealGasContext_ gas_struct = {0};
  gas                                         = &gas_struct;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromU(gas, qi);

    CeedScalar wdetJ, dXdx[9];
    QdataUnpack_ND(dim, Q, i, q_data, &wdetJ, dXdx);
    State grad_s[3];
    StatePhysicalGradientFromReference_ND(dim, Q, i, gas, s, STATEVAR_CONSERVATIVE, grad_q, dXdx, grad_s);

    const CeedScalar Grad_E[3] = {grad_s[0].U.E_total, grad_s[1].U.E_total, grad_s[2].U.E_total};

    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < dim; j++) grad_v[j][f][i] = 0;  // No Change in density or momentum
      v[f][i] = wdetJ * q_dot[f][i];                          // K Mass/transient term
    }

    CeedScalar div_u = 0;
    for (CeedInt j = 0; j < dim; j++) {
      for (CeedInt k = 0; k < dim; k++) {
        div_u += grad_s[k].Y.velocity[j];
      }
    }
    CeedScalar strong_conv = s.U.E_total * div_u + DotN(s.Y.velocity, Grad_E, dim);
    CeedScalar strong_res  = q_dot[4][i] + strong_conv;

    v[4][i] = wdetJ * q_dot[4][i];  // transient part (ALWAYS)

    CeedScalar uX[3] = {0.};
    MatVecNM(dXdx, s.Y.velocity, dim, dim, CEED_NOTRANSPOSE, uX);

    if (context->strong_form) {  // Strong Galerkin convection term: v div(E u)
      v[4][i] += wdetJ * strong_conv;
    } else {  // Weak Galerkin convection term: -dv \cdot (E u)
      for (CeedInt j = 0; j < dim; j++) grad_v[j][4][i] = -wdetJ * s.U.E_total * uX[j];
    }

    const CeedScalar TauS = CtauS / sqrt(Dot3(uX, uX));
    for (CeedInt j = 0; j < dim; j++) switch (context->stabilization) {
        case STAB_NONE:
          break;
        case STAB_SU:
          grad_v[j][4][i] += wdetJ * TauS * strong_conv * uX[j];
          break;
        case STAB_SUPG:
          grad_v[j][4][i] += wdetJ * TauS * strong_res * uX[j];
          break;
      }
    StoredValuesPack(Q, i, 0, 14, zeros, jac_data);
  }
}

#endif  // advection_generic_h
