// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef newtonian_types_h
#define newtonian_types_h

#ifndef __OCCA__
#include <ceed.h>
#endif

#include "stabilization_types.h"

// typedef enum {
//   STATEVAR_CONSERVATIVE = 0,
//   STATEVAR_PRIMITIVE    = 1,
// } StateVariable;

#define StateVariable int
static const int STATEVAR_CONSERVATIVE = 0;
static const int STATEVAR_PRIMITIVE    = 1;

// For use with PetscOptionsEnum
static const char *const StateVariables[] = {"CONSERVATIVE", "PRIMITIVE", "StateVariable", "STATEVAR_", NULL};

struct SetupContext_ {
  CeedScalar theta0;
  CeedScalar thetaC;
  CeedScalar P0;
  CeedScalar N;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g[3];
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar lz;
  CeedScalar center[3];
  CeedScalar dc_axis[3];
  CeedScalar time;
  int        wind_type;
  int        bubble_type;
  int        bubble_continuity_type;
};
#define SetupContext struct SetupContext_ *

struct NewtonianIdealGasContext_ {
  CeedScalar        lambda;
  CeedScalar        mu;
  CeedScalar        k;
  CeedScalar        cv;
  CeedScalar        cp;
  CeedScalar        g[3];
  CeedScalar        c_tau;
  CeedScalar        Ctau_t;
  CeedScalar        Ctau_v;
  CeedScalar        Ctau_C;
  CeedScalar        Ctau_M;
  CeedScalar        Ctau_E;
  CeedScalar        dt;
  CeedScalar        ijacobian_time_shift;
  CeedScalar        P0;
  bool              is_implicit;
  StateVariable     state_var;
  StabilizationType stabilization;
};
#define NewtonianIdealGasContext struct NewtonianIdealGasContext_ *

#endif  // newtonian_types_h
