// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef newtonian_types_h
#define newtonian_types_h

#include <ceed.h>

#include "stabilization_types.h"

typedef enum {
  STATEVAR_CONSERVATIVE = 0,
  STATEVAR_PRIMITIVE    = 1,
} StateVariable;

// For use with PetscOptionsEnum
static const char *const StateVariables[] = {"CONSERVATIVE", "PRIMITIVE", "StateVariable", "STATEVAR_", NULL};

typedef struct SetupContext_ *SetupContext;
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
  int        wind_type;               // See WindType: 0=ROTATION, 1=TRANSLATION
  int        bubble_type;             // See BubbleType: 0=SPHERE, 1=CYLINDER
  int        bubble_continuity_type;  // See BubbleContinuityType: 0=SMOOTH, 1=BACK_SHARP 2=THICK
};

typedef struct NewtonianIdealGasContext_ *NewtonianIdealGasContext;
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

#endif  // newtonian_types_h
