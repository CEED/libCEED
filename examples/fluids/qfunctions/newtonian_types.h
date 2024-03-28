// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>

#include "stabilization_types.h"

typedef enum {
  STATEVAR_CONSERVATIVE = 0,
  STATEVAR_PRIMITIVE    = 1,
} StateVariable;

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
  CeedScalar        time;
  CeedScalar        ijacobian_time_shift;
  CeedScalar        P0;
  bool              is_implicit;
  StateVariable     state_var;
  StabilizationType stabilization;
  bool              idl_enable;
  CeedScalar        idl_amplitude;
  CeedScalar        idl_start;
  CeedScalar        idl_length;
};

typedef struct {
  CeedScalar pressure;
  CeedScalar velocity[3];
  CeedScalar temperature;
} StatePrimitive;

typedef struct SetupContext_ *SetupContext;
struct SetupContext_ {
  StatePrimitive                   reference;
  struct NewtonianIdealGasContext_ gas;
  CeedScalar                       lx;
  CeedScalar                       ly;
  CeedScalar                       lz;
  CeedScalar                       time;
};
