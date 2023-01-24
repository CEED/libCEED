// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef turb_stats_types_h
#define turb_stats_types_h

#include "./newtonian_types.h"

enum TurbComponent {
  TURB_MEAN_DENSITY,
  TURB_MEAN_PRESSURE,
  TURB_MEAN_PRESSURE_SQUARED,
  TURB_MEAN_PRESSURE_VELOCITY_X,
  TURB_MEAN_PRESSURE_VELOCITY_Y,
  TURB_MEAN_PRESSURE_VELOCITY_Z,
  TURB_MEAN_DENSITY_TEMPERATURE,
  TURB_MEAN_DENSITY_TEMPERATURE_FLUX_X,
  TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Y,
  TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Z,
  TURB_MEAN_MOMENTUM_X,
  TURB_MEAN_MOMENTUM_Y,
  TURB_MEAN_MOMENTUM_Z,
  TURB_MEAN_MOMENTUMFLUX_XX,
  TURB_MEAN_MOMENTUMFLUX_YY,
  TURB_MEAN_MOMENTUMFLUX_ZZ,
  TURB_MEAN_MOMENTUMFLUX_YZ,
  TURB_MEAN_MOMENTUMFLUX_XZ,
  TURB_MEAN_MOMENTUMFLUX_XY,
  TURB_MEAN_VELOCITY_X,
  TURB_MEAN_VELOCITY_Y,
  TURB_MEAN_VELOCITY_Z,
  TURB_NUM_COMPONENTS,
};

typedef struct Turbulence_SpanStatsContext_ *Turbulence_SpanStatsContext;
struct Turbulence_SpanStatsContext_ {
  CeedScalar                       solution_time;
  CeedScalar                       previous_time;
  struct NewtonianIdealGasContext_ gas;
};

#endif  // turb_stats_types_h
