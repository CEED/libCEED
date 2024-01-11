// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef advection_types_h
#define advection_types_h

#include <ceed.h>
#include "stabilization_types.h"

typedef enum {
  WIND_ROTATION    = 0,
  WIND_TRANSLATION = 1,
} WindType;

// Advection - Initial Condition Types
typedef enum {
  ADVECTIONIC_BUBBLE_SPHERE   = 0,  // dim=3
  ADVECTIONIC_BUBBLE_CYLINDER = 1,  // dim=2
  ADVECTIONIC_COSINE_HILL     = 2,  // dim=2
  ADVECTIONIC_SKEW            = 3,
} AdvectionICType;

// Advection - Bubble Continuity Types
typedef enum {
  BUBBLE_CONTINUITY_SMOOTH     = 0,  // Original continuous, smooth shape
  BUBBLE_CONTINUITY_BACK_SHARP = 1,  // Discontinuous, sharp back half shape
  BUBBLE_CONTINUITY_THICK      = 2,  // Define a finite thickness
} BubbleContinuityType;

typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  CeedScalar        CtauS;
  CeedScalar        strong_form;
  CeedScalar        E_wind;
  bool              implicit;
  StabilizationType stabilization;
};

#endif /* ifndef advection_types_h */
