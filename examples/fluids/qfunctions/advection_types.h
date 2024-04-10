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
  BUBBLE_CONTINUITY_COSINE     = 3,  // Use cosine wave for smoothing
} BubbleContinuityType;

typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  CeedScalar           CtauS;
  bool                 strong_form;
  CeedScalar           E_wind;
  bool                 implicit;
  StabilizationType    stabilization;
  StabilizationTauType stabilization_tau;
  CeedScalar           Ctau_a;
  CeedScalar           Ctau_t;
  CeedScalar           dt;
  CeedScalar           diffusion_coeff;
};

typedef struct SetupContextAdv_ *SetupContextAdv;
struct SetupContextAdv_ {
  CeedScalar           rc;
  CeedScalar           lx;
  CeedScalar           ly;
  CeedScalar           lz;
  CeedScalar           wind[3];
  CeedScalar           time;
  WindType             wind_type;
  AdvectionICType      initial_condition_type;
  BubbleContinuityType bubble_continuity_type;
};
