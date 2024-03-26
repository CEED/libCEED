// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for particle-based methods with DMSwarm
#pragma once

#include <ceed.h>
#include <math.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petsc/private/petscfeimpl.h> /* For interpolation */

#include "petscutils.h"

// libCEED context data
typedef struct DMSwarmCeedContext_ *DMSwarmCeedContext;
struct DMSwarmCeedContext_ {
  Ceed         ceed;
  CeedVector   u_mesh, v_mesh, u_points;
  CeedOperator op_mass, op_mesh_to_points, op_points_to_mesh;
};

PetscErrorCode DMSwarmCeedContextCreate(DM dm_swarm, const char *ceed_resource, DMSwarmCeedContext *ctx);
PetscErrorCode DMSwarmCeedContextDestroy(DMSwarmCeedContext *ctx);

// Swarm point distribution
typedef enum { SWARM_GAUSS = 0, SWARM_UNIFORM = 1, SWARM_CELL_RANDOM = 2, SWARM_SINUSOIDAL = 3 } PointSwarmType;
static const char *const point_swarm_types[] = {"gauss", "uniform", "cell_random", "sinusoidal", "PointSwarmType", "SWARM", 0};

// Memory utilities
PetscErrorCode DMSwarmPICFieldP2C(DM dm_swarm, const char *field, CeedVector x_ceed);
PetscErrorCode DMSwarmPICFieldC2P(DM dm_swarm, const char *field, CeedVector x_ceed);

// Swarm helper function
PetscErrorCode DMSwarmInitalizePointLocations(DM dm_swarm, PointSwarmType point_swarm_type, PetscInt num_points, PetscInt num_points_per_cell);
PetscErrorCode DMSwarmCreateReferenceCoordinates(DM dm_swarm, IS *is_points, Vec *ref_coords);

// Swarm to mesh projection
PetscErrorCode DMSwarmCreateProjectionRHS(DM dm_swarm, const char *field, Vec U_points, Vec B_mesh);
PetscErrorCode MatMult_SwarmMass(Mat A, Vec U_mesh, Vec V_mesh);
PetscErrorCode DMSwarmProjectFromSwarmToCells(DM dm_swarm, const char *field, Vec U_points, Vec U_mesh);

PetscErrorCode SetupProblemSwarm(DM dm_swarm, Ceed ceed, BPData bp_data, CeedData data, PetscBool setup_rhs, Vec rhs, Vec target);
