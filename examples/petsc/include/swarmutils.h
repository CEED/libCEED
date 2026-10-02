// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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

// Memory utilities
PetscErrorCode DMSwarmPICFieldP2C(DM dm_swarm, const char *field, CeedVector x_ceed);
PetscErrorCode DMSwarmPICFieldC2P(DM dm_swarm, const char *field, CeedVector x_ceed);

// Swarm helper function
PetscErrorCode DMSwarmInitalizePointLocations(DM dm_swarm, PointSwarmType swarm_type, PetscInt num_points, PetscInt num_points_per_cell_1d);
PetscErrorCode DMSwarmCreateReferenceCoordinates(DM dm_swarm, IS *is_points, Vec *X_points_ref);

// Swarm to mesh projection
PetscErrorCode DMSwarmCreateProjectionRHS(DM dm_swarm, const char *field, Vec U_points, Vec B_mesh);
PetscErrorCode MatMult_SwarmMass(Mat A, Vec U_mesh, Vec V_mesh);
PetscErrorCode DMSwarmProjectFromSwarmToCells(DM dm_swarm, const char *field, Vec U_points, Vec U_mesh);

PetscErrorCode SetupProblemSwarm(DM dm_swarm, Ceed ceed, BPData bp_data, CeedData data, PetscBool setup_rhs, Vec rhs, Vec target);
