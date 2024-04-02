// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for using PETSc with libCEED
#pragma once

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

CeedMemType      MemTypeP2C(PetscMemType mtype);
PetscErrorCode   VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode   VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode   VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode   VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode   Kershaw(DM dm_orig, PetscScalar eps);
PetscErrorCode   SetupDMByDegree(DM dm, PetscInt p_degree, PetscInt q_extra, PetscInt num_comp_u, PetscInt topo_dim, bool enforce_bc);
PetscErrorCode   CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type);
PetscErrorCode   DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field);
PetscErrorCode   BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt face, PetscFE fe,
                                           PetscTabulation basis_tabulation, PetscQuadrature quadrature, CeedBasis *basis);
PetscErrorCode   CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field, BPData bp_data,
                                     CeedBasis *basis);
PetscErrorCode   CreateDistributedDM(RunParams rp, DM *dm);

/**
  @brief Translate array of `PetscInt` to `CeedInt`.
    If the types differ, `array_petsc` is freed with `PetscFree()` and `array_ceed` is allocated with `PetscMalloc1()`.
    Caller is responsible for freeing `array_ceed` with `PetscFree()`.

  Not collective across MPI processes.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_petsc  Array of `PetscInt`
  @param[out]     array_ceed   Array of `CeedInt`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode IntArrayCeedToPetsc(PetscInt num_entries, CeedInt **array_ceed, PetscInt **array_petsc) {
  const CeedInt  int_c = 0;
  const PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_petsc = (PetscInt *)*array_ceed;
  } else {
    *array_petsc = malloc(num_entries * sizeof(PetscInt));
    for (PetscInt i = 0; i < num_entries; i++) (*array_petsc)[i] = (*array_ceed)[i];
    free(*array_ceed);
  }
  *array_ceed = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Translate array of `PetscInt` to `CeedInt`.
    If the types differ, `array_petsc` is freed with `PetscFree()` and `array_ceed` is allocated with `PetscMalloc1()`.
    Caller is responsible for freeing `array_ceed` with `PetscFree()`.

  Not collective across MPI processes.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_petsc  Array of `PetscInt`
  @param[out]     array_ceed   Array of `CeedInt`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode IntArrayPetscToCeed(PetscInt num_entries, PetscInt **array_petsc, CeedInt **array_ceed) {
  const CeedInt  int_c = 0;
  const PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_ceed = (CeedInt *)*array_petsc;
  } else {
    PetscCall(PetscMalloc1(num_entries, array_ceed));
    for (PetscInt i = 0; i < num_entries; i++) (*array_ceed)[i] = (*array_petsc)[i];
    PetscCall(PetscFree(*array_petsc));
  }
  *array_petsc = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
