// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <bc_definition.h>

/**
   @brief Create `BCDefinition`

   @param[in]  name             Name of the boundary condition
   @param[in]  num_label_values Number of `DMLabel` values
   @param[in]  label_values     Array of label values that define the boundaries controlled by the `BCDefinition`, size `num_label_values`
   @param[out] bc_def           The new `BCDefinition`
**/
PetscErrorCode BCDefinitionCreate(const char *name, PetscInt num_label_values, PetscInt label_values[], BCDefinition *bc_def) {
  PetscFunctionBeginUser;
  PetscCall(PetscNew(bc_def));

  PetscCall(PetscStrallocpy(name, &(*bc_def)->name));
  (*bc_def)->num_label_values = num_label_values;
  PetscCall(PetscMalloc1(num_label_values, &(*bc_def)->label_values));
  for (PetscInt i = 0; i < num_label_values; i++) (*bc_def)->label_values[i] = label_values[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Get base information for `BCDefinition`

   @param[in]  bc_def           `BCDefinition` to get information from
   @param[out] name             Name of the `BCDefinition`
   @param[out] num_label_values Number of `DMLabel` values
   @param[out] label_values     Array of label values that define the boundaries controlled by the `BCDefinition`, size `num_label_values`
**/
PetscErrorCode BCDefinitionGetInfo(BCDefinition bc_def, const char *name[], PetscInt *num_label_values, const PetscInt *label_values[]) {
  PetscFunctionBeginUser;
  if (name) *name = bc_def->name;
  if (label_values) {
    *num_label_values = bc_def->num_label_values;
    *label_values     = bc_def->label_values;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Destory a `BCDefinition` object

   @param[in,out] bc_def `BCDefinition` to be destroyed
**/
PetscErrorCode BCDefinitionDestroy(BCDefinition *bc_def) {
  PetscFunctionBeginUser;
  if ((*bc_def)->name) PetscCall(PetscFree((*bc_def)->name));
  if ((*bc_def)->label_values) PetscCall(PetscFree((*bc_def)->label_values));
  if ((*bc_def)->essential_comps) PetscCall(PetscFree((*bc_def)->essential_comps));
  PetscCall(PetscFree(*bc_def));
  *bc_def = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Set `DM_BC_ESSENTIAL` boundary condition values

   @param[in,out] bc_def              `BCDefinition` to set values to
   @param[in]     num_essential_comps Number of components to set
   @param[in]     essential_comps     Array of components to set, size `num_essential_comps`
**/
PetscErrorCode BCDefinitionSetEssential(BCDefinition bc_def, PetscInt num_essential_comps, PetscInt essential_comps[]) {
  PetscFunctionBeginUser;
  bc_def->num_essential_comps = num_essential_comps;
  PetscCall(PetscMalloc1(num_essential_comps, &bc_def->essential_comps));
  PetscCall(PetscArraycpy(bc_def->essential_comps, essential_comps, num_essential_comps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Get `DM_BC_ESSENTIAL` boundary condition values

   @param[in]  bc_def              `BCDefinition` to set values to
   @param[out] num_essential_comps Number of components to set
   @param[out] essential_comps     Array of components to set, size `num_essential_comps`
**/
PetscErrorCode BCDefinitionGetEssential(BCDefinition bc_def, PetscInt *num_essential_comps, const PetscInt *essential_comps[]) {
  PetscFunctionBeginUser;
  *num_essential_comps = bc_def->num_essential_comps;
  *essential_comps     = bc_def->essential_comps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define LABEL_ARRAY_SIZE 256

// @brief See `PetscOptionsBCDefinition`
PetscErrorCode PetscOptionsBCDefinition_Private(PetscOptionItems *PetscOptionsObject, const char opt[], const char text[], const char man[],
                                                const char name[], BCDefinition *bc_def, PetscBool *set) {
  PetscInt num_label_values = LABEL_ARRAY_SIZE, label_values[LABEL_ARRAY_SIZE] = {0};

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsIntArray(opt, text, man, label_values, &num_label_values, set));
  if (num_label_values > 0) {
    PetscCall(BCDefinitionCreate(name, num_label_values, label_values, bc_def));
  } else {
    *bc_def = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
