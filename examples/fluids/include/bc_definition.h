// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <petsc.h>

typedef struct _p_BCDefinition *BCDefinition;
struct _p_BCDefinition {
  char *name;

  // Boundary ID information
  PetscInt num_label_values, *label_values, dm_field;

  // Essential Boundary information
  PetscInt num_essential_comps, *essential_comps;
};

/**
   @brief Creates a `BCDefinition` from an array of integers in an option in the database

   Must be between `PetscOptionsBegin()` and `PetscOptionsEnd()`.

   @param[in]  opt    The option one is seeking
   @param[in]  text   Short string describing option
   @param[in]  man    Manual page for the option
   @param[in]  name   String that sets the name of the `BCDefinition`
   @param[out] bc_def Resulting `BCDefinition`, `NULL` if option is not set
   @param[out] set    `PETSC_TRUE` if found, else `PETSC_FALSE`
**/
#define PetscOptionsBCDefinition(opt, text, man, name, bc_def, set) \
  PetscOptionsBCDefinition_Private(PetscOptionsObject, opt, text, man, name, bc_def, set)
PetscErrorCode PetscOptionsBCDefinition_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[],
                                                const char name[], BCDefinition *bc_def, PetscBool *set);

PetscErrorCode BCDefinitionCreate(const char *name, PetscInt num_label_values, PetscInt label_values[], BCDefinition *bc_def);
PetscErrorCode BCDefinitionGetInfo(BCDefinition bc_def, const char *name[], PetscInt *num_label_values, const PetscInt *label_values[]);
PetscErrorCode BCDefinitionDestroy(BCDefinition *bc_def);

PetscErrorCode BCDefinitionSetEssential(BCDefinition bc_def, PetscInt num_essential_comps, PetscInt essential_comps[]);
PetscErrorCode BCDefinitionGetEssential(BCDefinition bc_def, PetscInt *num_essential_comps, const PetscInt *essential_comps[]);
