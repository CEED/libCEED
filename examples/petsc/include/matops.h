// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operations for PETSc MatShell
#pragma once

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode SetupApplyOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data, Vec X_loc, OperatorApplyContext op_apply_ctx);
PetscErrorCode SetupErrorOperatorCtx(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data, Vec X_loc, CeedOperator op_error,
                                     OperatorApplyContext op_error_ctx);
PetscErrorCode MatGetDiag(Mat A, Vec D);
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, OperatorApplyContext op_apply_ctx);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx);
PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y);
PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y);
PetscErrorCode ComputeL2Error(Vec X, PetscScalar *l2_error, OperatorApplyContext op_error_ctx);
