// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <petscdm.h>
#include <petscksp.h>

typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  DM           dm_x, dm_y;
  Vec          X_loc, Y_loc;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op;
  Ceed         ceed;
};

PetscErrorCode OperatorApplyContextCreate(DM dm_x, DM dm_y, Ceed ceed, CeedOperator op_apply, CeedVector x_ceed, CeedVector y_ceed, Vec X_loc,
                                          Vec Y_loc, OperatorApplyContext *op_apply_ctx);
PetscErrorCode OperatorApplyContextDestroy(OperatorApplyContext op_apply_ctx);

PetscErrorCode DMGetGlobalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);
PetscErrorCode DMGetLocalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);

PetscErrorCode CeedOperatorCreateLocalVecs(CeedOperator op, VecType vec_type, MPI_Comm comm, Vec *input, Vec *output);
VecType        DMReturnVecType(DM dm);

PetscErrorCode ApplyCeedOperatorGlobalToGlobal(Vec X, Vec Y, OperatorApplyContext ctx);
PetscErrorCode ApplyCeedOperatorGlobalToLocal(Vec X, Vec Y_loc, OperatorApplyContext ctx);
PetscErrorCode ApplyCeedOperatorLocalToGlobal(Vec X_loc, Vec Y, OperatorApplyContext ctx);
PetscErrorCode ApplyCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx);
PetscErrorCode ApplyAddCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx);

PetscErrorCode DMGetLocalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);
PetscErrorCode DMGetGlobalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);

PetscErrorCode CreateSolveOperatorsFromMatCeed(KSP ksp, Mat mat_ceed, PetscBool assemble, Mat *Amat, Mat *Pmat);
PetscErrorCode KSPSetFromOptions_WithMatCeed(KSP ksp, Mat mat_ceed);
