// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef matops_h
#define matops_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>

// Data for PETSc Matshell
typedef struct OperatorApplyContext_ *MatopApplyContext;
struct OperatorApplyContext_ {
  DM           dm_x, dm_y;
  Vec          X_loc, Y_loc;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op;
  Ceed         ceed;
};

PetscErrorCode MatopApplyContextCreate(DM dm_x, DM dm_y, Ceed ceed, CeedOperator op_apply, CeedVector x_ceed, CeedVector y_ceed, Vec X_loc, Vec Y_loc,
                                       MatopApplyContext *op_apply_ctx);
PetscErrorCode MatopApplyContextDestroy(MatopApplyContext op_apply_ctx);
PetscErrorCode MatGetDiag_Ceed(Mat A, Vec D);
PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, MatopApplyContext op_apply_ctx);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);

PetscErrorCode VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);

#endif  // matops_h
