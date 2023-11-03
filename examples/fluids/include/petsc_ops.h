// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef petsc_ops_h
#define petsc_ops_h

#include <ceed.h>
#include <petscdm.h>

#if PETSC_VERSION_GE(3, 21, 0)
#define DMProjectCoordinates(a, b) DMSetCoordinateDisc(a, b, PETSC_TRUE)
#endif

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
PetscErrorCode MatGetDiag_Ceed(Mat A, Vec D);
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
PetscErrorCode CreateMatShell_Ceed(OperatorApplyContext ctx, Mat *mat);

PetscErrorCode DMGetGlobalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);
PetscErrorCode DMGetLocalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);

PetscErrorCode VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode VecCopyP2C(Vec X_petsc, CeedVector x_ceed);
PetscErrorCode CeedOperatorCreateLocalVecs(CeedOperator op, VecType vec_type, MPI_Comm comm, Vec *input, Vec *output);
VecType        DMReturnVecType(DM dm);

PetscErrorCode ApplyCeedOperatorGlobalToGlobal(Vec X, Vec Y, OperatorApplyContext ctx);
PetscErrorCode ApplyCeedOperatorGlobalToLocal(Vec X, Vec Y_loc, OperatorApplyContext ctx);
PetscErrorCode ApplyCeedOperatorLocalToGlobal(Vec X_loc, Vec Y, OperatorApplyContext ctx);
PetscErrorCode ApplyCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx);
PetscErrorCode ApplyAddCeedOperatorLocalToLocal(Vec X_loc, Vec Y_loc, OperatorApplyContext ctx);

PetscErrorCode DMGetLocalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);
PetscErrorCode DMGetGlobalVectorInfo(DM dm, PetscInt *local_size, PetscInt *global_size, VecType *vec_type);
#endif  // petsc_ops_h
