// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <petsc-ceed.h>
#include <petscdm.h>
#include <petscmat.h>
#include <petsc/private/petscimpl.h>

// MatCeed context for applying composite CeedOperator on a DM
typedef struct MatCeedContext_private *MatCeedContext;
struct MatCeedContext_private {
  Ceed           ceed;
  char          *name, *coo_mat_type;
  PetscMemType   mem_type;
  PetscInt       ref_count, num_mats_assembled_full, num_mats_assembled_pbd;
  PetscBool      is_destroyed, is_ceed_pbd_valid, is_ceed_vpbd_valid;
  PetscLogEvent  log_event_mult, log_event_mult_transpose, log_event_ceed_mult, log_event_ceed_mult_transpose;
  DM             dm_x, dm_y;
  Mat           *mats_assembled_full, *mats_assembled_pbd, mat_assembled_full_internal, mat_assembled_pbd_internal;
  Vec            X_loc, Y_loc_transpose;
  CeedVector     x_loc, y_loc, coo_values_full, coo_values_pbd;
  CeedOperator   op_mult, op_mult_transpose;
  PetscLogDouble flops_mult, flops_mult_transpose;
};

// Context data
PETSC_CEED_EXTERN PetscErrorCode MatCeedContextCreate(DM dm_x, DM dm_y, Vec X_loc, Vec Y_loc_transpose, CeedOperator op_mult,
                                                      CeedOperator op_mult_transpose, PetscLogEvent log_event_mult,
                                                      PetscLogEvent log_event_mult_transpose, PetscLogEvent log_event_ceed_mult,
                                                      PetscLogEvent log_event_ceed_mult_transpose, MatCeedContext *ctx);
PETSC_CEED_EXTERN PetscErrorCode MatCeedContextReference(MatCeedContext ctx);
PETSC_CEED_EXTERN PetscErrorCode MatCeedContextReferenceCopy(MatCeedContext ctx, MatCeedContext *ctx_copy);
PETSC_CEED_EXTERN PetscErrorCode MatCeedContextDestroy(MatCeedContext ctx);

// MatCEED
PETSC_CEED_EXTERN PetscErrorCode MatGetDiagonal_Ceed(Mat A, Vec D);
PETSC_CEED_EXTERN PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
PETSC_CEED_EXTERN PetscErrorCode MatMultTranspose_Ceed(Mat A, Vec Y, Vec X);

extern PetscClassId  MATCEED_CLASSID;
extern PetscLogEvent MATCEED_MULT, MATCEED_MULT_TRANSPOSE;
