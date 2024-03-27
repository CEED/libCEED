// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <petscdm.h>
#include <petscmat.h>

#if defined(__clang_analyzer__)
#define MATCEED_EXTERN extern
#elif defined(__cplusplus)
#define MATCEED_EXTERN extern "C"
#else
#define MATCEED_EXTERN extern
#endif

#if defined(__clang_analyzer__)
#define MATCEED_INTERN
#else
#define MATCEED_INTERN MATCEED_EXTERN __attribute__((visibility("hidden")))
#endif

/**
  @brief Calls a libCEED function and then checks the resulting error code.
  If the error code is non-zero, then a PETSc error is set with the libCEED error message.
**/
#ifndef PetscCallCeed
#define PetscCallCeed(ceed_, ...)                                   \
  do {                                                              \
    int ierr_q_ = __VA_ARGS__;                                      \
    if (ierr_q_ != CEED_ERROR_SUCCESS) {                            \
      const char *error_message;                                    \
      CeedGetErrorMessage(ceed_, &error_message);                   \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", error_message); \
    }                                                               \
  } while (0)
#endif

// MatCeed context for applying composite CeedOperator on a DM
typedef struct MatCeedContext_private *MatCeedContext;
struct MatCeedContext_private {
  Ceed           ceed;
  char          *name, *internal_mat_type;
  PetscMemType   mem_type;
  PetscInt       ref_count, num_mats_assembled_full, num_mats_assembled_pbd;
  PetscBool      is_destroyed, is_ceed_pbd_valid, is_ceed_vpbd_valid;
  PetscLogEvent  log_event_mult, log_event_mult_transpose;
  DM             dm_x, dm_y;
  Mat           *mats_assembled_full, *mats_assembled_pbd, mat_assembled_full_internal, mat_assembled_pbd_internal;
  Vec            X_loc, Y_loc_transpose;
  CeedVector     x_loc, y_loc, coo_values_full, coo_values_pbd;
  CeedOperator   op_mult, op_mult_transpose;
  PetscLogDouble flops_mult, flops_mult_transpose;
};

// Context data
MATCEED_INTERN PetscErrorCode MatCeedContextCreate(DM dm_x, DM dm_y, Vec X_loc, Vec Y_loc_transpose, CeedOperator op_mult,
                                                   CeedOperator op_mult_transpose, PetscLogEvent log_event_mult,
                                                   PetscLogEvent log_event_mult_transpose, MatCeedContext *ctx);
MATCEED_INTERN PetscErrorCode MatCeedContextReference(MatCeedContext ctx);
MATCEED_INTERN PetscErrorCode MatCeedContextReferenceCopy(MatCeedContext ctx, MatCeedContext *ctx_copy);
MATCEED_INTERN PetscErrorCode MatCeedContextDestroy(MatCeedContext ctx);

// Mat Ceed
MATCEED_INTERN PetscErrorCode MatGetDiagonal_Ceed(Mat A, Vec D);
MATCEED_INTERN PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y);
MATCEED_INTERN PetscErrorCode MatMultTranspose_Ceed(Mat A, Vec Y, Vec X);

extern PetscClassId  MATCEED_CLASSID;
extern PetscLogEvent MATCEED_MULT, MATCEED_MULT_TRANSPOSE;
