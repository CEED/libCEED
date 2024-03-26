// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <petscdm.h>
#include <petscmat.h>

#define MATCEED "ceed"

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

// Context data
MATCEED_INTERN PetscErrorCode MatCeedCreate(DM dm_x, DM dm_y, CeedOperator op_mult, CeedOperator op_mult_transpose, Mat *mat);
MATCEED_INTERN PetscErrorCode MatCeedCopy(Mat mat_ceed, Mat mat_other);
MATCEED_INTERN PetscErrorCode MatCeedAssembleCOO(Mat mat_ceed, Mat mat_coo);
MATCEED_INTERN PetscErrorCode MatCeedSetContext(Mat mat, PetscErrorCode (*f)(void *), void *ctx);
MATCEED_INTERN PetscErrorCode MatCeedGetContext(Mat mat, void *ctx);
MATCEED_INTERN PetscErrorCode MatCeedSetInnerMatType(Mat mat, MatType type);
MATCEED_INTERN PetscErrorCode MatCeedGetInnerMatType(Mat mat, MatType *type);
MATCEED_INTERN PetscErrorCode MatCeedSetOperation(Mat mat, MatOperation op, void (*g)(void));
MATCEED_INTERN PetscErrorCode MatCeedSetLocalVectors(Mat mat, Vec X_loc, Vec Y_loc_transpose);
MATCEED_INTERN PetscErrorCode MatCeedGetLocalVectors(Mat mat, Vec *X_loc, Vec *Y_loc_transpose);
MATCEED_INTERN PetscErrorCode MatCeedRestoreLocalVectors(Mat mat, Vec *X_loc, Vec *Y_loc_transpose);
MATCEED_INTERN PetscErrorCode MatCeedGetCeedOperators(Mat mat, CeedOperator *op_mult, CeedOperator *op_mult_transpose);
MATCEED_INTERN PetscErrorCode MatCeedRestoreCeedOperators(Mat mat, CeedOperator *op_mult, CeedOperator *op_mult_transpose);
MATCEED_INTERN PetscErrorCode MatCeedSetLogEvents(Mat mat, PetscLogEvent log_event_mult, PetscLogEvent log_event_mult_transpose);
MATCEED_INTERN PetscErrorCode MatCeedGetLogEvents(Mat mat, PetscLogEvent *log_event_mult, PetscLogEvent *log_event_mult_transpose);
