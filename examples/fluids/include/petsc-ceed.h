// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <petscsys.h>

#if defined(__clang_analyzer__)
#define PETSC_CEED_EXTERN extern
#elif defined(__cplusplus)
#define PETSC_CEED_EXTERN extern "C"
#else
#define PETSC_CEED_EXTERN extern
#endif

#if defined(__clang_analyzer__)
#define PETSC_CEED_INTERN
#else
#define PETSC_CEED_INTERN PETSC_CEED_EXTERN __attribute__((visibility("hidden")))
#endif

/**
  @brief Calls a libCEED function and then checks the resulting error code.
  If the error code is non-zero, then a PETSc error is set with the libCEED error message.
**/
/// @ingroup RatelInternal
#ifndef PetscCallCeed
#define PetscCallCeed(ceed_, ...)                                   \
  do {                                                              \
    int ierr_q_;                                                    \
    PetscStackUpdateLine;                                           \
    ierr_q_ = __VA_ARGS__;                                          \
    if (PetscUnlikely(ierr_q_ != CEED_ERROR_SUCCESS)) {             \
      const char *error_message;                                    \
      CeedGetErrorMessage(ceed_, &error_message);                   \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", error_message); \
    }                                                               \
  } while (0)
#endif
