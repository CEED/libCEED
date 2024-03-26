// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <c_client.h>
#include <petscsys.h>
#include <sr_enums.h>

#if defined(__clang_analyzer__)
void PetscSmartRedisCall(SRError);
#else
#define PetscSmartRedisCall(...)                                                                                                      \
  do {                                                                                                                                \
    SRError   ierr_smartredis_call_q_;                                                                                                \
    PetscBool disable_calls = PETSC_FALSE;                                                                                            \
    PetscStackUpdateLine;                                                                                                             \
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-smartsim_disable_calls", &disable_calls, NULL));                                      \
    if (disable_calls == PETSC_TRUE) break;                                                                                           \
    ierr_smartredis_call_q_ = __VA_ARGS__;                                                                                            \
    if (PetscUnlikely(ierr_smartredis_call_q_ != SRNoError))                                                                          \
      SETERRQ(PETSC_COMM_SELF, ierr_smartredis_call_q_, "SmartRedis Error (Code %d): %s", ierr_smartredis_call_q_, SRGetLastError()); \
  } while (0)
#endif

PetscErrorCode SmartRedisVerifyPutTensor(void *c_client, const char *name, const size_t name_length);
