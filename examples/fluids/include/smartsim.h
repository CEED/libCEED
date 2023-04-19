// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <c_client.h>
#include <petscsys.h>
#include <sr_enums.h>

#if defined(__clang_analyzer__)
void SmartRedisCall(SRError);
#else
#define SmartRedisCall(...)                                                                                                           \
  do {                                                                                                                                \
    SRError ierr_smartredis_call_q_;                                                                                                  \
    PetscStackUpdateLine;                                                                                                             \
    ierr_smartredis_call_q_ = __VA_ARGS__;                                                                                            \
    if (PetscUnlikely(ierr_smartredis_call_q_ != SRNoError))                                                                          \
      SETERRQ(PETSC_COMM_SELF, ierr_smartredis_call_q_, "SmartRedis Error (Code %d): %s", ierr_smartredis_call_q_, SRGetLastError()); \
  } while (0)
#endif

static PetscErrorCode SmartRedisVerifyPutTensor(void *c_client, const char *name, const size_t name_length) {
  bool does_exist = false;
  PetscFunctionBeginUser;

  SmartRedisCall(tensor_exists(c_client, name, name_length, &does_exist));
  PetscCheck(does_exist, PETSC_COMM_SELF, -1, "Tensor of name '%s' was not written to the database successfully", name);

  PetscFunctionReturn(0);
}
