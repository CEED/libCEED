// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_TYPES_HEADER
#define CEED_OCCA_TYPES_HEADER

#include <ceed/backend.h>

#include <occa.hpp>

#define CeedOccaFromChk(ierr) \
  do {                        \
    if (ierr) {               \
      return NULL;            \
    }                         \
  } while (0)

#define CeedCallOcca(...)      \
  do {                         \
    int ierr_q_ = __VA_ARGS__; \
    CeedOccaFromChk(ierr_q_);  \
  } while (0);

#define CeedOccaValidChk(isValidVar, ierr) \
  do {                                     \
    if (ierr) {                            \
      isValidVar = false;                  \
      return;                              \
    }                                      \
  } while (0)

#define CeedCallOccaValid(isValidVar, ...) \
  do {                                     \
    int ierr_q_ = __VA_ARGS__;             \
    CeedOccaValidChk(isValidVar, ierr_q_); \
  } while (0);

#define CeedHandleOccaException(exc)                           \
  do {                                                         \
    std::string error = exc.toString();                        \
    return CeedError(ceed, CEED_ERROR_BACKEND, error.c_str()); \
  } while (0)

#define CeedOccaCastRegisterFunction(func) (ceed::occa::ceedFunction)(void*) func

#define CeedOccaRegisterBaseFunction(name, func) CeedCallBackend(registerCeedFunction(ceed, name, CeedOccaCastRegisterFunction(func)));

#define CeedOccaRegisterFunction(object, name, func) CeedCallBackend(registerCeedFunction(ceed, object, name, CeedOccaCastRegisterFunction(func)));

namespace ceed {
namespace occa {
typedef int (*ceedFunction)();
}
}  // namespace ceed

#endif
