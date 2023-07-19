// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_common_hpp
#define _ceed_sycl_common_hpp

#include <ceed/backend.h>
#include <ceed/jit-source/sycl/sycl-types.h>

#include <sycl/sycl.hpp>
#include <type_traits>

#define CeedCallSycl(ceed, ...)                               \
  do {                                                        \
    try {                                                     \
      __VA_ARGS__;                                            \
    } catch (sycl::exception const &e) {                      \
      return CeedError((ceed), CEED_ERROR_BACKEND, e.what()); \
    }                                                         \
  } while (0);

using CeedBackendFunction = int (*)();

template <typename R, class... Args>
int CeedSetBackendFunctionCpp(Ceed ceed, const char *type, void *object, const char *fname, R (*f)(Args...)) {
  static_assert(std::is_same_v<int, R>, "Ceed backend functions must return int");
  // Kris: this is potentially undefined behavior by C++ standards
  auto *bf = reinterpret_cast<CeedBackendFunction>(f);
  return CeedSetBackendFunction(ceed, type, object, fname, bf);
}

typedef struct {
  sycl::context sycl_context;
  sycl::device  sycl_device;
  sycl::queue   sycl_queue;
} Ceed_Sycl;

CEED_INTERN int CeedGetResourceRoot_Sycl(Ceed ceed, const char *resource, char **resource_root);

CEED_INTERN int CeedInit_Sycl(Ceed ceed, const char *resource);

CEED_INTERN int CeedDestroy_Sycl(Ceed ceed);

CEED_INTERN int CeedSetStream_Sycl(Ceed ceed, void *handle);

#endif  // _ceed_sycl_common_h
