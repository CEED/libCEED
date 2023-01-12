// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_common_hpp
#define _ceed_sycl_common_hpp

#include <ceed/backend.h>
#include <sycl/sycl.hpp>

#include <type_traits>

// Kris: Add sycl exception handling functions here.

using CeedBackendFunction = int (*)();

template<typename R,class... Args>
int CeedSetBackendFunctionCpp(Ceed ceed, const char *type, const char *fname, R (*f)(Args...)) {
  static_assert(std::is_same_v<int,R>,"Ceed backend functions must return int");
  // Kris: this is potentially undefined behavior by C++ standards
  auto* bf = reinterpret_cast<CeedBackendFunction>(f); 
  return CeedSetBackendFunction(ceed, type, ceed, fname, bf);
}

typedef struct {
  sycl::context sycl_context;
  sycl::device sycl_device;
  sycl::queue sycl_queue;
} Ceed_Sycl;

CEED_INTERN int CeedSyclGetResourceRoot(Ceed ceed, const char *resource, char **resource_root);

CEED_INTERN int CeedSyclInit(Ceed ceed, const char *resource);

CEED_INTERN int CeedDestroy_Sycl(Ceed ceed);

#endif  // _ceed_sycl_common_h
