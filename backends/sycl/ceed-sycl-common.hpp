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

// Kris: Add sycl exception handling functions here.

CEED_INTERN int CeedSyclGetResourceRoot(Ceed ceed, const char *resource, char **resource_root);

CEED_INTERN int CeedSyclInit(Ceed ceed, const char *resource);

CEED_INTERN int CeedDestroy_Sycl(Ceed ceed);

#endif  // _ceed_sycl_common_h
