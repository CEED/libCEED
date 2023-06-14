// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_h
#define _ceed_sycl_h

#include <ceed/ceed.h>

CEED_EXTERN int CeedSetStream_Sycl(Ceed ceed, void *handle);

int CeedSetStream_Sycl(Ceed ceed, void *handle) __attribute__((weak));
int CeedSetStream_Sycl(Ceed ceed, void *handle) {
  // TODO Add Error to say that sycl needs to be installed for this to work.
  return -1;
}

#endif
