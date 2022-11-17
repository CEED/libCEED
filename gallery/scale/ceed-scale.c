// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-scale.h>
#include <string.h>

/**
  @brief  Set fields for vector scaling QFunction that scales inputs
**/
static int CeedQFunctionInit_Scale(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Scale";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // QFunction fields 'input' and 'output' with requested emodes added
  //   by the library rather than being added here

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register scaling QFunction
**/
CEED_INTERN int CeedQFunctionRegister_Scale(void) { return CeedQFunctionRegister("Scale", Scale_loc, 1, Scale, CeedQFunctionInit_Scale); }
