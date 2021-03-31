// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-scale.h"

/**
  @brief  Set fields for vector scaling QFunction that scales inputs
**/
static int CeedQFunctionInit_Scale(Ceed ceed, const char *requested,
                                   CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Scale";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // QFunction fields 'input' and 'output' with requested emodes added
  //   by the library rather than being added here

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register scaling QFunction
**/
CEED_INTERN int CeedQFunctionRegister_Scale(void) {
  return CeedQFunctionRegister("Scale", Scale_loc, 1, Scale,
                               CeedQFunctionInit_Scale);
}
