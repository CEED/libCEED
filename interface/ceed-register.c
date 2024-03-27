// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>

static bool register_all_called;

#define CEED_BACKEND(name, ...) CEED_INTERN int name(void);
#include "../backends/ceed-backend-list.h"
#undef CEED_BACKEND

/**
  @brief Register all pre-configured backends.

  This is called automatically by @ref CeedInit() and thus normally need not be called by users.
  Users can call @ref CeedRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedRegister()

  @ref User
**/
int CeedRegisterAll(void) {
  int ierr = 0;

  CeedPragmaCritical(CeedRegisterAll) {
    if (!register_all_called) {
      CeedDebugEnv256(1, "\n---------- Registering Backends ----------\n");
#define CEED_BACKEND(name, ...) \
  if (!ierr) ierr = name();
#include "../backends/ceed-backend-list.h"
#undef CEED_BACKEND
      register_all_called = true;
    }
  }
  return ierr;
}
