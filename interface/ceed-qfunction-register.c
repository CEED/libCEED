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

#define CEED_GALLERY_QFUNCTION(name) CEED_INTERN int name(void);
#include "../gallery/ceed-gallery-list.h"
#undef CEED_GALLERY_QFUNCTION

/**
  @brief Register the gallery of pre-configured @ref CeedQFunction.

  This is called automatically by @ref CeedQFunctionCreateInteriorByName() and thus normally need not be called by users.
  Users can call @ref CeedQFunctionRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedQFunctionRegister()

  @ref User
**/
int CeedQFunctionRegisterAll(void) {
  int ierr = 0;

  CeedPragmaCritical(CeedQFunctionRegisterAll) {
    if (!register_all_called) {
      CeedDebugEnv256(1, "\n---------- Registering Gallery QFunctions ----------\n");
#define CEED_GALLERY_QFUNCTION(name) \
  if (!ierr) ierr = name();
#include "../gallery/ceed-gallery-list.h"
#undef CEED_GALLERY_QFUNCTION
      register_all_called = true;
    }
  }
  return ierr;
}
