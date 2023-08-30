// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  @brief Register the gallery of preconfigured QFunctions.

  This is called automatically by CeedQFunctionCreateInteriorByName() and thus normally need not be called by users.
  Users can call CeedQFunctionRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedQFunctionRegister()

  @ref User
**/
int CeedQFunctionRegisterAll() {
  CeedPragmaCritical(CeedQFunctionRegisterAll) {
    if (!register_all_called) {
      CeedDebugEnv256(1, "\n---------- Registering Gallery QFunctions ----------\n");
#define CEED_GALLERY_QFUNCTION(name) CeedChk(name());
#include "../gallery/ceed-gallery-list.h"
#undef CEED_GALLERY_QFUNCTION
      register_all_called = true;
    }
  }
  return CEED_ERROR_SUCCESS;
}
