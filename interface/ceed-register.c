// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <stdio.h>

static bool register_all_called;

#define MACRO(name, ...) CEED_INTERN int name(void);
#include "../backends/ceed-backend-list.h"
#undef MACRO

/**
  @brief Register all preconfigured backends.

  This is called automatically by CeedInit() and thus normally need not be called by users.
  Users can call CeedRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedRegister()

  @ref User
**/
int CeedRegisterAll() {
  if (register_all_called) return 0;
  register_all_called = true;

#define MACRO(name, ...) CeedChk(name());
#include "../backends/ceed-backend-list.h"
#undef MACRO
  return CEED_ERROR_SUCCESS;
}
