// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <string.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Memcheck(const char *resource, Ceed ceed) {
  Ceed ceed_ref;

  CeedCheck(!strcmp(resource, "/cpu/self/memcheck") || !strcmp(resource, "/cpu/self/memcheck/serial"), ceed, CEED_ERROR_BACKEND,
            "Valgrind Memcheck backend cannot use resource: %s", resource);

  // Create reference Ceed that implementation will be dispatched through unless overridden
  CeedCallBackend(CeedInit("/cpu/self/ref/serial", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateBlocked", CeedElemRestrictionCreate_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateAtPoints", CeedElemRestrictionCreate_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate", CeedQFunctionContextCreate_Memcheck));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Memcheck_Serial(void) { return CeedRegister("/cpu/self/memcheck/serial", CeedInit_Memcheck, 100); }

//------------------------------------------------------------------------------
