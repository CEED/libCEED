// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-source/gallery/ceed-identity.h>
#include <stddef.h>
#include <string.h>

/**
  @brief Set fields identity QFunction that copies inputs directly into outputs
**/
static int CeedQFunctionInit_Identity(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Identity";
  if (strcmp(name, requested)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);
    // LCOV_EXCL_STOP
  }

  // QFunction fields 'input' and 'output' with requested emodes added
  //   by the library rather than being added here

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 0));

  // Context data
  CeedQFunctionContext ctx;
  IdentityCtx          ctx_data = {.size = 1};
  CeedCall(CeedQFunctionContextCreate(ceed, &ctx));
  CeedCall(CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctx_data), (void *)&ctx_data));
  CeedCall(CeedQFunctionContextRegisterInt32(ctx, "size", offsetof(IdentityCtx, size), 1, "field size of identity QFunction"));
  CeedCall(CeedQFunctionSetContext(qf, ctx));
  CeedCall(CeedQFunctionContextDestroy(&ctx));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register identity QFunction that copies inputs directly into outputs
**/
CEED_INTERN int CeedQFunctionRegister_Identity(void) {
  return CeedQFunctionRegister("Identity", Identity_loc, 1, Identity, CeedQFunctionInit_Identity);
}
