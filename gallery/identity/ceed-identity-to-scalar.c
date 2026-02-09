// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/gallery/ceed-identity-to-scalar.h>
#include <stddef.h>
#include <string.h>

/**
  @brief Set fields identity `CeedQFunction` that copies first input component directly into output
**/
static int CeedQFunctionInit_IdentityScalar(Ceed ceed, const char *requested, CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Identity to scalar";
  CeedCheck(!strcmp(name, requested), ceed, CEED_ERROR_UNSUPPORTED, "QFunction '%s' does not match requested name: %s", name, requested);

  // QFunction fields 'input' and 'output' with requested emodes added by the library rather than being added here

  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf, 0));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register identity `CeedQFunction` that copies first input component directly into output
**/
CEED_INTERN int CeedQFunctionRegister_IdentityScalar(void) {
  return CeedQFunctionRegister("Identity to scalar", IdentityScalar_loc, 1, IdentityScalar, CeedQFunctionInit_IdentityScalar);
}
