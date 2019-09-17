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

#include <string.h>
#include "ceed-backend.h"
#include "ceed-mass2dbuild.h"

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 2D
           mass matrix
**/
static int CeedQFunctionInit_Mass2DBuild(Ceed ceed, const char *requested,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "Mass2DBuild";
  if (strcmp(name, requested))
    return CeedError(ceed, 1, "QFunction '%s' does not match requested name: %s",
                     name, requested);

  // Add QFunction fields
  const CeedInt dim = 2;
  ierr = CeedQFunctionAddInput(qf, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);

  return 0;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 2D mass
           matrix
**/
__attribute__((constructor))
static void Register(void) {
  CeedQFunctionRegister("Mass2Dbuild", Mass2DBuild_loc, 1, Mass2DBuild,
                        CeedQFunctionInit_Mass2DBuild);
}
