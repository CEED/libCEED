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
#include "ceed-vectormassapply.h"

/**
  @brief Set fields for Ceed QFunction for applying the mass matrix
           on a vector system with three components
**/
static int CeedQFunctionInit_VectorMassApply(Ceed ceed, const char *requested,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "VectorMassApply";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  const CeedInt num_comp = 3;
  ierr = CeedQFunctionAddInput(qf, "u", num_comp, CEED_EVAL_INTERP);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "v", num_comp, CEED_EVAL_INTERP);
  CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the mass matrix on a vector system
           with three components
**/
CEED_INTERN int CeedQFunctionRegister_VectorMassApply(void) {
  return CeedQFunctionRegister("VectorMassApply", VectorMassApply_loc, 1,
                               VectorMassApply, CeedQFunctionInit_VectorMassApply);
}
