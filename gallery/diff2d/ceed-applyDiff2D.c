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
#include "ceed-applyDiff2D.h"

/**
  @brief Set fields for Ceed QFunction applying the geometric data for the 2D
           diff operator
**/
static int CeedQFunctionInit_ApplyDiff2D(Ceed ceed, const char *name,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  if (strcmp(name, "applyDiff2D"))
    return CeedError(ceed, 1, "QFunction does not match name: %s", name);

  // Add QFunction fields
  const CeedInt dim = 2;
  ierr = CeedQFunctionAddInput(qf, "du", dim, CEED_EVAL_GRAD); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "dv", dim, CEED_EVAL_GRAD); CeedChk(ierr);

  return 0;
}

/**
  @brief Register Ceed QFunction for applying the geometric data for the 2D diff
           operator
**/
__attribute__((constructor))
static void Register(void) {
  CeedQFunctionRegister("applyDiff2D", applyDiff2D_loc, 1, applyDiff2D,
                        CeedQFunctionInit_ApplyDiff2D);
}
