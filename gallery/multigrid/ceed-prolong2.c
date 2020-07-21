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
#include "ceed-multigrid2.h"

/**
  @brief Set fields for prolongation QFunction that scales inputs for multiplicity
**/
static int CeedQFunctionInit_Prolong2(Ceed ceed, const char *requested,
                                      CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "Prolong2";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  const CeedInt size = 2;
  ierr = CeedQFunctionAddInput(qf, "input", size, CEED_EVAL_INTERP);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "mult", size, CEED_EVAL_NONE); CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "output", size, CEED_EVAL_NONE);
  CeedChk(ierr);

  return 0;
}

/**
  @brief Register prolongation QFunction
**/
__attribute__((constructor))
static void Register(void) {
  CeedQFunctionRegister("Prolong2", Multigrid2_loc, 1, Multigrid2,
                        CeedQFunctionInit_Prolong2);
}
