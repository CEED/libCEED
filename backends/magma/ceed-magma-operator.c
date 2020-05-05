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

#include <ceed-backend.h>
#include "../cuda/ceed-cuda.h"
#include "ceed-magma.h"
#include <string.h>

// Create a cuda-ref operator and modify its eandqdiffer value
int CeedOperatorCreate_Magma(CeedOperator op) {
  int ierr;

  // Explicitly set up a cuda-ref Operator
  ierr = CeedOperatorCreate_Cuda(op); CeedChk(ierr);

  // Get the backend data for this op
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

  // Set this value to false since the E- and Q-Vector layouts
  // use the same ordering strategy for this backend (but not cuda-ref)
  impl->eandqdiffer = false;

  return 0;
}
