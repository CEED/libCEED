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

#ifndef _ceed_hip_shared_h
#define _ceed_hip_shared_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include "../hip/ceed-hip-common.h"

typedef struct {
  hipModule_t module;
  hipFunction_t interp;
  hipFunction_t grad;
  hipFunction_t weight;
  CeedInt blksizes[3]; // interp, grad, weight thread block sizes
  CeedScalar *d_interp1d;
  CeedScalar *d_grad1d;
  CeedScalar *d_collograd1d;
  CeedScalar *d_qweight1d;
} CeedBasis_Hip_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
    const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

#endif // _ceed_hip_shared_h
