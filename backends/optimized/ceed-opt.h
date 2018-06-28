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

#include <ceed-impl.h>
#include <string.h>

typedef struct {
  Ceed ceedref;
} Ceed_Opt;

typedef struct {
  CeedScalar *colograd1d;
} CeedBasis_Opt;

CEED_INTERN int CeedVectorCreate_Opt(Ceed ceed, CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Opt(CeedElemRestriction r,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices);

CEED_INTERN int CeedBasisCreateTensorH1_Opt(Ceed ceed, CeedInt dim,
    CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedQFunctionCreate_Opt(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Opt(CeedOperator op);
