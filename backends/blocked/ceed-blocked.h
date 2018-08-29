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
} Ceed_Blocked;

typedef struct {
  CeedScalar *colograd1d;
} CeedBasis_Blocked;

typedef struct {
  CeedElemRestriction *blkrestr; /// Blocked versions of restrictions
  CeedVector
  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedScalar **qdata; /// Inputs followed by outputs
  CeedScalar
  **qdata_alloc; /// Allocated quadrature data arrays (to be freed by us)
  CeedScalar **indata;
  CeedScalar **outdata;
  CeedInt    numein;
  CeedInt    numeout;
  CeedInt    numqin;
  CeedInt    numqout;
} CeedOperator_Blocked;

CEED_INTERN int CeedBasisCreateTensorH1_Blocked(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Blocked(CeedElemTopology topo, CeedInt dim,
                                      CeedInt ndof, CeedInt nqpts,
                                      const CeedScalar *interp,
                                      const CeedScalar *grad,
                                      const CeedScalar *qref,
                                      const CeedScalar *qweight,
                                      CeedBasis basis);

CEED_INTERN int CeedOperatorCreate_Blocked(CeedOperator op);
