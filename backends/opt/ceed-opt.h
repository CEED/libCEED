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
#include <string.h>

typedef struct {
  CeedInt blksize;
} Ceed_Opt;

typedef struct {
  CeedScalar *colograd1d;
} CeedBasis_Opt;

typedef struct {
  bool add;
  bool identityqf;
  CeedElemRestriction *blkrestr; /// Blocked versions of restrictions
  CeedVector
  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  uint64_t *inputstate;  /// State counter of inputs
  CeedVector *evecsin;   /// Input E-vectors needed to apply operator
  CeedVector *evecsout;  /// Output E-vectors needed to apply operator
  CeedVector *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector *qvecsout;  /// Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
} CeedOperator_Opt;

CEED_INTERN int CeedOperatorCreate_Opt(CeedOperator op);
