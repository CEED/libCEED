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
#include <math.h>

typedef struct {
  CeedScalar *collograd1d;
  bool collointerp;
} CeedBasis_Ref;

typedef struct {
  CeedScalar *array;
  CeedScalar *array_allocated;
} CeedVector_Ref;

typedef struct {
  const CeedInt *indices;
  CeedInt *indices_allocated;
} CeedElemRestriction_Ref;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
  bool setupdone;
} CeedQFunction_Ref;

typedef struct {
  bool add;
  bool identityqf;
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
} CeedOperator_Ref;

CEED_INTERN int CeedVectorCreate_Ref(CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Ref(CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r);

CEED_INTERN int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Ref(CeedElemTopology topo,
                                      CeedInt dim, CeedInt ndof, CeedInt nqpts,
                                      const CeedScalar *interp,
                                      const CeedScalar *grad,
                                      const CeedScalar *qref,
                                      const CeedScalar *qweight,
                                      CeedBasis basis);

CEED_INTERN int CeedTensorContractCreate_Ref(CeedBasis basis,
    CeedTensorContract contract);

CEED_INTERN int CeedQFunctionCreate_Ref(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Ref(CeedOperator op);

CEED_INTERN int CeedCompositeOperatorCreate_Ref(CeedOperator op);
