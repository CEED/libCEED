// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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
#include "ceed-occa.h"

// *****************************************************************************
// * BASIS: Apply, Destroy, CreateTensorH1
// *****************************************************************************
static int CeedBasisApplyOcca(CeedBasis basis, CeedTransposeMode tmode,
                              CeedEvalMode emode,
                              const CeedScalar *u, CeedScalar *v) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ndof = basis->ndof;

  dbg("\033[38;5;249m[CeedBasis][Apply]");
  switch (emode) {
  case CEED_EVAL_NONE: break;
  case CEED_EVAL_INTERP: {
    CeedInt P = basis->P1d, Q = basis->Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d; Q = basis->P1d;
    }
    CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
    CeedScalar tmp[2][Q*CeedPowInt(P>Q?P:Q, dim-1)];
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContractOcca(basis->ceed, pre, P, post, Q, basis->interp1d,
                                    tmode,
                                    d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]); CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
  }
  break;
  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt Q = basis->Q1d;
    for (CeedInt d=0; d<dim; d++) {
      CeedInt pre = CeedPowInt(Q, dim-d-1), post = CeedPowInt(Q, d);
      for (CeedInt i=0; i<pre; i++) {
        for (CeedInt j=0; j<Q; j++) {
          for (CeedInt k=0; k<post; k++) {
            v[(i*Q + j)*post + k] = basis->qweight1d[j]
                                    * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
          }
        }
      }
    }
  } break;
  default:
    return CeedError(basis->ceed, 1, "EvalMode %d not supported", emode);
  }
  return 0;
}

// *****************************************************************************
static int CeedBasisDestroyOcca(CeedBasis basis) {
  dbg("\033[38;5;249m[CeedBasis][Destroy]");
  return 0;
}

// *****************************************************************************
int CeedBasisCreateTensorH1Occa(Ceed ceed, CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  basis->Apply = CeedBasisApplyOcca;
  basis->Destroy = CeedBasisDestroyOcca;
  dbg("\033[38;5;249m[CeedBasis][Create][TensorH1]");
  return 0;
}
