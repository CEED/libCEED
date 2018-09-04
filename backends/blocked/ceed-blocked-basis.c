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
#include "ceed-blocked.h"

// Contracts on the middle index
// NOTRANSPOSE: V_ajc = T_jb U_abc
// TRANSPOSE:   V_ajc = T_bj U_abc
// If Add != 0, "=" is replaced by "+="
static int CeedTensorContract_Blocked(Ceed ceed,
                                  CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                  const CeedScalar *restrict t,
                                  CeedTransposeMode tmode, const CeedInt Add,
                                  const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  if (!Add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  for (CeedInt a=0; a<A; a++)
    for (CeedInt b=0; b<B; b++)
      for (CeedInt j=0; j<J; j++) {
        CeedScalar tq = t[j*tstride0 + b*tstride1];
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }
  return 0;
}

static int CeedBasisApply_Blocked(CeedBasis basis, CeedInt nelem,
                              CeedTransposeMode tmode, CeedEvalMode emode,
                              const CeedScalar *u, CeedScalar *v) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt nqpt = CeedIntPow(basis->Q1d, dim);
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  const CeedInt blksize = 8;

  if ((nelem != 1) && (nelem != blksize))
    return CeedError(basis->ceed, 1,
                     "This backend does not support BasisApply for %d elements", nelem);

  if (tmode == CEED_TRANSPOSE) {
    const CeedInt vsize = nelem*ncomp*CeedIntPow(basis->P1d, dim);
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = (CeedScalar) 0.0;
  }
  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P = basis->P1d, Q = basis->Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d; Q = basis->P1d;
    }
    CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
    CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Blocked(basis->ceed, pre, P, post, Q,
                                    basis->interp1d, tmode, add&&(d==dim-1),
                                    d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
  } break;
  case CEED_EVAL_GRAD: {
    // In CEED_NOTRANSPOSE mode:
    // u has shape [dim, ncomp, P^dim, nelem], row-major layout
    // v has shape [dim, ncomp, Q^dim, nelem], row-major layout
    // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
    CeedInt P = basis->P1d, Q = basis->Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d, Q = basis->Q1d;
    }
    CeedBasis_Blocked *impl = basis->data;
    CeedScalar interp[nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
    CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    // Interpolate to quadrature points (NoTranspose)
    //  or Grad to quadrature points (Transpose)
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Blocked(basis->ceed, pre, P, post, Q,
                                    (tmode == CEED_NOTRANSPOSE
                                     ? basis->interp1d
                                     : impl->colograd1d),
                                    tmode, add&&(d>0),
                                    (tmode == CEED_NOTRANSPOSE
                                     ? (d==0?u:tmp[d%2])
                                     : u + d*nqpt*ncomp*nelem),
                                    (tmode == CEED_NOTRANSPOSE
                                     ? (d==dim-1?interp:tmp[(d+1)%2])
                                     : interp));
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
    // Grad to quadrature points (NoTranspose)
    //  or Interpolate to dofs (Transpose)
    P = basis->Q1d, Q = basis->Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = basis->Q1d, Q = basis->P1d;
    }
    pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Blocked(basis->ceed, pre, P, post, Q,
                                    (tmode == CEED_NOTRANSPOSE
                                     ? impl->colograd1d
                                     : basis->interp1d),
                                    tmode, add&&(d==dim-1),
                                    (tmode == CEED_NOTRANSPOSE
                                     ? interp
                                     : (d==0?interp:tmp[d%2])),
                                    (tmode == CEED_NOTRANSPOSE
                                     ? v + d*nqpt*ncomp*nelem
                                     : (d==dim-1?v:tmp[(d+1)%2])));
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt Q = basis->Q1d;
    for (CeedInt d=0; d<dim; d++) {
      CeedInt pre = CeedIntPow(Q, dim-d-1), post = CeedIntPow(Q, d);
      for (CeedInt i=0; i<pre; i++)
        for (CeedInt j=0; j<Q; j++)
          for (CeedInt k=0; k<post; k++) {
            CeedScalar w = basis->qweight1d[j]
                           * (d == 0 ? 1 : v[((i*Q + j)*post + k)*nelem]);
            for (CeedInt e=0; e<nelem; e++)
              v[((i*Q + j)*post + k)*nelem + e] = w;
          }
    }
  } break;
  case CEED_EVAL_DIV:
    return CeedError(basis->ceed, 1, "CEED_EVAL_DIV not supported");
  case CEED_EVAL_CURL:
    return CeedError(basis->ceed, 1, "CEED_EVAL_CURL not supported");
  case CEED_EVAL_NONE:
    return CeedError(basis->ceed, 1,
                     "CEED_EVAL_NONE does not make sense in this context");
  }
  return 0;
}

static int CeedBasisDestroy_Blocked(CeedBasis basis) {
  CeedBasis_Blocked *impl = basis->data;
  int ierr;

  ierr = CeedFree(&impl->colograd1d); CeedChk(ierr);
  ierr = CeedFree(&basis->data); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Blocked(Ceed ceed, CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  CeedBasis_Blocked *impl;
  int ierr;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*Q1d, &impl->colograd1d); CeedChk(ierr);
  ierr = CeedBasisGetCollocatedGrad(basis, impl->colograd1d); CeedChk(ierr);
  basis->data = impl;

  basis->Apply = CeedBasisApply_Blocked;
  basis->Destroy = CeedBasisDestroy_Blocked;
  return 0;
}



int CeedBasisCreateH1_Blocked(Ceed ceed, CeedElemTopology topo, CeedInt dim,
                          CeedInt ndof, CeedInt nqpts,
                          const CeedScalar *interp,
                          const CeedScalar *grad,
                          const CeedScalar *qref,
                          const CeedScalar *qweight,
                          CeedBasis basis) {
  return CeedError(basis->ceed, 1, "Backend does not implement non-tensor bases");
}
