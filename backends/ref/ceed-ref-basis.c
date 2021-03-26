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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Basis Apply
//------------------------------------------------------------------------------
static int CeedBasisApply_Ref(CeedBasis basis, CeedInt nelem,
                              CeedTransposeMode tmode, CeedEvalMode emode,
                              CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedInt dim, ncomp, nnodes, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &nnodes); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChkBackend(ierr);
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChkBackend(ierr);
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  const CeedScalar *u;
  CeedScalar *v;
  if (U != CEED_VECTOR_NONE) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChkBackend(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChkBackend(ierr);

  // Clear v if operating in transpose
  if (tmode == CEED_TRANSPOSE) {
    const CeedInt vsize = nelem*ncomp*nnodes;
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = (CeedScalar) 0.0;
  }
  bool tensorbasis;
  ierr = CeedBasisIsTensor(basis, &tensorbasis); CeedChkBackend(ierr);
  // Tensor basis
  if (tensorbasis) {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    switch (emode) {
    // Interpolate to/from quadrature points
    case CEED_EVAL_INTERP: {
      CeedBasis_Ref *impl;
      ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);
      if (impl->collointerp) {
        memcpy(v, u, nelem*ncomp*nnodes*sizeof(u[0]));
      } else {
        CeedInt P = P1d, Q = Q1d;
        if (tmode == CEED_TRANSPOSE) {
          P = Q1d; Q = P1d;
        }
        CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
        CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
        const CeedScalar *interp1d;
        ierr = CeedBasisGetInterp1D(basis, &interp1d); CeedChkBackend(ierr);
        for (CeedInt d=0; d<dim; d++) {
          ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                         interp1d, tmode, add&&(d==dim-1),
                                         d==0?u:tmp[d%2],
                                         d==dim-1?v:tmp[(d+1)%2]);
          CeedChkBackend(ierr);
          pre /= P;
          post *= Q;
        }
      }
    } break;
    // Evaluate the gradient to/from quadrature points
    case CEED_EVAL_GRAD: {
      // In CEED_NOTRANSPOSE mode:
      // u has shape [dim, ncomp, P^dim, nelem], row-major layout
      // v has shape [dim, ncomp, Q^dim, nelem], row-major layout
      // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
      CeedInt P = P1d, Q = Q1d;
      if (tmode == CEED_TRANSPOSE) {
        P = Q1d, Q = Q1d;
      }
      CeedBasis_Ref *impl;
      ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
      const CeedScalar *interp1d;
      ierr = CeedBasisGetInterp1D(basis, &interp1d); CeedChkBackend(ierr);
      if (impl->collograd1d) {
        CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
        CeedScalar interp[nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
        // Interpolate to quadrature points (NoTranspose)
        //  or Grad to quadrature points (Transpose)
        for (CeedInt d=0; d<dim; d++) {
          ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                         (tmode == CEED_NOTRANSPOSE
                                          ? interp1d
                                          : impl->collograd1d),
                                         tmode, add&&(d>0),
                                         (tmode == CEED_NOTRANSPOSE
                                          ? (d==0?u:tmp[d%2])
                                          : u + d*nqpt*ncomp*nelem),
                                         (tmode == CEED_NOTRANSPOSE
                                          ? (d==dim-1?interp:tmp[(d+1)%2])
                                          : interp));
          CeedChkBackend(ierr);
          pre /= P;
          post *= Q;
        }
        // Grad to quadrature points (NoTranspose)
        //  or Interpolate to nodes (Transpose)
        P = Q1d, Q = Q1d;
        if (tmode == CEED_TRANSPOSE) {
          P = Q1d, Q = P1d;
        }
        pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
        for (CeedInt d=0; d<dim; d++) {
          ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                         (tmode == CEED_NOTRANSPOSE
                                          ? impl->collograd1d
                                          : interp1d),
                                         tmode, add&&(d==dim-1),
                                         (tmode == CEED_NOTRANSPOSE
                                          ? interp
                                          : (d==0?interp:tmp[d%2])),
                                         (tmode == CEED_NOTRANSPOSE
                                          ? v + d*nqpt*ncomp*nelem
                                          : (d==dim-1?v:tmp[(d+1)%2])));
          CeedChkBackend(ierr);
          pre /= P;
          post *= Q;
        }
      } else if (impl->collointerp) { // Qpts collocated with nodes
        const CeedScalar *grad1d;
        ierr = CeedBasisGetGrad1D(basis, &grad1d); CeedChkBackend(ierr);

        // Dim contractions, identity in other directions
        CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
        for (CeedInt d=0; d<dim; d++) {
          ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                         grad1d, tmode, add&&(d>0),
                                         tmode == CEED_NOTRANSPOSE
                                         ? u : u+d*ncomp*nqpt*nelem,
                                         tmode == CEED_TRANSPOSE
                                         ? v : v+d*ncomp*nqpt*nelem);
          CeedChkBackend(ierr);
          pre /= P;
          post *= Q;
        }
      } else { // Underintegration, P > Q
        const CeedScalar *grad1d;
        ierr = CeedBasisGetGrad1D(basis, &grad1d); CeedChkBackend(ierr);

        if (tmode == CEED_TRANSPOSE) {
          P = Q1d, Q = P1d;
        }
        CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];

        // Dim**2 contractions, apply grad when pass == dim
        for (CeedInt p=0; p<dim; p++) {
          CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
          for (CeedInt d=0; d<dim; d++) {
            ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                           (p==d)? grad1d : interp1d,
                                           tmode, add&&(d==dim-1),
                                           (d == 0
                                            ? (tmode == CEED_NOTRANSPOSE
                                               ? u : u+p*ncomp*nqpt*nelem)
                                            : tmp[d%2]),
                                           (d == dim-1
                                            ? (tmode == CEED_TRANSPOSE
                                               ? v : v+p*ncomp*nqpt*nelem)
                                            : tmp[(d+1)%2]));
            CeedChkBackend(ierr);
            pre /= P;
            post *= Q;
          }
        }
      }
    } break;
    // Retrieve interpolation weights
    case CEED_EVAL_WEIGHT: {
      if (tmode == CEED_TRANSPOSE)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      // LCOV_EXCL_STOP
      CeedInt Q = Q1d;
      const CeedScalar *qweight1d;
      ierr = CeedBasisGetQWeights(basis, &qweight1d); CeedChkBackend(ierr);
      for (CeedInt d=0; d<dim; d++) {
        CeedInt pre = CeedIntPow(Q, dim-d-1), post = CeedIntPow(Q, d);
        for (CeedInt i=0; i<pre; i++)
          for (CeedInt j=0; j<Q; j++)
            for (CeedInt k=0; k<post; k++) {
              CeedScalar w = qweight1d[j]
                             * (d == 0 ? 1 : v[((i*Q + j)*post + k)*nelem]);
              for (CeedInt e=0; e<nelem; e++)
                v[((i*Q + j)*post + k)*nelem + e] = w;
            }
      }
    } break;
    // LCOV_EXCL_START
    // Evaluate the divergence to/from the quadrature points
    case CEED_EVAL_DIV:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
    // Take no action, BasisApply should not have been called
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
    }
  } else {
    // Non-tensor basis
    switch (emode) {
    // Interpolate to/from quadrature points
    case CEED_EVAL_INTERP: {
      CeedInt P = nnodes, Q = nqpt;
      const CeedScalar *interp;
      ierr = CeedBasisGetInterp(basis, &interp); CeedChkBackend(ierr);
      if (tmode == CEED_TRANSPOSE) {
        P = nqpt; Q = nnodes;
      }
      ierr = CeedTensorContractApply(contract, ncomp, P, nelem, Q,
                                     interp, tmode, add, u, v);
      CeedChkBackend(ierr);
    }
    break;
    // Evaluate the gradient to/from quadrature points
    case CEED_EVAL_GRAD: {
      CeedInt P = nnodes, Q = nqpt;
      CeedInt dimstride = nqpt * ncomp * nelem;
      CeedInt gradstride = nqpt * nnodes;
      const CeedScalar *grad;
      ierr = CeedBasisGetGrad(basis, &grad); CeedChkBackend(ierr);
      if (tmode == CEED_TRANSPOSE) {
        P = nqpt; Q = nnodes;
        for (CeedInt d = 0; d < dim; d++) {
          ierr = CeedTensorContractApply(contract, ncomp, P, nelem, Q,
                                         grad + d * gradstride, tmode, add,
                                         u + d * dimstride, v); CeedChkBackend(ierr);
        }
      } else {
        for (CeedInt d = 0; d < dim; d++) {
          ierr = CeedTensorContractApply(contract, ncomp, P, nelem, Q,
                                         grad + d * gradstride, tmode, add,
                                         u, v + d * dimstride); CeedChkBackend(ierr);
        }
      }
    }
    break;
    // Retrieve interpolation weights
    case CEED_EVAL_WEIGHT: {
      if (tmode == CEED_TRANSPOSE)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      // LCOV_EXCL_STOP
      const CeedScalar *qweight;
      ierr = CeedBasisGetQWeights(basis, &qweight); CeedChkBackend(ierr);
      for (CeedInt i=0; i<nqpt; i++)
        for (CeedInt e=0; e<nelem; e++)
          v[i*nelem + e] = qweight[i];
    } break;
    // LCOV_EXCL_START
    // Evaluate the divergence to/from the quadrature points
    case CEED_EVAL_DIV:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
    // Take no action, BasisApply should not have been called
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
    }
  }
  if (U != CEED_VECTOR_NONE) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Destroy Non-Tensor
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Ref(CeedBasis basis) {
  int ierr;
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChkBackend(ierr);
  ierr = CeedTensorContractDestroy(&contract); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Non-Tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Ref(CeedElemTopology topo, CeedInt dim,
                          CeedInt nnodes, CeedInt nqpts,
                          const CeedScalar *interp,
                          const CeedScalar *grad,
                          const CeedScalar *qref,
                          const CeedScalar *qweight,
                          CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  Ceed parent;
  ierr = CeedGetParent(ceed, &parent); CeedChkBackend(ierr);
  CeedTensorContract contract;
  ierr = CeedTensorContractCreate(parent, basis, &contract); CeedChkBackend(ierr);
  ierr = CeedBasisSetTensorContract(basis, &contract); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Ref); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Destroy Tensor
//------------------------------------------------------------------------------
static int CeedBasisDestroyTensor_Ref(CeedBasis basis) {
  int ierr;
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChkBackend(ierr);
  ierr = CeedTensorContractDestroy(&contract); CeedChkBackend(ierr);

  CeedBasis_Ref *impl;
  ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->collograd1d); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Ref *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  // Check for collocated interp
  if (Q1d == P1d) {
    bool collocated = 1;
    for (CeedInt i=0; i<P1d; i++) {
      collocated = collocated && (fabs(interp1d[i+P1d*i] - 1.0) < 1e-14);
      for (CeedInt j=0; j<P1d; j++)
        if (j != i)
          collocated = collocated && (fabs(interp1d[j+P1d*i]) < 1e-14);
    }
    impl->collointerp = collocated;
  }
  // Calculate collocated grad
  if (Q1d >= P1d && !impl->collointerp) {
    ierr = CeedMalloc(Q1d*Q1d, &impl->collograd1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, impl->collograd1d);
    CeedChkBackend(ierr);
  }
  ierr = CeedBasisSetData(basis, impl); CeedChkBackend(ierr);

  Ceed parent;
  ierr = CeedGetParent(ceed, &parent); CeedChkBackend(ierr);
  CeedTensorContract contract;
  ierr = CeedTensorContractCreate(parent, basis, &contract); CeedChkBackend(ierr);
  ierr = CeedBasisSetTensorContract(basis, &contract); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyTensor_Ref); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
