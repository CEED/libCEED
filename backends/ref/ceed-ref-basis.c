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

#include "ceed-ref.h"

static int CeedBasisApply_Ref(CeedBasis basis, CeedInt nelem,
                              CeedTransposeMode tmode, CeedEvalMode emode,
                              CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedInt dim, ncomp, nnodes, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &nnodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChk(ierr);
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChk(ierr);

  // Clear v if operating in transpose
  if (tmode == CEED_TRANSPOSE) {
    const CeedInt vsize = nelem*ncomp*nnodes;
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = (CeedScalar) 0.0;
  }
  bool tensorbasis;
  ierr = CeedBasisGetTensorStatus(basis, &tensorbasis); CeedChk(ierr);
  // Tensor basis
  if (tensorbasis) {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    switch (emode) {
    // Interpolate to/from quadrature points
    case CEED_EVAL_INTERP: {
      CeedBasis_Ref *impl;
      ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);
      if (impl->collointerp) {
        memcpy(v, u, nelem*ncomp*nnodes*sizeof(u[0]));
      } else {
        CeedInt P = P1d, Q = Q1d;
        if (tmode == CEED_TRANSPOSE) {
          P = Q1d; Q = P1d;
        }
        CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
        CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
        CeedScalar *interp1d;
        ierr = CeedBasisGetInterp(basis, &interp1d); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++) {
          ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                         interp1d, tmode, add&&(d==dim-1),
                                         d==0?u:tmp[d%2],
                                         d==dim-1?v:tmp[(d+1)%2]);
          CeedChk(ierr);
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
      ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
      CeedScalar *interp1d;
      ierr = CeedBasisGetInterp(basis, &interp1d); CeedChk(ierr);
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
          CeedChk(ierr);
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
          CeedChk(ierr);
          pre /= P;
          post *= Q;
        }
      } else if (impl->collointerp) { // Qpts collocated with nodes
        CeedScalar *grad1d;
        ierr = CeedBasisGetGrad(basis, &grad1d); CeedChk(ierr);

        // Dim contractions, identity in other directions
        for (CeedInt d=0; d<dim; d++) {
          CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
          ierr = CeedTensorContractApply(contract, pre, P, post, Q,
                                         grad1d, tmode, add&&(d>0),
                                         tmode == CEED_NOTRANSPOSE
                                         ? u : u+d*ncomp*nqpt*nelem,
                                         tmode == CEED_TRANSPOSE
                                         ? v : v+d*ncomp*nqpt*nelem);
          CeedChk(ierr);
        }
      } else { // Underintegration, P > Q
        CeedScalar *grad1d;
        ierr = CeedBasisGetGrad(basis, &grad1d); CeedChk(ierr);

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
            CeedChk(ierr);
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
        return CeedError(ceed, 1,
                         "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      // LCOV_EXCL_STOP
      CeedInt Q = Q1d;
      CeedScalar *qweight1d;
      ierr = CeedBasisGetQWeights(basis, &qweight1d); CeedChk(ierr);
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
      return CeedError(ceed, 1, "CEED_EVAL_DIV not supported");
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL:
      return CeedError(ceed, 1, "CEED_EVAL_CURL not supported");
    // Take no action, BasisApply should not have been called
    case CEED_EVAL_NONE:
      return CeedError(ceed, 1,
                       "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
    }
  } else {
    // Non-tensor basis
    switch (emode) {
    // Interpolate to/from quadrature points
    case CEED_EVAL_INTERP: {
      CeedInt P = nnodes, Q = nqpt;
      CeedScalar *interp;
      ierr = CeedBasisGetInterp(basis, &interp); CeedChk(ierr);
      if (tmode == CEED_TRANSPOSE) {
        P = nqpt; Q = nnodes;
      }
      ierr = CeedTensorContractApply(contract, ncomp, P, nelem, Q,
                                     interp, tmode, add, u, v);
      CeedChk(ierr);
    }
    break;
    // Evaluate the gradient to/from quadrature points
    case CEED_EVAL_GRAD: {
      CeedInt P = nnodes, Q = dim*nqpt;
      CeedScalar *grad;
      ierr = CeedBasisGetGrad(basis, &grad); CeedChk(ierr);
      if (tmode == CEED_TRANSPOSE) {
        P = dim*nqpt; Q = nnodes;
      }
      ierr = CeedTensorContractApply(contract, ncomp, P, nelem, Q,
                                     grad, tmode, add, u, v);
      CeedChk(ierr);
    }
    break;
    // Retrieve interpolation weights
    case CEED_EVAL_WEIGHT: {
      if (tmode == CEED_TRANSPOSE)
        // LCOV_EXCL_START
        return CeedError(ceed, 1,
                         "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      // LCOV_EXCL_STOP
      CeedScalar *qweight;
      ierr = CeedBasisGetQWeights(basis, &qweight); CeedChk(ierr);
      for (CeedInt i=0; i<nqpt; i++)
        for (CeedInt e=0; e<nelem; e++)
          v[i*nelem + e] = qweight[i];
    } break;
    // LCOV_EXCL_START
    // Evaluate the divergence to/from the quadrature points
    case CEED_EVAL_DIV:
      return CeedError(ceed, 1, "CEED_EVAL_DIV not supported");
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL:
      return CeedError(ceed, 1, "CEED_EVAL_CURL not supported");
    // Take no action, BasisApply should not have been called
    case CEED_EVAL_NONE:
      return CeedError(ceed, 1,
                       "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
    }
  }
  if (U) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChk(ierr);
  return 0;
}

static int CeedBasisDestroyNonTensor_Ref(CeedBasis basis) {
  int ierr;
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChk(ierr);
  ierr = CeedTensorContractDestroy(&contract); CeedChk(ierr);
  return 0;
}

static int CeedBasisDestroyTensor_Ref(CeedBasis basis) {
  int ierr;
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChk(ierr);
  ierr = CeedTensorContractDestroy(&contract); CeedChk(ierr);

  CeedBasis_Ref *impl;
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);
  ierr = CeedFree(&impl->collograd1d); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedBasis_Ref *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
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
    ierr = CeedMalloc(Q1d*Q1d, &impl->collograd1d); CeedChk(ierr);
    ierr = CeedBasisGetCollocatedGrad(basis, impl->collograd1d); CeedChk(ierr);
  }
  ierr = CeedBasisSetData(basis, (void *)&impl); CeedChk(ierr);

  Ceed parent;
  ierr = CeedGetParent(ceed, &parent); CeedChk(ierr);
  CeedTensorContract contract;
  ierr = CeedTensorContractCreate(parent, basis, &contract); CeedChk(ierr);
  ierr = CeedBasisSetTensorContract(basis, &contract); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyTensor_Ref); CeedChk(ierr);
  return 0;
}



int CeedBasisCreateH1_Ref(CeedElemTopology topo, CeedInt dim,
                          CeedInt nnodes, CeedInt nqpts,
                          const CeedScalar *interp,
                          const CeedScalar *grad,
                          const CeedScalar *qref,
                          const CeedScalar *qweight,
                          CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  Ceed parent;
  ierr = CeedGetParent(ceed, &parent); CeedChk(ierr);
  CeedTensorContract contract;
  ierr = CeedTensorContractCreate(parent, basis, &contract); CeedChk(ierr);
  ierr = CeedBasisSetTensorContract(basis, &contract); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Ref); CeedChk(ierr);

  return 0;
}
