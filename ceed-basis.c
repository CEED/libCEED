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

#include <ceed-impl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int CeedBasisScalarGenericCreate(CeedInt ndof, CeedInt nqpt, CeedInt dim,
                                 CeedBasisScalarGeneric *basis) {
  // TODO: cleanup in case of error
  int ierr;
  ierr = CeedCalloc(1, basis); CeedChk(ierr);
  (*basis)->ndof = ndof;
  (*basis)->nqpt = nqpt;
  (*basis)->dim = dim;
  ierr = CeedMalloc(nqpt*ndof, &(*basis)->interp); CeedChk(ierr);
  ierr = CeedMalloc(nqpt*dim*ndof, &(*basis)->grad); CeedChk(ierr);
  ierr = CeedMalloc(nqpt, &(*basis)->qweights); CeedChk(ierr);
  return 0;
}

int CeedBasisScalarGenericDestroy(CeedBasisScalarGeneric *basis) {
  int ierr;
  if (!(*basis)) return 0;
  ierr = CeedFree(&(*basis)->qweights); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->grad); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->interp); CeedChk(ierr);
  ierr = CeedFree(basis); CeedChk(ierr);
  return 0;
}

int CeedBasisScalarTensorCreate(CeedInt ndof1d, CeedInt nqpt1d, CeedInt dim,
                                CeedBasisScalarTensor *basis) {
  int ierr;
  ierr = CeedCalloc(1, basis); CeedChk(ierr);
  (*basis)->P1d = ndof1d;
  (*basis)->Q1d = nqpt1d;
  (*basis)->dim = dim;
  ierr = CeedMalloc(nqpt1d, &(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedMalloc(nqpt1d, &(*basis)->qweight1d); CeedChk(ierr);
  ierr = CeedMalloc(nqpt1d*ndof1d, &(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedMalloc(nqpt1d*ndof1d, &(*basis)->grad1d); CeedChk(ierr);
  return 0;
}

int CeedBasisScalarTensorDestroy(CeedBasisScalarTensor *basis) {
  int ierr;
  if (!(*basis)) return 0;
  ierr = CeedFree(&(*basis)->grad1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qweight1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedFree(basis); CeedChk(ierr);
  return 0;
}

int CeedBasisCreate(Ceed ceed, CeedBasis *basis_ptr) {
  int ierr;
  ierr = CeedCalloc(1, basis_ptr); CeedChk(ierr);
  return 0;
}

int CeedBasisSetElement(CeedBasis basis, CeedTopology topo) {
  basis->topology = topo;
  return 0;
}

int CeedBasisSetType(CeedBasis basis, CeedBasisType btype, CeedInt degree,
                     CeedQuadMode nloc) {
  basis->btype = btype;
  basis->degree = degree;
  basis->node_locations = nloc;
  return 0;
}

int CeedBasisSetQuadrature(CeedBasis basis, CeedInt qorder,
                           CeedQuadMode qmode) {
  basis->qorder = qorder;
  basis->qmode = qmode;
  return 0;
}

/* This function completes the construction of a scalar, tensor-product basis.
   It expects that the following entries in basis are set.
   1. If host_basis == NULL:
     - ceed, dim, ncomp;
     - btype, degree, node_locations;
     - qorder, qmode.
   2. If host_basis != NULL:
     - ceed, dim, ncomp;
     - all entries of host_basis. */
static int CeedBasisCompleteScalarTensor(CeedBasis basis) {
  int ierr, i, j, k;
  CeedScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;
  CeedInt P, Q = 0;
  CeedBasisScalarTensor h_data = basis->host_data;
  Ceed ceed = basis->ceed;
  if (!ceed->BasisCreateScalarTensor)
    return CeedError(ceed, 1, "Backend does not support BasisCreateScalarTensor");
  switch (basis->dim) {
  case 1: basis->topology = CEED_LINE; break;
  case 2: basis->topology = CEED_QUAD; break;
  case 3: basis->topology = CEED_HEX; break;
  default: return CeedError(ceed, 1, "Invalid dimension: %d", basis->dim);
  }
  if (!h_data) {
    switch (basis->btype) {
    case CEED_BASIS_LAGRANGE:
      break;
    default: return CeedError(basis->ceed, 1, "Basis type not supported: %d",
                                basis->btype);
    }
    P = basis->degree + 1;
    switch (basis->qmode) {
    case CEED_GAUSS: Q = basis->qorder/2 + 1; break;
    case CEED_GAUSS_LOBATTO: Q = basis->qorder/2 + 2; break;
    default: return CeedError(ceed, 1, "CeedQuadMode not supported: %d",
                                basis->qmode);
    }
    ierr = CeedBasisScalarTensorCreate(P, Q, basis->dim, &h_data); CeedChk(ierr);
    basis->host_data = h_data;
    ierr = CeedCalloc(P, &nodes); CeedChk(ierr);
    qref1d = h_data->qref1d;
    qweight1d = h_data->qweight1d;
    interp1d = h_data->interp1d;
    grad1d = h_data->grad1d;
    // Get Nodes and Weights
    switch (basis->node_locations) {
    case CEED_GAUSS_LOBATTO:
      ierr = CeedLobattoQuadrature(P, nodes, NULL); CeedChk(ierr); break;
    default: return CeedError(ceed, 1, "Node locations not supported: %d",
                                basis->node_locations);
    }
    switch (basis->qmode) {
    case CEED_GAUSS:
      ierr = CeedGaussQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
      break;
    case CEED_GAUSS_LOBATTO:
      ierr = CeedLobattoQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
      break;
    default: break;
    }
    // Build B, D matrix
    // Fornberg, 1998
    for (i = 0; i < Q; i++) {
      c1 = 1.0;
      c3 = nodes[0] - qref1d[i];
      interp1d[i*P+0] = 1.0;
      for (j = 1; j < P; j++) {
        c2 = 1.0;
        c4 = c3;
        c3 = nodes[j] - qref1d[i];
        for (k = 0; k < j; k++) {
          dx = nodes[j] - nodes[k];
          c2 *= dx;
          if (k == j - 1) {
            grad1d[i*P + j] = c1*(interp1d[i*P + k] - c4*grad1d[i*P + k]) / c2;
            interp1d[i*P + j] = - c1*c4*interp1d[i*P + k] / c2;
          }
          grad1d[i*P + k] = (c3*grad1d[i*P + k] - interp1d[i*P + k]) / dx;
          interp1d[i*P + k] = c3*interp1d[i*P + k] / dx;
        }
        c1 = c2;
      }
    }
    ierr = CeedFree(&nodes); CeedChk(ierr);
  } else {
    basis->btype = CEED_BASIS_CUSTOM;
    basis->degree = h_data->P1d - 1;
    basis->node_locations = CEED_CUSTOM_QMODE;
    basis->qorder = -1; // unknown (custom) order
    basis->qmode = CEED_CUSTOM_QMODE;
  }
  // At this point, h_data == basis->host_data is initialized.
  basis->ndof = CeedPowInt(h_data->P1d, basis->dim);
  basis->nqpt = CeedPowInt(h_data->Q1d, basis->dim);
  basis->dtype = CEED_BASIS_SCALAR_TENSOR;
  return ceed->BasisCreateScalarTensor(basis, h_data);
}

int CeedBasisComplete(CeedBasis basis) {
  switch (basis->topology) {
  case CEED_POINT:
  case CEED_TRIANGLE:
  case CEED_TET:
    return CeedError(basis->ceed, 1, "Unsupported CeedTopology: %d",
                     basis->topology);
  case CEED_LINE:
  case CEED_QUAD:
  case CEED_HEX:
    basis->dim = CeedTopologyDimension[basis->topology];
    basis->ncomp = 1;
    return CeedBasisCompleteScalarTensor(basis);
  default:
    return CeedError(basis->ceed, 2, "Invalid CeedTopology: %d", basis->topology);
  }
  return 0;
}

int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ncomp, CeedInt P1d,
                            CeedInt Q1d, const CeedScalar *interp1d,
                            const CeedScalar *grad1d, const CeedScalar *qref1d,
                            const CeedScalar *qweight1d, CeedBasis *basis) {
  int ierr;
  ierr = CeedCalloc(1,basis); CeedChk(ierr);
  (*basis)->ceed = ceed;
  (*basis)->dim = dim;
  (*basis)->ncomp = ncomp;
  CeedBasisScalarTensor h_data;
  ierr = CeedBasisScalarTensorCreate(P1d, Q1d, dim, &h_data); CeedChk(ierr);
  memcpy(h_data->qref1d, qref1d, Q1d*sizeof(qref1d[0]));
  memcpy(h_data->qweight1d, qweight1d, Q1d*sizeof(qweight1d[0]));
  memcpy(h_data->interp1d, interp1d, Q1d*P1d*sizeof(interp1d[0]));
  memcpy(h_data->grad1d, grad1d, Q1d*P1d*sizeof(interp1d[0]));
  (*basis)->host_data = h_data;
  return CeedBasisCompleteScalarTensor(*basis);
}

int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                    CeedInt degree, CeedInt Q,
                                    CeedQuadMode qmode, CeedBasis *basis) {
  int ierr;
  ierr = CeedCalloc(1, basis); CeedChk(ierr);
  (*basis)->ceed = ceed;
  (*basis)->dim = dim;
  (*basis)->ncomp = ncomp;
  (*basis)->btype = CEED_BASIS_LAGRANGE;
  (*basis)->degree = degree;
  (*basis)->node_locations = CEED_GAUSS_LOBATTO;
  switch (qmode) {
  case CEED_GAUSS: (*basis)->qorder = 2*Q - 1; break;
  case CEED_GAUSS_LOBATTO: (*basis)->qorder = 2*Q - 3; break;
  default: return CeedError(ceed, 1, "Quadrature mode is not supported: %d",
                              qmode);
  }
  (*basis)->qmode = qmode;
  return CeedBasisCompleteScalarTensor(*basis);
}

int CeedGaussQuadrature(CeedInt Q, CeedScalar *qref1d, CeedScalar *qweight1d) {
  // Allocate
  CeedScalar P0, P1, P2, dP2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  for (int i = 0; i <= Q/2; i++) {
    // Guess
    xi = cos(PI*(CeedScalar)(2*i+1)/((CeedScalar)(2*Q)));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    for (int j = 2; j <= Q; j++) {
      P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
    xi = xi-P2/dP2;
    // Newton to convergence
    for (int k=0; k<100 && fabs(P2)>1e-15; k++) {
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j <= Q; j++) {
        P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
      xi = xi-P2/dP2;
    }
    // Save xi, wi
    wi = 2.0/((1.0-xi*xi)*dP2*dP2);
    qweight1d[i] = wi;
    qweight1d[Q-1-i] = wi;
    qref1d[i] = -xi;
    qref1d[Q-1-i]= xi;
  }
  return 0;
}

int CeedLobattoQuadrature(CeedInt Q, CeedScalar *qref1d,
                          CeedScalar *qweight1d) {
  // Allocate
  CeedScalar P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  // Set endpoints
  wi = 2.0/((CeedScalar)(Q*(Q-1)));
  if (qweight1d) {
    qweight1d[0] = wi;
    qweight1d[Q-1] = wi;
  }
  qref1d[0] = -1.0;
  qref1d[Q-1] = 1.0;
  // Interior
  for (int i = 1; i <= (Q-1)/2; i++) {
    // Guess
    xi = cos(PI*(CeedScalar)(i)/(CeedScalar)(Q-1));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    for (int j = 2; j < Q; j++) {
      P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton step
    dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
    d2P2 = (2*xi*dP2 - (CeedScalar)(Q*(Q-1))*P2)/(1.0-xi*xi);
    xi = xi-dP2/d2P2;
    // Newton to convergence
    for (int k=0; k<100 && fabs(dP2)>1e-15; k++) {
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j < Q; j++) {
        P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
      d2P2 = (2*xi*dP2 - (CeedScalar)(Q*(Q-1))*P2)/(1.0-xi*xi);
      xi = xi-dP2/d2P2;
    }
    // Save xi, wi
    wi = 2.0/(((CeedScalar)(Q*(Q-1)))*P2*P2);
    if (qweight1d) {
      qweight1d[i] = wi;
      qweight1d[Q-1-i] = wi;
    }
    qref1d[i] = -xi;
    qref1d[Q-1-i]= xi;
  }
  return 0;
}

int CeedBasisCreateScalarGeneric(Ceed ceed, CeedInt dim, CeedInt ndof,
                                 CeedInt nqpt, const CeedScalar *interp,
                                 const CeedScalar *grad,
                                 const CeedScalar *qweights, CeedBasis *basis) {
  // TODO: cleanup in case of error
  int ierr;
  if (!ceed->BasisCreateScalarGeneric) return CeedError(ceed, 1,
        "The backend does not support BasisCreateScalarGeneric");
  ierr = CeedCalloc(1,basis); CeedChk(ierr);
  (*basis)->ceed = ceed;
  // topology, btype, node_locations, qmode: zero -> "custom" type
  (*basis)->degree = -1; // unknown/custom degree
  (*basis)->qorder = -1; // unknown/custom quadrature order
  (*basis)->dim = dim;
  (*basis)->ndof = ndof;
  (*basis)->nqpt = nqpt;
  (*basis)->ncomp = 1;
  CeedBasisScalarGeneric h_data;
  ierr = CeedBasisScalarGenericCreate(ndof, nqpt, dim, &h_data); CeedChk(ierr);
  memcpy(h_data->interp, interp, nqpt*ndof*sizeof(interp[0]));
  memcpy(h_data->grad, grad, nqpt*dim*ndof*sizeof(grad[0]));
  memcpy(h_data->qweights, qweights, nqpt*sizeof(qweights[0]));
  (*basis)->dtype = CEED_BASIS_SCALAR_GENERIC;
  (*basis)->host_data = h_data;
  ierr = ceed->BasisCreateScalarGeneric(*basis, h_data); CeedChk(ierr);
  return 0;
}

static int CeedScalarView(const char *name, const char *fpformat, CeedInt m,
                          CeedInt n, const CeedScalar *a, FILE *stream) {
  for (int i=0; i<m; i++) {
    if (m > 1) fprintf(stream, "%12s[%d]:", name, i);
    else fprintf(stream, "%12s:", name);
    for (int j=0; j<n; j++) fprintf(stream, fpformat, a[i*n+j]);
    fputs("\n", stream);
  }
  return 0;
}

int CeedBasisView(CeedBasis basis, FILE *stream) {
  int ierr;
  switch (basis->dtype) {
  case CEED_BASIS_NO_DATA:
    fprintf(stream, "CeedBasis: (no host data)\n"); break;
  case CEED_BASIS_SCALAR_GENERIC: {
    CeedBasisScalarGeneric h_data = basis->host_data;
    fprintf(stream, "CeedBasis: (generic scalar basis)\n");
    fprintf(stream, "CeedBasis: dim=%d ndof=%d nqpt=%d\n", h_data->dim,
            h_data->ndof, h_data->nqpt);
    ierr = CeedScalarView("qweights", "\t% 12.8f", 1, h_data->nqpt,
                          h_data->qweights, stream); CeedChk(ierr);
    ierr = CeedScalarView("interp", "\t% 12.8f", h_data->ndof, h_data->nqpt,
                          h_data->interp, stream); CeedChk(ierr);
    ierr = CeedScalarView("grad", "\t% 12.8f", h_data->ndof*h_data->dim,
                          h_data->nqpt, h_data->grad, stream); CeedChk(ierr);
    break;
  }
  case CEED_BASIS_SCALAR_TENSOR: {
    CeedBasisScalarTensor h_data = basis->host_data;
    fprintf(stream, "CeedBasis: dim=%d P=%d Q=%d\n", h_data->dim, h_data->P1d,
            h_data->Q1d);
    ierr = CeedScalarView("qref1d", "\t% 12.8f", 1, h_data->Q1d, h_data->qref1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("qweight1d", "\t% 12.8f", 1, h_data->Q1d,
                          h_data->qweight1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("interp1d", "\t% 12.8f", h_data->Q1d, h_data->P1d,
                          h_data->interp1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("grad1d", "\t% 12.8f", h_data->Q1d, h_data->P1d,
                          h_data->grad1d, stream); CeedChk(ierr);
    break;
  }
  default: return CeedError(basis->ceed, 1, "Invalid CeedBasis");
  }
  return 0;
}

int CeedBasisApply(CeedBasis basis, CeedTransposeMode tmode, CeedEvalMode emode,
                   const CeedScalar *u, CeedScalar *v) {
  int ierr;
  if (!basis->Apply) return CeedError(basis->ceed, 1,
                                        "Backend does not support BasisApply");
  ierr = basis->Apply(basis, tmode, emode, u, v); CeedChk(ierr);
  return 0;
}

int CeedBasisDestroy(CeedBasis *basis) {
  int ierr;

  if (!(*basis)) return 0;
  if ((*basis)->Destroy) {
    ierr = (*basis)->Destroy(*basis); CeedChk(ierr);
  }
  switch ((*basis)->dtype) {
  case CEED_BASIS_NO_DATA:
    break;
  case CEED_BASIS_SCALAR_GENERIC: {
    CeedBasisScalarGeneric h_data = (*basis)->host_data;
    ierr = CeedBasisScalarGenericDestroy(&h_data); CeedChk(ierr);
    break;
  }
  case CEED_BASIS_SCALAR_TENSOR: {
    CeedBasisScalarTensor h_data = (*basis)->host_data;
    ierr = CeedBasisScalarTensorDestroy(&h_data); CeedChk(ierr);
    break;
  }
  default: return CeedError((*basis)->ceed, 1, "Invalid CeedBasis");
  }
  ierr = CeedFree(basis); CeedChk(ierr);
  return 0;
}
