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

/// @file
/// Implementation of public CeedBasis interfaces
///
/// @defgroup CeedBasis CeedBasis: fully discrete finite element-like objects
/// @{

/// Create a tensor product basis for H^1 discretizations
///
/// @param ceed   Ceed
/// @param dim    Topological dimension
/// @param ncomp  Number of field components (1 for scalar fields)
/// @param P1d    Number of nodes in one dimension
/// @param Q1d    Number of quadrature points in one dimension
/// @param interp1d Row-major Q1d × P1d matrix expressing the values of nodal
///               basis functions at quadrature points
/// @param grad1d  Row-major Q1d × P1d matrix expressing derivatives of nodal
///               basis functions at quadrature points
/// @param qref1d Array of length Q1d holding the locations of quadrature points
///               on the 1D reference element [-1, 1]
/// @param qweight1d Array of length Q1d holding the quadrature weights on the
///               reference element
/// @param[out] basis New basis
///
/// @sa CeedBasisCreateTensorH1Lagrange()
int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ncomp, CeedInt P1d,
                            CeedInt Q1d, const CeedScalar *interp1d,
                            const CeedScalar *grad1d, const CeedScalar *qref1d,
                            const CeedScalar *qweight1d, CeedBasis *basis) {
  int ierr;

  if (!ceed->BasisCreateTensorH1)
    return CeedError(ceed, 1, "Backend does not support BasisCreateTensorH1");
  ierr = CeedCalloc(1,basis); CeedChk(ierr);
  (*basis)->ceed = ceed;
  (*basis)->dim = dim;
  (*basis)->ndof = ncomp;
  (*basis)->P1d = P1d;
  (*basis)->Q1d = Q1d;
  ierr = CeedMalloc(Q1d,&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d,&(*basis)->qweight1d); CeedChk(ierr);
  memcpy((*basis)->qref1d, qref1d, Q1d*sizeof(qref1d[0]));
  memcpy((*basis)->qweight1d, qweight1d, Q1d*sizeof(qweight1d[0]));
  ierr = CeedMalloc(Q1d*P1d,&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*P1d,&(*basis)->grad1d); CeedChk(ierr);
  memcpy((*basis)->interp1d, interp1d, Q1d*P1d*sizeof(interp1d[0]));
  memcpy((*basis)->grad1d, grad1d, Q1d*P1d*sizeof(interp1d[0]));
  ierr = ceed->BasisCreateTensorH1(ceed, dim, P1d, Q1d, interp1d, grad1d, qref1d,
                                   qweight1d, *basis); CeedChk(ierr);
  return 0;
}

/// Create a tensor product Lagrange basis
///
/// @param ceed Ceed
/// @param dim Topological dimension of element
/// @param ncomp Number of field components
/// @param P Number of Gauss-Lobatto nodes in one dimension.  The polynomial degree
///     of the resulting Q_k element is k=P-1.
/// @param Q Number of quadrature points in one dimension.
/// @param qmode Distribution of the Q quadrature points (affects order of
///     accuracy for the quadrature)
/// @param[out] basis New basis
///
/// @sa CeedBasisCreateTensorH1()
int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                    CeedInt P, CeedInt Q,
                                    CeedQuadMode qmode, CeedBasis *basis) {
  // Allocate
  int ierr, i, j, k;
  CeedScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;
  ierr = CeedCalloc(P*Q, &interp1d); CeedChk(ierr);
  ierr = CeedCalloc(P*Q, &grad1d); CeedChk(ierr);
  ierr = CeedCalloc(P, &nodes); CeedChk(ierr);
  ierr = CeedCalloc(Q, &qref1d); CeedChk(ierr);
  ierr = CeedCalloc(Q, &qweight1d); CeedChk(ierr);
  // Get Nodes and Weights
  ierr = CeedLobattoQuadrature(P, nodes, NULL); CeedChk(ierr);
  switch (qmode) {
  case CEED_GAUSS:
    ierr = CeedGaussQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
    break;
  case CEED_GAUSS_LOBATTO:
    ierr = CeedLobattoQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
    break;
  }
  // Build B, D matrix
  // Fornberg, 1998
  for (i = 0; i  < Q; i++) {
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
  //  // Pass to CeedBasisCreateTensorH1
  ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp, P, Q, interp1d, grad1d, qref1d,
                                 qweight1d, basis); CeedChk(ierr);
  ierr = CeedFree(&interp1d); CeedChk(ierr);
  ierr = CeedFree(&grad1d); CeedChk(ierr);
  ierr = CeedFree(&nodes); CeedChk(ierr);
  ierr = CeedFree(&qref1d); CeedChk(ierr);
  ierr = CeedFree(&qweight1d); CeedChk(ierr);
  return 0;
}

/// Construct a Gauss-Legendre quadrature
///
/// @param Q Number of quadrature points (integrates polynomials of degree 2*Q-1 exactly)
/// @param qref1d Array of length Q to hold the abscissa on [-1, 1]
/// @param qweight1d Array of length Q to hold the weights
/// @sa CeedLobattoQuadrature()
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
    P2 = 0.0;
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

/// Construct a Gauss-Legendre-Lobatto quadrature
///
/// @param Q Number of quadrature points (integrates polynomials of degree 2*Q-3 exactly)
/// @param qref1d Array of length Q to hold the abscissa on [-1, 1]
/// @param qweight1d Array of length Q to hold the weights
/// @sa CeedGaussQuadrature()
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
    P2 = 0.0;
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

static int CeedScalarView(const char *name, const char *fpformat, CeedInt m,
                          CeedInt n, const CeedScalar *a, FILE *stream) {
  for (int i=0; i<m; i++) {
    if (m > 1) fprintf(stream, "%12s[%d]:", name, i);
    else fprintf(stream, "%12s:", name);
    for (int j=0; j<n; j++) {
      fprintf(stream, fpformat, fabs(a[i*n+j]) > 1E-14 ? a[i*n+j] : 0);
    }
    fputs("\n", stream);
  }
  return 0;
}

/// View a basis
///
/// @param basis Basis to view
/// @param stream Stream to view to, e.g., stdout
int CeedBasisView(CeedBasis basis, FILE *stream) {
  int ierr;

  fprintf(stream, "CeedBasis: dim=%d P=%d Q=%d\n", basis->dim, basis->P1d,
          basis->Q1d);
  ierr = CeedScalarView("qref1d", "\t% 12.8f", 1, basis->Q1d, basis->qref1d,
                        stream); CeedChk(ierr);
  ierr = CeedScalarView("qweight1d", "\t% 12.8f", 1, basis->Q1d, basis->qweight1d,
                        stream); CeedChk(ierr);
  ierr = CeedScalarView("interp1d", "\t% 12.8f", basis->Q1d, basis->P1d,
                        basis->interp1d, stream); CeedChk(ierr);
  ierr = CeedScalarView("grad1d", "\t% 12.8f", basis->Q1d, basis->P1d,
                        basis->grad1d, stream); CeedChk(ierr);
  return 0;
}

/// Apply basis evaluation from nodes to quadrature points or vice-versa
///
/// @param basis Basis to evaluate
/// @param tmode \ref CEED_NOTRANSPOSE to evaluate from nodes to quadrature
///     points, \ref CEED_TRANSPOSE to apply the transpose, mapping from
///     quadrature points to nodes
/// @param emode \ref CEED_EVAL_INTERP to obtain interpolated values,
///     \ref CEED_EVAL_GRAD to obtain gradients.
/// @param u input vector
/// @param v output vector
int CeedBasisApply(CeedBasis basis, CeedTransposeMode tmode, CeedEvalMode emode,
                   const CeedScalar *u, CeedScalar *v) {
  int ierr;
  if (!basis->Apply) return CeedError(basis->ceed, 1,
                                        "Backend does not support BasisApply");
  ierr = basis->Apply(basis, tmode, emode, u, v); CeedChk(ierr);
  return 0;
}

/// Get total number of nodes (in dim dimensions)
int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P) {
  *P = CeedPowInt(basis->P1d, basis->dim);
  return 0;
}

/// Get total number of quadrature points (in dim dimensions)
int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q) {
  *Q = CeedPowInt(basis->Q1d, basis->dim);
  return 0;
}

/// Destroy a CeedBasis
int CeedBasisDestroy(CeedBasis *basis) {
  int ierr;

  if (!*basis) return 0;
  if ((*basis)->Destroy) {
    ierr = (*basis)->Destroy(*basis); CeedChk(ierr);
  }
  ierr = CeedFree(&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->grad1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qweight1d); CeedChk(ierr);
  ierr = CeedFree(basis); CeedChk(ierr);
  return 0;
}
