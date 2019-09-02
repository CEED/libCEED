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
#include <ceed-backend.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @cond DOXYGEN_SKIP
static struct CeedBasis_private ceed_basis_collocated;
/// @endcond

/// @file
/// Implementation of public CeedBasis interfaces
///
/// @addtogroup CeedBasis
/// @{

/**
  @brief Create a tensor product basis for H^1 discretizations

  @param ceed       A Ceed object where the CeedBasis will be created
  @param dim        Topological dimension
  @param ncomp      Number of field components (1 for scalar fields)
  @param P1d        Number of nodes in one dimension
  @param Q1d        Number of quadrature points in one dimension
  @param interp1d   Row-major Q1d × P1d matrix expressing the values of nodal
                      basis functions at quadrature points
  @param grad1d     Row-major Q1d × P1d matrix expressing derivatives of nodal
                      basis functions at quadrature points
  @param qref1d     Array of length Q1d holding the locations of quadrature points
                      on the 1D reference element [-1, 1]
  @param qweight1d  Array of length Q1d holding the quadrature weights on the
                      reference element
  @param[out] basis Address of the variable where the newly created
                      CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ncomp, CeedInt P1d,
                            CeedInt Q1d, const CeedScalar *interp1d,
                            const CeedScalar *grad1d, const CeedScalar *qref1d,
                            const CeedScalar *qweight1d, CeedBasis *basis) {
  int ierr;

  if (dim<1)
    return CeedError(ceed, 1, "Basis dimension must be a positive value");

  if (!ceed->BasisCreateTensorH1) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Basis"); CeedChk(ierr);

    if (!delegate)
      return CeedError(ceed, 1, "Backend does not support BasisCreateTensorH1");

    ierr = CeedBasisCreateTensorH1(delegate, dim, ncomp, P1d,
                                   Q1d, interp1d, grad1d, qref1d,
                                   qweight1d, basis); CeedChk(ierr);
    return 0;
  }
  ierr = CeedCalloc(1,basis); CeedChk(ierr);
  (*basis)->ceed = ceed;
  ceed->refcount++;
  (*basis)->refcount = 1;
  (*basis)->tensorbasis = 1;
  (*basis)->dim = dim;
  (*basis)->ncomp = ncomp;
  (*basis)->P1d = P1d;
  (*basis)->Q1d = Q1d;
  (*basis)->P = CeedIntPow(P1d, dim);
  (*basis)->Q = CeedIntPow(Q1d, dim);
  ierr = CeedMalloc(Q1d,&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d,&(*basis)->qweight1d); CeedChk(ierr);
  memcpy((*basis)->qref1d, qref1d, Q1d*sizeof(qref1d[0]));
  memcpy((*basis)->qweight1d, qweight1d, Q1d*sizeof(qweight1d[0]));
  ierr = CeedMalloc(Q1d*P1d,&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*P1d,&(*basis)->grad1d); CeedChk(ierr);
  memcpy((*basis)->interp1d, interp1d, Q1d*P1d*sizeof(interp1d[0]));
  memcpy((*basis)->grad1d, grad1d, Q1d*P1d*sizeof(grad1d[0]));
  ierr = ceed->BasisCreateTensorH1(dim, P1d, Q1d, interp1d, grad1d, qref1d,
                                   qweight1d, *basis); CeedChk(ierr);
  return 0;
}

/**
  @brief Create a tensor product Lagrange basis

  @param ceed       A Ceed object where the CeedBasis will be created
  @param dim        Topological dimension of element
  @param ncomp      Number of field components
  @param P          Number of Gauss-Lobatto nodes in one dimension.  The
                      polynomial degree of the resulting Q_k element is k=P-1.
  @param Q          Number of quadrature points in one dimension.
  @param qmode      Distribution of the Q quadrature points (affects order of
                      accuracy for the quadrature)
  @param[out] basis Address of the variable where the newly created
                      CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                    CeedInt P, CeedInt Q,
                                    CeedQuadMode qmode, CeedBasis *basis) {
  // Allocate
  int ierr, i, j, k;
  CeedScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;

  if (dim<1)
    return CeedError(ceed, 1, "Basis dimension must be a positive value");

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

/**
  @brief Create a non tensor product basis for H^1 discretizations

  @param ceed       A Ceed object where the CeedBasis will be created
  @param topo       Topology of element, e.g. hypercube, simplex, ect
  @param ncomp      Number of field components (1 for scalar fields)
  @param nnodes       Total number of nodes
  @param nqpts      Total number of quadrature points
  @param interp     Row-major nqpts × nnodes matrix expressing the values of
                      nodal basis functions at quadrature points
  @param grad       Row-major (nqpts x dim) × nnodes matrix expressing
                      derivatives of nodal basis functions at quadrature points
  @param qref       Array of length nqpts holding the locations of quadrature
                      points on the reference element [-1, 1]
  @param qweight    Array of length nqpts holding the quadrature weights on the
                      reference element
  @param[out] basis Address of the variable where the newly created
                      CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisCreateH1(Ceed ceed, CeedElemTopology topo, CeedInt ncomp,
                      CeedInt nnodes, CeedInt nqpts,
                      const CeedScalar *interp,
                      const CeedScalar *grad, const CeedScalar *qref,
                      const CeedScalar *qweight, CeedBasis *basis) {
  int ierr;
  CeedInt P = nnodes, Q = nqpts, dim = 0;

  if (!ceed->BasisCreateH1) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Basis"); CeedChk(ierr);

    if (!delegate)
      return CeedError(ceed, 1, "Backend does not support BasisCreateH1");

    ierr = CeedBasisCreateH1(delegate, topo, ncomp, nnodes,
                             nqpts, interp, grad, qref,
                             qweight, basis); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1,basis); CeedChk(ierr);

  ierr = CeedBasisGetTopologyDimension(topo, &dim); CeedChk(ierr);

  (*basis)->ceed = ceed;
  ceed->refcount++;
  (*basis)->refcount = 1;
  (*basis)->tensorbasis = 0;
  (*basis)->dim = dim;
  (*basis)->ncomp = ncomp;
  (*basis)->P = P;
  (*basis)->Q = Q;
  ierr = CeedMalloc(Q*dim,&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedMalloc(Q,&(*basis)->qweight1d); CeedChk(ierr);
  memcpy((*basis)->qref1d, qref, Q*dim*sizeof(qref[0]));
  memcpy((*basis)->qweight1d, qweight, Q*sizeof(qweight[0]));
  ierr = CeedMalloc(Q*P,&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedMalloc(dim*Q*P,&(*basis)->grad1d); CeedChk(ierr);
  memcpy((*basis)->interp1d, interp, Q*P*sizeof(interp[0]));
  memcpy((*basis)->grad1d, grad, dim*Q*P*sizeof(grad[0]));
  ierr = ceed->BasisCreateH1(topo, dim, P, Q, interp, grad, qref,
                             qweight, *basis); CeedChk(ierr);
  return 0;
}

/**
  @brief Construct a Gauss-Legendre quadrature

  @param Q              Number of quadrature points (integrates polynomials of
                          degree 2*Q-1 exactly)
  @param[out] qref1d    Array of length Q to hold the abscissa on [-1, 1]
  @param[out] qweight1d Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
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

/**
  @brief Construct a Gauss-Legendre-Lobatto quadrature

  @param Q              Number of quadrature points (integrates polynomials of
                          degree 2*Q-3 exactly)
  @param[out] qref1d    Array of length Q to hold the abscissa on [-1, 1]
  @param[out] qweight1d Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
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

/**
  @brief View an array stored in a CeedBasis

  @param name      Name of array
  @param fpformat  Printing format
  @param m         Number of rows in array
  @param n         Number of columns in array
  @param a         Array to be viewed
  @param stream    Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
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

/**
  @brief View a CeedBasis

  @param basis  CeedBasis to view
  @param stream Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisView(CeedBasis basis, FILE *stream) {
  int ierr;

  if (basis->tensorbasis) {
    fprintf(stream, "CeedBasis: dim=%d P=%d Q=%d\n", basis->dim, basis->P1d,
            basis->Q1d);
    ierr = CeedScalarView("qref1d", "\t% 12.8f", 1, basis->Q1d, basis->qref1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("qweight1d", "\t% 12.8f", 1, basis->Q1d,
                          basis->qweight1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("interp1d", "\t% 12.8f", basis->Q1d, basis->P1d,
                          basis->interp1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("grad1d", "\t% 12.8f", basis->Q1d, basis->P1d,
                          basis->grad1d, stream); CeedChk(ierr);
  } else {
    fprintf(stream, "CeedBasis: dim=%d P=%d Q=%d\n", basis->dim, basis->P,
            basis->Q);
    ierr = CeedScalarView("qref", "\t% 12.8f", 1, basis->Q*basis->dim,
                          basis->qref1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("qweight", "\t% 12.8f", 1, basis->Q, basis->qweight1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("interp", "\t% 12.8f", basis->Q, basis->P,
                          basis->interp1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("grad", "\t% 12.8f", basis->dim*basis->Q, basis->P,
                          basis->grad1d, stream); CeedChk(ierr);
  }
  return 0;
}

/**
  @brief Compute Householder Reflection

    Computes A = (I - b v v^T) A
    where A is an mxn matrix indexed as A[i*row + j*col]

  @param[out] A  Matrix to apply Householder reflection to, in place
  @param v       Householder vector
  @param b       Scaling factor
  @param m       Number of rows in A
  @param n       Number of columns in A
  @param row     Col stride
  @param col     Row stride

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedHouseholderReflect(CeedScalar *A, const CeedScalar *v,
                                  CeedScalar b, CeedInt m, CeedInt n,
                                  CeedInt row, CeedInt col) {
  for (CeedInt j=0; j<n; j++) {
    CeedScalar w = A[0*row + j*col];
    for (CeedInt i=1; i<m; i++) w += v[i] * A[i*row + j*col];
    A[0*row + j*col] -= b * w;
    for (CeedInt i=1; i<m; i++) A[i*row + j*col] -= b * w * v[i];
  }
  return 0;
}

/**
  @brief Apply Householder Q matrix

    Compute A = Q A where Q is mxk and A is mxn. k<m

  @param[out] A  Matrix to apply Householder Q to, in place
  @param Q       Householder Q matrix
  @param tau     Householder scaling factors
  @param tmode   Transpose mode for application
  @param m       Number of rows in A
  @param n       Number of columns in A
  @param k       Index of row targeted
  @param row     Col stride
  @param col     Row stride

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedHouseholderApplyQ(CeedScalar *A, const CeedScalar *Q,
                                 const CeedScalar *tau, CeedTransposeMode tmode,
                                 CeedInt m, CeedInt n, CeedInt k,
                                 CeedInt row, CeedInt col) {
  CeedScalar v[m];
  for (CeedInt ii=0; ii<k; ii++) {
    CeedInt i = tmode == CEED_TRANSPOSE ? ii : k-1-ii;
    for (CeedInt j=i+1; j<m; j++) {
      v[j] = Q[j*k+i];
    }
    // Apply Householder reflector (I - tau v v^T) colograd1d^T
    CeedHouseholderReflect(&A[i*row], &v[i], tau[i], m-i, n, row, col);
  }
  return 0;
}

/**
  @brief Return QR Factorization of matrix

  @param[out] mat  Row-major matrix to be factorized in place
  @param[out] tau  Vector of length m of scaling factors
  @param m         Number of rows
  @param n         Number of columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedQRFactorization(Ceed ceed, CeedScalar *mat, CeedScalar *tau,
                        CeedInt m, CeedInt n) {
  CeedInt i, j;
  CeedScalar v[m];

  // Check m >= n
  if (n > m)
    return CeedError(ceed, 1, "Cannot compute QR factorization with n > m");

  for (i=0; i<n; i++) {
    // Calculate Householder vector, magnitude
    CeedScalar sigma = 0.0;
    v[i] = mat[i+n*i];
    for (j=i+1; j<m; j++) {
      v[j] = mat[i+n*j];
      sigma += v[j] * v[j];
    }
    CeedScalar norm = sqrt(v[i]*v[i] + sigma); // norm of v[i:m]
    CeedScalar Rii = -copysign(norm, v[i]);
    v[i] -= Rii;
    // norm of v[i:m] after modification above and scaling below
    //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
    //   tau = 2 / (norm*norm)
    tau[i] = 2 * v[i]*v[i] / (v[i]*v[i] + sigma);
    for (j=i+1; j<m; j++) v[j] /= v[i];

    // Apply Householder reflector to lower right panel
    CeedHouseholderReflect(&mat[i*n+i+1], &v[i], tau[i], m-i, n-i-1, n, 1);
    // Save v
    mat[i+n*i] = Rii;
    for (j=i+1; j<m; j++) {
      mat[i+n*j] = v[j];
    }
  }

  return 0;
}

/**
  @brief Return collocated grad matrix

  @param basis           CeedBasis
  @param[out] colograd1d Row-major Q1d × Q1d matrix expressing derivatives of
                           basis functions at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetCollocatedGrad(CeedBasis basis, CeedScalar *colograd1d) {
  int i, j, k;
  Ceed ceed;
  CeedInt ierr, P1d=(basis)->P1d, Q1d=(basis)->Q1d;
  CeedScalar *interp1d, *grad1d, tau[Q1d];

  ierr = CeedMalloc(Q1d*P1d, &interp1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*P1d, &grad1d); CeedChk(ierr);
  memcpy(interp1d, (basis)->interp1d, Q1d*P1d*sizeof(basis)->interp1d[0]);
  memcpy(grad1d, (basis)->grad1d, Q1d*P1d*sizeof(basis)->interp1d[0]);

  // QR Factorization, interp1d = Q R
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  ierr = CeedQRFactorization(ceed, interp1d, tau, Q1d, P1d); CeedChk(ierr);

  // Apply Rinv, colograd1d = grad1d Rinv
  for (i=0; i<Q1d; i++) { // Row i
    colograd1d[Q1d*i] = grad1d[P1d*i]/interp1d[0];
    for (j=1; j<P1d; j++) { // Column j
      colograd1d[j+Q1d*i] = grad1d[j+P1d*i];
      for (k=0; k<j; k++) {
        colograd1d[j+Q1d*i] -= interp1d[j+P1d*k]*colograd1d[k+Q1d*i];
      }
      colograd1d[j+Q1d*i] /= interp1d[j+P1d*j];
    }
    for (j=P1d; j<Q1d; j++) {
      colograd1d[j+Q1d*i] = 0;
    }
  }

  // Apply Qtranspose, colograd = colograd Qtranspose
  CeedHouseholderApplyQ(colograd1d, interp1d, tau, CEED_NOTRANSPOSE,
                        Q1d, Q1d, P1d, 1, Q1d);

  ierr = CeedFree(&interp1d); CeedChk(ierr);
  ierr = CeedFree(&grad1d); CeedChk(ierr);

  return 0;
}

/**
  @brief Apply basis evaluation from nodes to quadrature points or vice-versa

  @param basis  CeedBasis to evaluate
  @param nelem  The number of elements to apply the basis evaluation to;
                  the backend will specify the ordering in
                  ElemRestrictionCreateBlocked
  @param tmode  \ref CEED_NOTRANSPOSE to evaluate from nodes to quadrature
                  points, \ref CEED_TRANSPOSE to apply the transpose, mapping
                  from quadrature points to nodes
  @param emode  \ref CEED_EVAL_INTERP to obtain interpolated values,
                  \ref CEED_EVAL_GRAD to obtain gradients.
  @param[in] u  Input array
  @param[out] v Output array

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisApply(CeedBasis basis, CeedInt nelem, CeedTransposeMode tmode,
                   CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  CeedInt ulength = 0, vlength, nnodes, nqpt;
  if (!basis->Apply) return CeedError(basis->ceed, 1,
                                        "Backend does not support BasisApply");
  // check compatibility of topological and geometrical dimensions
  ierr = CeedBasisGetNumNodes(basis, &nnodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  ierr = CeedVectorGetLength(v, &vlength); CeedChk(ierr);

  if (u) {
    ierr = CeedVectorGetLength(u, &ulength); CeedChk(ierr);
  }

  if ((tmode == CEED_TRANSPOSE   && (vlength % nnodes != 0
                                     || ulength % nqpt != 0))
      ||
      (tmode == CEED_NOTRANSPOSE && (ulength % nnodes != 0 || vlength % nqpt != 0)))
    return CeedError(basis->ceed, 1,
                     "Length of input/output vectors incompatible with basis dimensions");

  ierr = basis->Apply(basis, nelem, tmode, emode, u, v); CeedChk(ierr);
  return 0;
}

/**
  @brief Get Ceed associated with a CeedBasis

  @param basis      CeedBasis
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetCeed(CeedBasis basis, Ceed *ceed) {
  *ceed = basis->ceed;

  return 0;
};

/**
  @brief Get dimension for given CeedBasis

  @param basis     CeedBasis
  @param[out] dim  Variable to store dimension of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetDimension(CeedBasis basis, CeedInt *dim) {
  *dim = basis->dim;

  return 0;
};

/**
  @brief Get tensor status for given CeedBasis

  @param basis        CeedBasis
  @param[out] tensor  Variable to store tensor status

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTensorStatus(CeedBasis basis, bool *tensor) {
  *tensor = basis->tensorbasis;

  return 0;
};

/**
  @brief Get number of components for given CeedBasis

  @param basis     CeedBasis
  @param[out] dim  Variable to store number of components of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumComponents(CeedBasis basis, CeedInt *numcomp) {
  *numcomp = basis->ncomp;

  return 0;
};

/**
  @brief Get total number of nodes (in 1 dimension) of a CeedBasis

  @param basis     CeedBasis
  @param[out] P1d  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumNodes1D(CeedBasis basis, CeedInt *P1d) {
  if (!basis->tensorbasis) return CeedError(basis->ceed, 1,
                                    "Cannot supply P1d for non-tensor basis");
  *P1d = basis->P1d;
  return 0;
}

/**
  @brief Get total number of quadrature points (in 1 dimension) of a CeedBasis

  @param basis     CeedBasis
  @param[out] Q1d  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumQuadraturePoints1D(CeedBasis basis, CeedInt *Q1d) {
  if (!basis->tensorbasis) return CeedError(basis->ceed, 1,
                                    "Cannot supply Q1d for non-tensor basis");
  *Q1d = basis->Q1d;
  return 0;
}

/**
  @brief Get total number of nodes (in dim dimensions) of a CeedBasis

  @param basis   CeedBasis
  @param[out] P  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P) {
  *P = basis->P;
  return 0;
}

/**
  @brief Get total number of quadrature points (in dim dimensions) of a CeedBasis

  @param basis   CeedBasis
  @param[out] Q  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q) {
  *Q = basis->Q;
  return 0;
}

/**
  @brief Get reference coordinates of quadrature points (in dim dimensions)
         of a CeedBasis

  @param basis      CeedBasis
  @param[out] qref  Variable to store reference coordinates of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQRef(CeedBasis basis, CeedScalar* *qref) {
  *qref = basis->qref1d;
  return 0;
}

/**
  @brief Get quadrature weights of quadrature points (in dim dimensions)
         of a CeedBasis

  @param basis         CeedBasis
  @param[out] qweight  Variable to store quadrature weights

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQWeights(CeedBasis basis, CeedScalar* *qweight) {
  *qweight = basis->qweight1d;
  return 0;
}

/**
  @brief Get interpolation matrix of a CeedBasis

  @param basis      CeedBasis
  @param[out] qref  Variable to store interpolation matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetInterp(CeedBasis basis, CeedScalar* *interp) {
  *interp = basis->interp1d;
  return 0;
}

/**
  @brief Get gradient matrix of a CeedBasis

  @param basis      CeedBasis
  @param[out] qref  Variable to store gradient matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetGrad(CeedBasis basis, CeedScalar* *grad) {
  *grad = basis->grad1d;
  return 0;
}

/**
  @brief Get backend data of a CeedBasis

  @param basis      CeedBasis
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetData(CeedBasis basis, void* *data) {
  *data = basis->data;
  return 0;
}

/**
  @brief Set backend data of a CeedBasis

  @param[out] basis CeedBasis
  @param data       Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisSetData(CeedBasis basis, void* *data) {
  basis->data = *data;
  return 0;
}

/**
  @brief Get CeedTensorContract of a CeedBasis

  @param basis          CeedBasis
  @param[out] contract  Variable to store CeedTensorContract

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTensorContract(CeedBasis basis,
                               CeedTensorContract *contract) {
  *contract = basis->contract;
  return 0;
}

/**
  @brief Set CeedTensorContract of a CeedBasis

  @param[out] basis     CeedBasis
  @param contract       CeedTensorContract to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisSetTensorContract(CeedBasis basis,
                               CeedTensorContract *contract) {
  basis->contract = *contract;
  return 0;
}

/**
  @brief Get dimension for given CeedElemTopology

  @param topo      CeedElemTopology
  @param[out] dim  Variable to store dimension of topology

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTopologyDimension(CeedElemTopology topo, CeedInt *dim) {
  *dim = (CeedInt) topo >> 16;

  return 0;
};

/**
  @brief Destroy a CeedBasis

  @param basis CeedBasis to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisDestroy(CeedBasis *basis) {
  int ierr;

  if (!*basis || --(*basis)->refcount > 0) return 0;
  if ((*basis)->Destroy) {
    ierr = (*basis)->Destroy(*basis); CeedChk(ierr);
  }
  ierr = CeedFree(&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->grad1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qweight1d); CeedChk(ierr);
  ierr = CeedDestroy(&(*basis)->ceed); CeedChk(ierr);
  ierr = CeedFree(basis); CeedChk(ierr);
  return 0;
}

/// @cond DOXYGEN_SKIP
// Indicate that the quadrature points are collocated with the nodes
CeedBasis CEED_BASIS_COLLOCATED = &ceed_basis_collocated;
/// @endcond
/// @}
